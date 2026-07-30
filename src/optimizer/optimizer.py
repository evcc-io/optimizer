from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional

import numpy as np
import pulp

from .settings import OptimizerSettings


@dataclass
class OptimizationStrategy:
    charging_strategy: str
    discharging_strategy: str


# charging strategies that level grid peaks, mapped to the metered sides they level.
# 'imp' is the demand side (grid import), 'exp' the feed-in side (grid export).
PEAK_STRATEGY_SIDES = {
    'attenuate_demand_peaks': ('imp',),
    'attenuate_feedin_peaks': ('exp',),
    'attenuate_grid_peaks': ('imp', 'exp'),
}

# magnitude the largest objective coefficient is placed at before the model goes to the solver.
# CBC judges improvements against absolute tolerances (~1e-7), and with prices given per Wh the
# raw coefficients land close to that bound, so real improvements get pruned as numerical noise.
# Scaling does not change the argmax, it only moves the window. The reported objective value is
# recalculated from the solution and stays in its original unit.
OBJECTIVE_TARGET = 1e6

# fixed factor to use instead of deriving one from the coefficients. None derives it, which is the
# production setting; the tests pin it to compare two scalings of the same model.
OBJECTIVE_SCALE = None


def objective_scale(objective) -> float:
    """
    Factor that puts the largest objective coefficient at OBJECTIVE_TARGET.

    Derived from the coefficients rather than fixed, because their magnitude is set by the request:
    prices per Wh, a demand rate per W, the penalty base they scale from. Across the stored cases
    the largest coefficient moves by three orders, from 1.6e-1 where market prices go negative to
    3e2 on the peak levelling ones, so a single constant lands the same model anywhere in that
    range and a fixed 1e6 pushes six of nineteen past 1e6 in the model handed to the solver.

    Anchored on the largest coefficient and not on the smallest: the smallest is a strategy tie
    breaker, deliberately tiny, and aiming that one at a floor drags the whole objective down with
    it. Measured, that costs 3 requests of 335 their solution and runs others three times past the
    time limit, because the objective ends up orders below a constraint matrix that carries a big M
    of 1e6. The span within a model is up to 1e8 and no single factor can fix that, only the model.
    """
    if OBJECTIVE_SCALE is not None:
        return OBJECTIVE_SCALE
    coefficients = [abs(c) for c in pulp.LpAffineExpression(objective).values() if c]
    if not coefficients:
        return 1.0
    return OBJECTIVE_TARGET / max(coefficients)


# grid energy variable and limit exceedance variable per leveled side
PEAK_SIDE_VARIABLES = {
    'imp': ('n', 'e_imp_lim_exc'),
    'exp': ('e', 'e_exp_lim_exc'),
}


@dataclass
class GridConfig:
    p_max_imp: float
    p_max_exp: float
    prc_p_exc_imp: float


@dataclass
class BatteryConfig:
    charge_from_grid: bool
    discharge_to_grid: bool
    s_capacity: float
    s_min: float
    s_max: float
    s_initial: float
    c_min: float
    c_max: float
    d_max: float
    p_a: float
    p_demand: Optional[List[float]] = None  # Minimum charge demand (Wh)
    s_goal: Optional[List[float]] = None  # Goal state of charge (Wh)
    c_priority: int = 0


@dataclass
class TimeSeriesData:
    dt: List[int]  # time step length [s]
    gt: List[float]  # Required total energy [Wh]
    ft: List[float]  # Forecasted production [Wh]
    p_N: List[float]  # Import prices [currency unit/Wh]
    p_E: List[float]  # Export prices [currency unit/Wh]


class Optimizer:
    """
    Optimizer class building the MILP model from the input data, and provides
    solve() function to run optimization and return the results
    """

    def __init__(self, strategy: OptimizationStrategy, grid: GridConfig, batteries: List[BatteryConfig], time_series: TimeSeriesData,
                 eta_c: float = 0.95, eta_d: float = 0.95, M: float = 1e6, optimizer_settings: OptimizerSettings | None = None):
        """
        Optimizer Constructor
        """

        self.settings = optimizer_settings or OptimizerSettings()

        self.strategy = strategy
        self.grid = grid
        self.batteries = batteries
        self.time_series = time_series
        self.eta_c = eta_c
        self.eta_d = eta_d
        self.M = M
        # number of time steps
        self.T = len(time_series.gt)
        # time step range
        self.time_steps = range(self.T)
        # the optimization problem
        self.problem = None
        # factor the objective went to the solver with, set by _setup_target_function
        self.objective_scale = None
        # dictionary of optimizer variables
        self.variables = {}

        # with a demand rate given, the grid import limit is the threshold beyond which the rate
        # applies. Needed by the penalty scaling below as well as by the constraints and objective.
        self.is_grid_demand_rate_active = (self.grid.p_max_imp is not None
                                           and self.grid.prc_p_exc_imp is not None)

        # Compute scaling for strategy control parameters
        self.min_import_price = np.min(self.time_series.p_N)
        self.max_import_price = np.max(self.time_series.p_N)
        self.max_export_price = np.max(self.time_series.p_E)

        # scaling base for penalty parameters, derived from the largest real currency/Wh rate
        # any economic term in the model can command, so that the avoidance penalties below
        # (which are all multiples of penalty_base applied to Wh-denominated violations) stay
        # above real cost trade-offs regardless of the price data. Make sure it's always positive.
        real_prices_per_wh = [
            self.max_import_price,
            self.max_export_price,
            *(bat.p_a for bat in self.batteries),
        ]
        if self.is_grid_demand_rate_active:
            # prc_p_exc_imp is currency/W and the charge is set by the single worst time step, so
            # 1 Wh kept out of the binding step is worth 3600/dt[t] W of it. Converting over the
            # shortest step keeps the penalties above that incentive everywhere, converting over the
            # horizon does not: a violation confined to one short step would outweigh them.
            real_prices_per_wh.append(self.grid.prc_p_exc_imp * 3600. / min(self.time_series.dt))
        self.penalty_base = np.max(real_prices_per_wh + [0.1e-3])

        # scaling for penalty parameters
        self.prc_e_goal_pen = self.penalty_base * 10e1
        self.prc_p_goal_pen = self.penalty_base * 10e1
        self.prc_soc_exc_pen = self.penalty_base * 10e2

        # penalty for exceeding grid import limit. Result shall not become infeasible but report the violation
        # with helpful information
        self.prc_e_grid_imp_pen = self.penalty_base * 10e1
        # penalty for exceeding the grid export limit. Result shall not become infeasible but report the 'lost'
        # solar power
        self.prc_e_grid_exp_pen = self.penalty_base * 10e1

        # weights of the grid peak leveling strategies. capping the horizon maximum alone leaves the
        # profile below the cap arbitrary, so the step to step ramp is penalized as well. The ramp
        # weight is the smaller one, so lowering the peak wins wherever the two disagree.
        self.prc_p_peak = self.penalty_base * 1e-3
        self.prc_p_ramp = self.penalty_base * 1e-5

        # weight the peak leveling strategies defer grid export at, so that a leveled profile
        # still fills the batteries before it sends the surplus out. Two orders below the ramp
        # weight, so leveling decides wherever the two disagree and this only picks between the
        # schedules leveling rates equal.
        self.prc_e_early = self.penalty_base * 1e-7

        # grid sides leveled by the active peak attenuation strategy, empty for all other strategies
        self.peak_sides = PEAK_STRATEGY_SIDES.get(strategy.charging_strategy, ())

    def create_model(self):
        """
        Create and initialize the MILP model
        """

        # Create problem
        self.problem = pulp.LpProblem("EV_Charging_Optimization", pulp.LpMaximize)

        self._setup_variables()
        self._setup_target_function()
        self._add_energy_balance_constraints()
        self._add_battery_constraints()

    def _setup_variables(self):
        """
        Set up the variables of the MILP optimizer
        """

        # Charging power variables [Wh]
        self.variables['c'] = {}
        for i, bat in enumerate(self.batteries):
            self.variables['c'][i] = [
                pulp.LpVariable(f"c_{i}_{t}", lowBound=0, upBound=bat.c_max * self.time_series.dt[t] / 3600.)
                for t in self.time_steps
            ]

        # Discharging power variables [Wh]
        self.variables['d'] = {}
        for i, bat in enumerate(self.batteries):
            self.variables['d'][i] = [
                pulp.LpVariable(f"d_{i}_{t}", lowBound=0, upBound=bat.d_max * self.time_series.dt[t] / 3600.)
                for t in self.time_steps
            ]

        # State of charge variables [Wh]
        self.variables['s'] = {}
        for i, bat in enumerate(self.batteries):
            self.variables['s'][i] = [
                pulp.LpVariable(f"s_{i}_{t}", lowBound=0, upBound=bat.s_capacity)
                for t in self.time_steps
            ]

        # penalty variable for not reaching given charge goals
        # variables are kept in a matrix Batteries X time steps, only those elements will have an
        # entry != None that have a SOC goal > 0 defined in the input data
        self.variables['s_goal_pen'] = [[None for t in self.time_steps] for i in range(len(self.batteries))]
        for i, bat in enumerate(self.batteries):
            if self.batteries[i].s_goal is not None:
                for t in self.time_steps:
                    if self.batteries[i].s_goal[t] > 0:
                        self.variables['s_goal_pen'][i][t] = pulp.LpVariable(f"s_goal_pen_{i}_{t}", lowBound=0)

        # penalty variable for not being able to charge with the required power
        self.variables['p_demand_pen'] = [[None for t in self.time_steps] for i in range(len(self.batteries))]
        # binary variable to allow one out of two alternative constraints
        self.variables['z_p_demand'] = [[None for t in self.time_steps] for i in range(len(self.batteries))]
        # binary variable to capture when s_max is reached
        self.variables['z_s_max_reached'] = [[None for t in self.time_steps] for i in range(len(self.batteries))]
        for i, bat in enumerate(self.batteries):
            if bat.p_demand is not None:
                for t in self.time_steps:
                    self.variables['p_demand_pen'][i][t] = pulp.LpVariable(f"p_demand_pen_{i}_{t}", lowBound=0)
                    self.variables['z_p_demand'][i][t] = pulp.LpVariable(f"z_p_demand_{i}_{t}", cat='Binary')
                    self.variables['z_s_max_reached'][i][t] = pulp.LpVariable(f"z_s_max_reached_{i}_{t}", cat='Binary')

        # penalty variable for staying above max SOC and below min SOC
        self.variables['s_max_pen'] = [[pulp.LpVariable(f"s_max_pen_{i}_{t}", lowBound=0) for t in self.time_steps] for i in range(len(self.batteries))]
        self.variables['s_min_pen'] = [[pulp.LpVariable(f"s_min_pen_{i}_{t}", lowBound=0) for t in self.time_steps] for i in range(len(self.batteries))]

        # Grid import/export variables [Wh]
        self.variables['n'] = [pulp.LpVariable(f"n_{t}", lowBound=0) for t in self.time_steps]
        self.variables['e'] = [pulp.LpVariable(f"e_{t}", lowBound=0) for t in self.time_steps]

        # penalty variables for exceeding grid power limits (W)
        # for grid import
        if self.grid.p_max_imp is not None:
            self.variables['e_imp_lim_exc'] = [pulp.LpVariable(f"p_imp_pen_{t}", lowBound=0) for t in self.time_steps]
            # binary variable to allow limit exceeding only if the regular import actually hits the limit
            # this is required to avoid that limit exceeds are shifted to other time steps
            self.variables['z_imp_lim'] = [pulp.LpVariable(f"z_imp_lim_{t}", cat='Binary') for t in self.time_steps]

        # for grid export
        if self.grid.p_max_exp is not None:
            self.variables['e_exp_lim_exc'] = [pulp.LpVariable(f"e_exp_lim_exc_{t}", lowBound=0) for t in self.time_steps]
            # binary variable to allow limit exceeding only if the regular export actually hits the limit
            self.variables['z_exp_lim'] = [pulp.LpVariable(f"z_exp_lim_{t}", cat='Binary') for t in self.time_steps]

        # for demand rate calculation, we need to track the actual maximum import power
        # within the time horizon (W)
        if self.is_grid_demand_rate_active:
            self.variables['p_max_imp_exc'] = pulp.LpVariable("p_max_imp_exc", lowBound=0)

        # highest grid power over the whole horizon (W) and step to step ramp of the grid power (W)
        # per side, used by the peak attenuation strategies. there is no ramp into the first time
        # step, so the ramp of step t is held at index t - 1
        for side in self.peak_sides:
            self.variables[f'p_{side}_peak'] = pulp.LpVariable(f"p_{side}_peak", lowBound=0)
            self.variables[f'p_{side}_ramp'] = [
                pulp.LpVariable(f"p_{side}_ramp_{t}", lowBound=0)
                for t in range(1, self.T)
            ]

        # Binary variable: power flow direction to / from grid variables
        # these variables
        # 1. avoid direct export from import if export remuneration is greater than import cost
        # 2. control grid charging to batteries and grid export from batteries acc. to configuration
        self.variables['y'] = []
        for t in self.time_steps:
            self.variables['y'].append(pulp.LpVariable(f"y_{t}", cat='Binary'))

        # Binary variable for charging activation
        self.variables['z_c'] = {}
        for i, bat in enumerate(self.batteries):
            if bat.c_min > 0:
                self.variables['z_c'][i] = [
                    pulp.LpVariable(f"z_c_{i}_{t}", cat='Binary')
                    for t in self.time_steps
                ]
            else:
                self.variables['z_c'][i] = None

        # Binary variable to lock charging against discharging
        self.variables['z_cd'] = {}
        for i, bat in enumerate(self.batteries):
            self.variables['z_cd'][i] = [
                pulp.LpVariable(f"z_cd_{i}_{t}", cat='Binary')
                for t in self.time_steps
            ]

    def _setup_target_function(self):
        """
        Gather all target function contributions and instantiate the objective
        """

        # Objective function (1): Maximize economic benefit
        objective = 0

        ############################################################################
        # actual cost & benefit elements

        # Grid import cost (negative because we want to minimize cost) [currency unit]
        for t in self.time_steps:
            # if a demand rate beyond p_max_imp is applied, both portions have to be considered
            # for energy cost. If only an import limit is given, there should never be power
            # import beyond p_max_imp, however, if the limit gets violated, we account for
            # the energy cost as well to stay consistent.
            if self.grid.p_max_imp is not None:
                objective -= (
                    # grid import up to the demand rate threshold
                    self.variables['n'][t]
                    # import beyond the threshold
                    + self.variables['e_imp_lim_exc'][t]
                ) * self.time_series.p_N[t]
            else:
                # standard case
                objective -= self.variables['n'][t] * self.time_series.p_N[t]

        # Grid export revenue [currency unit]
        for t in self.time_steps:
            objective += self.variables['e'][t] * self.time_series.p_E[t]

        # Final state of charge value [currency unit]
        for i, bat in enumerate(self.batteries):
            objective += self.variables['s'][i][-1] * bat.p_a

        # charge for import power demand rate. The demand rate is applied to the maximum
        # power draw beyond the threshold within the time horizon.
        if self.is_grid_demand_rate_active:
            objective += - self.grid.prc_p_exc_imp * self.variables['p_max_imp_exc']

        ############################################################################
        # Penalties for exceeding battery SOC limits at start
        for i, bat in enumerate(self.batteries):
            for t in self.time_steps:
                objective += - self.prc_soc_exc_pen * (self.variables['s_max_pen'][i][t] + self.variables['s_min_pen'][i][t])

        ############################################################################
        # Penalties for goals that cannot be met
        for i, bat in enumerate(self.batteries):
            # unmet battery charging goals
            if self.batteries[i].s_goal is not None:
                for t in self.time_steps:
                    if self.batteries[i].s_goal[t] > 0:
                        # negative target function contribution in a maximizing optimization
                        objective += - self.prc_e_goal_pen * self.variables['s_goal_pen'][i][t]
            # unmet charging demand due to battery reaching maximum SOC with incentive to do charging early
            if bat.p_demand is not None:
                for t in self.time_steps:
                    objective += - self.prc_p_goal_pen \
                        * self.variables['p_demand_pen'][i][t] \
                        * (1 + (self.T - t)/self.T)

        # penalties for grid power limits that cannot be met.
        for t in self.time_steps:

            # penalty for exceeding the given import limit
            if self.grid.p_max_imp is not None and not self.is_grid_demand_rate_active:
                # negative target function contribution in a maximizing optimization
                objective += - self.prc_e_grid_imp_pen * self.variables['e_imp_lim_exc'][t]

            # penalty for exceeding the grid export limit
            if self.grid.p_max_exp is not None:
                # negative target function contribution in a maximizing optimization
                # decrease penalty slightly over time to push limit exceeding to late times
                objective += - self.prc_e_grid_exp_pen * (1.0 - t * 1e-5) * self.variables['e_exp_lim_exc'][t]

        #############################################################################
        # Secondary strategies to implement preferences without impact to actual cost

        # prefer charging first, then grid export
        if self.strategy.charging_strategy == 'charge_before_export':
            for i, bat in enumerate(self.batteries):
                for t in self.time_steps:
                    objective += - self.variables['e'][t] * self.min_import_price * 2e-5 * (self.T - t)

        # level the grid profile to unload the public grid from peaks. attenuate_demand_peaks levels
        # grid import, attenuate_feedin_peaks levels grid export, attenuate_grid_peaks levels both.
        # the penalty sits on the horizon maximum and on the step to step ramp instead of on charge
        # power, so the optimizer spreads charging at partial power over several time steps rather
        # than running one step at full power, and keeps the profile below the cap leveled too.
        # penalty_base is used instead of min_import_price because negative market prices would turn
        # this penalty into a reward for peaks.
        for side in self.peak_sides:
            objective += - self.variables[f'p_{side}_peak'] * self.prc_p_peak
            objective += - pulp.lpSum(self.variables[f'p_{side}_ramp']) * self.prc_p_ramp

        # peak and ramp say how high the grid profile may go, not when the batteries fill, and
        # under flat commercials that leaves most of the horizon undecided: the schedule then comes
        # back as arbitrary as with no strategy at all, with the stored examples charging in the
        # last third of the day. So the peak strategies defer export as well, the same tie break
        # charge_before_export makes, at prc_e_early instead of its weight so that leveling keeps
        # the last word.
        if self.peak_sides:
            for t in self.time_steps:
                objective += - self.variables['e'][t] * self.prc_e_early * (self.T - t) / self.T

        # prefer discharging batteries completely before importing from grid
        if self.strategy.discharging_strategy == 'discharge_before_import':
            for i, bat in enumerate(self.batteries):
                for t in self.time_steps:
                    objective += - self.variables['n'][t] * self.min_import_price * 5e-6 * (self.T - t)

        # charging and discharging priorities
        for i, bat in enumerate(self.batteries):
            for t in self.time_steps:
                objective += self.variables['c'][i][t] * self.min_import_price * 5e-5 * (self.T - t) * bat.c_priority
                objective += self.variables['d'][i][t] * self.min_import_price * 5e-5 * (self.T - t) * bat.c_priority

        self.objective_scale = objective_scale(objective)
        self.problem += objective * self.objective_scale

    def _add_energy_balance_constraints(self):
        """
        Add constraints related to the energy balance to the model.
        """

        # Constraint (2): Power balance for each time step:
        # - solar yield
        # - household consumption
        # - net charge and discharge of each battery
        # - grid import
        # - grid export
        for t in self.time_steps:
            # battery charge + discharge balance
            battery_net_discharge = 0
            for i, bat in enumerate(self.batteries):
                battery_net_discharge += (- self.variables['c'][i][t]
                                          + self.variables['d'][i][t])

            # grid import: if there is an import power limit, the power exceeding the limit
            # is going to the penalty variable. If a demand rate is active, it is applied
            # to power drawn beyond the p_max_imp threshold.
            e_grid_imp = self.variables['n'][t]
            if self.grid.p_max_imp is not None:
                if self.is_grid_demand_rate_active:
                    # demand rate calculation
                    e_grid_imp = self.variables['n'][t]+self.variables['e_imp_lim_exc'][t]
                else:
                    # grid import power limit
                    e_grid_imp = self.variables['n'][t]+self.variables['e_imp_lim_exc'][t]

            # grid export: if there is a limit, the power exceeding the limit
            # is going to the penalty variable
            e_grid_exp = self.variables['e'][t]
            if self.grid.p_max_exp is not None:
                e_grid_exp = self.variables['e'][t]+self.variables['e_exp_lim_exc'][t]

            self.problem += (battery_net_discharge
                             + self.time_series.ft[t]
                             + e_grid_imp
                             == e_grid_exp
                             + self.time_series.gt[t])

        # Constraints (4)-(5): Grid flow direction. Export/import have natural
        # per-step caps (export: solar plus discharge-to-grid capacity; import:
        # demand plus grid-charge capacity) that tighten the LP relaxation vs
        # the global big-M without excluding any integer point. When a grid
        # limit is active, the opposite side's unbounded excess variable enters
        # the energy balance, so only then fall back to the global big-M.
        cap_d_exp = sum(b.d_max for b in self.batteries if b.discharge_to_grid)
        cap_c_imp = sum(b.c_max for b in self.batteries if b.charge_from_grid)
        for t in self.time_steps:
            dth = self.time_series.dt[t] / 3600.
            m_exp = self.time_series.ft[t] + cap_d_exp * dth if self.grid.p_max_imp is None else self.M
            m_imp = self.time_series.gt[t] + cap_c_imp * dth if self.grid.p_max_exp is None else self.M
            # Export constraint
            self.problem += self.variables['e'][t] <= m_exp * self.variables['y'][t]
            # Import constraint
            self.problem += self.variables['n'][t] <= m_imp * (1 - self.variables['y'][t])

        # limit regular grid import power
        if self.grid.p_max_imp is not None:
            if self.is_grid_demand_rate_active:
                # limit the demand rate free portion of the power
                for t in self.time_steps:
                    self.problem += self.variables['n'][t] <= self.grid.p_max_imp * self.time_series.dt[t] / 3600
                    self.problem += (self.grid.p_max_imp * self.time_series.dt[t] / 3600 - self.variables['n'][t]
                                     <= self.grid.p_max_imp * self.time_series.dt[t] / 3600
                                     * self.variables['z_imp_lim'][t])
                    self.problem += (self.variables['e_imp_lim_exc'][t]
                                     <= self.M * (1 - self.variables['z_imp_lim'][t]))
            else:
                # limit the actual import power
                for t in self.time_steps:
                    self.problem += self.variables['n'][t] <= self.grid.p_max_imp * self.time_series.dt[t] / 3600
                    self.problem += (self.grid.p_max_imp * self.time_series.dt[t] / 3600 - self.variables['n'][t]
                                     <= self.grid.p_max_imp * self.time_series.dt[t] / 3600
                                     * self.variables['z_imp_lim'][t])
                    self.problem += (self.variables['e_imp_lim_exc'][t]
                                     <= self.M * (1 - self.variables['z_imp_lim'][t]))

        # limit regular grid export power
        if self.grid.p_max_exp is not None:
            for t in self.time_steps:
                self.problem += self.variables['e'][t] <= self.grid.p_max_exp * self.time_series.dt[t] / 3600
                self.problem += (self.grid.p_max_exp * self.time_series.dt[t] / 3600 - self.variables['e'][t]
                                 <= self.grid.p_max_exp * self.time_series.dt[t] / 3600
                                 * self.variables['z_exp_lim'][t])
                self.problem += (self.variables['e_exp_lim_exc'][t]
                                 <= self.M * (1 - self.variables['z_exp_lim'][t]))

        # track the horizon maximum and the step to step ramp of the total grid power for every side
        # the strategy levels. Both include the portion beyond p_max_imp / p_max_exp, so they stay
        # correct in demand rate mode and when a limit is violated.
        for side in self.peak_sides:
            grid_var, lim_exc_var = PEAK_SIDE_VARIABLES[side]
            # total grid power of this side per time step (W)
            p_grid = []
            for t in self.time_steps:
                e_grid = self.variables[grid_var][t]
                if lim_exc_var in self.variables:
                    e_grid = e_grid + self.variables[lim_exc_var][t]
                p_grid.append(e_grid * 3600 / self.time_series.dt[t])

            for t in self.time_steps:
                self.problem += p_grid[t] <= self.variables[f'p_{side}_peak']
            # ramp magnitude: p_ramp[t - 1] >= |p_grid[t] - p_grid[t-1]|
            for t in range(1, self.T):
                self.problem += self.variables[f'p_{side}_ramp'][t - 1] >= p_grid[t] - p_grid[t - 1]
                self.problem += self.variables[f'p_{side}_ramp'][t - 1] >= p_grid[t - 1] - p_grid[t]

        # if demand rate is applied, the maximum grid import power value
        # of all time steps drives the demand rate charge
        if self.is_grid_demand_rate_active:
            for t in self.time_steps:
                self.problem += self.variables['e_imp_lim_exc'][t] \
                    <= self.variables['p_max_imp_exc'] * self.time_series.dt[t] / 3600

    def _add_battery_constraints(self):
        """
        Add constraints related to battery behavior to the model.
        """
        # constraint for the max and min SOC. If the battery starts with an initial SOC
        # greater than the maximum SOC or lesser than min SOC, maximum discharging is forced until the max.
        # SOC is reached or max. charing will be forced until min SOC is reached.
        for i, bat in enumerate(self.batteries):
            for t in range(0, self.T):
                self.problem += (self.variables['s_max_pen'][i][t] >= self.variables['s'][i][t] - bat.s_max)
                self.problem += (self.variables['s_min_pen'][i][t] >= bat.s_min - self.variables['s'][i][t])

        # Constraint (3): Battery dynamics
        for i, bat in enumerate(self.batteries):
            # Initial state of charge
            if len(self.time_steps) > 0:
                self.problem += (self.variables['s'][i][0]
                                 == bat.s_initial
                                 + self.eta_c * self.variables['c'][i][0]
                                 - (1 / self.eta_d) * self.variables['d'][i][0])

            # State of charge evolution
            for t in range(1, self.T):
                self.problem += (self.variables['s'][i][t]
                                 == self.variables['s'][i][t - 1]
                                 + self.eta_c * self.variables['c'][i][t]
                                 - (1 / self.eta_d) * self.variables['d'][i][t])

            # Constraint (6): Battery SOC goal constraints (for t > 0)
            if bat.s_goal is not None:
                for t in range(1, self.T):
                    if bat.s_goal[t] > 0:
                        self.problem += (self.variables['s'][i][t]
                                         + self.variables['s_goal_pen'][i][t] >= bat.s_goal[t])

            # Constraint: Minimum battery charge demand (for t > 0)
            if bat.p_demand is not None:
                for t in self.time_steps:
                    if bat.p_demand[t] > 0:
                        # clip requested charging power to max charging power if needed
                        p_demand = min(bat.c_max * self.time_series.dt[t] / 3600., bat.p_demand[t])

                        # introduce z_p_demand to become only 1 if s_max is almost reached and the
                        # charging with c_min would stop before the end of the time slot
                        # s is bounded below by 0, so s_max is the largest the left side can get
                        self.problem += ((bat.s_max - self.variables['s'][i][t])
                                         <= (1 - self.variables['z_p_demand'][i][t]) * bat.s_max
                                         + bat.c_min * self.time_series.dt[t] / 3600.)

                        # introduce z_s_max_reached to become only 1 if s_max is fully reached
                        self.problem += ((bat.s_max - self.variables['s'][i][t])
                                         <= (1 - self.variables['z_s_max_reached'][i][t]) * bat.s_max)

                        # set a "soft" constraint to reach the requested charging rate if possible.
                        # deactivate it if s_max is already reached
                        self.problem += (self.variables['c'][i][t] + self.variables['p_demand_pen'][i][t]
                                         >= (1 - self.variables['z_s_max_reached'][i][t]) * p_demand)
                    else:
                        # if there is no p_demand set, make sure z_p_demand is 0 to keep the below c_min constraint effective
                        self.problem += (self.variables['z_p_demand'][i][t] <= 0)

                    if bat.c_min > 0:
                        # set constraints to exclude charging between 0 and c_min if z_p_demand is not 1.
                        self.problem += (self.variables['c'][i][t]
                                         + bat.c_min * self.time_series.dt[t] / 3600. * self.variables['z_p_demand'][i][t]
                                         >= bat.c_min * self.time_series.dt[t] / 3600. * self.variables['z_c'][i][t])
                        self.problem += (self.variables['c'][i][t] <= bat.c_max * self.time_series.dt[t] / 3600.
                                         * self.variables['z_c'][i][t])

            # Constraint (7): Minimum charge power limits if there is no charge demand
            elif bat.c_min > 0:
                for t in self.time_steps:
                    # Lower bound: either 0 or at least c_min
                    self.problem += (self.variables['c'][i][t] >= bat.c_min * self.time_series.dt[t] / 3600.
                                     * self.variables['z_c'][i][t])
                    self.problem += (self.variables['c'][i][t] <= bat.c_max * self.time_series.dt[t] / 3600.
                                     * self.variables['z_c'][i][t])

            # control battery charging from grid
            if not bat.charge_from_grid:
                for t in self.time_steps:
                    self.problem += (self.variables['c'][i][t] <= bat.c_max * self.time_series.dt[t] / 3600.
                                     * self.variables['y'][t])

            # control battery discharging to grid
            if not bat.discharge_to_grid:
                for t in self.time_steps:
                    self.problem += (self.variables['d'][i][t] <= bat.d_max * self.time_series.dt[t] / 3600.
                                     * (1 - self.variables['y'][t]))

            # lock charging against discharging
            for t in self.time_steps:
                # Discharge constraint
                self.problem += (self.variables['d'][i][t] <= bat.d_max * self.time_series.dt[t] / 3600.
                                 * self.variables['z_cd'][i][t])
                # Charge constraint
                self.problem += (self.variables['c'][i][t] <= bat.c_max * self.time_series.dt[t] / 3600.
                                 * (1 - self.variables['z_cd'][i][t]))

    def solve(self) -> Dict:
        """
        Creates the MILP model if none exists and solves the optimization problem.
        Returns a dictionary with the optimization results
        """

        if self.problem is None:
            self.create_model()

        # Solve the problem
        solver = pulp.PULP_CBC_CMD(
            msg=0,
            threads=self.settings.num_threads,
            timeLimit=self.settings.time_limit,
        )
        with TemporaryDirectory() as tmpdir:
            solver.tmpDir = tmpdir
            self.problem.solve(solver)

        # Extract results.
        #
        # pulp reports LpStatusOptimal whenever CBC came back with any feasible solution, including
        # one it stopped on at the time limit, so status alone cannot tell a proved schedule from a
        # truncated one. Measured on a captured request, a 2 s run and a 30 s run both said Optimal
        # with objective values of -682466848 and 59714881. sol_status carries the distinction and
        # is folded into the reported status here rather than exposed as a second field: callers
        # already branch on this one, and 'Optimal' claiming more than it can back is the bug.
        status = pulp.LpStatus[self.problem.status]
        if status == 'Optimal' and self.problem.sol_status != pulp.LpSolutionOptimal:
            status = 'Feasible'

        # grid import and export if no demand rate is active
        # if a limit is set and exceeded, this is the part that is actually imported / exported.
        # the exceeding portion is captured in 'e_imp_lim_exc' and / or 'e_exp_lim_exc'
        e_grid_import = [pulp.value(var) for var in self.variables['n']]
        e_grid_export = [pulp.value(var) for var in self.variables['e']]
        # if a demand rate is active, the actual import power is both parts, 'n' and 'e_imp_lim_exc'
        if self.is_grid_demand_rate_active:
            for t in self.time_steps:
                e_grid_import[t] += pulp.value(self.variables['e_imp_lim_exc'][t])

        # get limit violations
        # grid import limit
        grid_imp_limit_violated = False
        e_grid_imp_overshoot = []
        if self.grid.p_max_imp is not None:
            grid_imp_limit_violated = (np.max([pulp.value(var) for var in self.variables['e_imp_lim_exc']]) > 0)
            e_grid_imp_overshoot = [pulp.value(var) for var in self.variables['e_imp_lim_exc']]
        # grid export limit
        grid_exp_limit_hit = False
        e_grid_exp_overshoot = []
        if self.grid.p_max_exp is not None:
            grid_exp_limit_hit = (np.max([pulp.value(var) for var in self.variables['e_exp_lim_exc']]) > 0)
            e_grid_exp_overshoot = [pulp.value(var) for var in self.variables['e_exp_lim_exc']]

        # a Feasible solve carries a full schedule, it just is not proved, so it is returned like
        # an optimal one. Only the label changes.
        if status in ('Optimal', 'Feasible'):
            result = {
                'status': status,
                'objective_value': self.get_clean_objective_value(),
                'limit_violations': {
                    'grid_import_limit_exceeded': grid_imp_limit_violated,
                    'grid_export_limit_hit': grid_exp_limit_hit
                },
                'batteries': [],
                'grid_import': e_grid_import,
                'grid_export': e_grid_export,
                'flow_direction': [],
                'grid_import_overshoot': e_grid_imp_overshoot,
                'grid_export_overshoot': e_grid_exp_overshoot
            }

            # Extract battery results
            for i, bat in enumerate(self.batteries):
                battery_result = {
                    'charging_power': [pulp.value(var) for var in self.variables['c'][i]],
                    'discharging_power': [pulp.value(var) for var in self.variables['d'][i]],
                    'state_of_charge': [pulp.value(var) for var in self.variables['s'][i]]
                }
                result['batteries'].append(battery_result)

            # Extract flow direction
            for y_var in self.variables['y']:
                if y_var is not None:
                    result['flow_direction'].append(int(pulp.value(y_var)))
                else:
                    result['flow_direction'].append(0)  # Default to import when constraint not active

            return result
        else:
            return {
                'status': status,
                'objective_value': None,
                'limit_violations': {
                    'grid_import_limit_exceeded': False,
                    'grid_export_limit_hit': False
                },
                'batteries': [],
                'grid_import': [],
                'grid_export': [],
                'flow_direction': [],
                'grid_import_overshoot': [],
                'grid_export_overshoot': []
            }

    def get_clean_objective_value(self):
        '''
        recalculate the objective value without penalties and strategy icentives
        '''
        clean_objective = 0
        # Grid import cost (negative because we want to minimize cost) [currency unit]
        for t in self.time_steps:
            if self.grid.p_max_imp is not None:
                clean_objective -= (
                    # grid import up to the demand rate threshold
                    pulp.value(self.variables['n'][t])
                    # import beyond the threshold
                    + pulp.value(self.variables['e_imp_lim_exc'][t])
                ) * self.time_series.p_N[t]
            else:
                # standard case
                clean_objective -= pulp.value(self.variables['n'][t]) * self.time_series.p_N[t]

        # Grid export revenue [currency unit]
        for t in self.time_steps:
            clean_objective += pulp.value(self.variables['e'][t]) * self.time_series.p_E[t]

        # Final state of charge value [currency unit]
        for i, bat in enumerate(self.batteries):
            clean_objective += (pulp.value(self.variables['s'][i][self.T-1])
                                - pulp.value(self.variables['s'][i][0])) * bat.p_a

        # charge for import power demand rate. The demand rate is applied to the maximum
        # power draw beyond the threshold within the time horizon.
        if self.is_grid_demand_rate_active:
            clean_objective += - self.grid.prc_p_exc_imp \
                * pulp.value(self.variables['p_max_imp_exc'])

        return clean_objective
