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
    prc_dpl_soc_high: float = 0.0
    prc_dpl_soc_low: float = 0.0


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
        # dictionary of optimizer variables
        self.variables = {}

        # Compute scaling for strategy control parameters
        self.min_import_price = np.min(self.time_series.p_N)
        self.max_import_price = np.max(self.time_series.p_N)

        # scaling for penalty parameters. Make sure goal_penalty is always positive
        self.prc_e_goal_pen = np.min([self.max_import_price, 0.1e-3]) * 10e1
        self.prc_p_goal_pen = np.min([self.max_import_price, 0.1e-3]) * np.max(self.time_series.dt) / 3600 * 10e1
        self.prc_soc_exc_pen = np.min([self.max_import_price, 0.1e-3]) * 10e2

        # penalty for exceeding grid import limit. Result shall not become infeasible but report the violation
        # with helpful information
        self.prc_e_grid_imp_pen = np.min([self.max_import_price, 0.1e-3]) * 10e1
        # penalty for exceeding the grid export limit. Result shall not become infeasible but report the 'lost'
        # solar power
        self.prc_e_grid_exp_pen = np.min([self.max_import_price, 0.1e-3]) * 10e1

        # if there is a demand rate given in the input, the grid import limit will be interpreted as the
        # threshold beyond wich the demand rate is to be applied. Compute a demand rate flag for use in the
        # build constraint and build objective methods.
        self.is_grid_demand_rate_active = False
        if self.grid.p_max_imp is not None and self.grid.prc_p_exc_imp is not None:
            self.is_grid_demand_rate_active = True

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
        for i, bat in enumerate(self.batteries):
            if bat.p_demand is not None:
                for t in self.time_steps:
                    self.variables['p_demand_pen'][i][t] = pulp.LpVariable(f"p_demand_pen_{i}_{t}", lowBound=0)
                    self.variables['z_p_demand'][i][t] = pulp.LpVariable(f"z_p_demand_{i}_{t}", cat='Binary')

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

        # Cost variable to charge for battery depletion at very high and very low battery SOC state
        self.variables['cst_bat_dpl'] = {}
        for i, bat in enumerate(self.batteries):
            self.variables['cst_bat_dpl'][i] = [
                pulp.LpVariable(f"cst_bat_dpl_{i}_{t}", lowBound=0)
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

        # cost for depleting battery life by sitting at very high or very low SOC
        for i, bat in enumerate(self.batteries):
            for t in self.time_steps:
                objective += - self.variables['cst_bat_dpl'][i][t]

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

        # prefer charging at high solar production times to unload public grid from peaks
        if self.strategy.charging_strategy == 'attenuate_grid_peaks':
            for i, bat in enumerate(self.batteries):
                for t in self.time_steps:
                    objective += self.variables['c'][i][t] * self.time_series.ft[t] * self.min_import_price * 1e-6

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

        self.problem += objective

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

        # Constraints (4)-(5): Grid flow direction
        for t in self.time_steps:
            # Export constraint
            self.problem += self.variables['e'][t] <= self.M * self.variables['y'][t]
            # Import constraint
            self.problem += self.variables['n'][t] <= self.M * (1 - self.variables['y'][t])

        # limit regular grid import power
        if self.grid.p_max_imp is not None:
            if self.is_grid_demand_rate_active:
                # limit the demand rate free portion of the power
                for t in self.time_steps:
                    self.problem += self.variables['n'][t] <= self.grid.p_max_imp * self.time_series.dt[t] / 3600
                    self.problem += (self.grid.p_max_imp * self.time_series.dt[t] / 3600 - self.variables['n'][t]
                                     <= self.M * self.variables['z_imp_lim'][t])
                    self.problem += (self.variables['e_imp_lim_exc'][t]
                                     <= self.M * (1 - self.variables['z_imp_lim'][t]))
            else:
                # limit the actual import power
                for t in self.time_steps:
                    self.problem += self.variables['n'][t] <= self.grid.p_max_imp * self.time_series.dt[t] / 3600
                    self.problem += (self.grid.p_max_imp * self.time_series.dt[t] / 3600 - self.variables['n'][t]
                                     <= self.M * self.variables['z_imp_lim'][t])
                    self.problem += (self.variables['e_imp_lim_exc'][t]
                                     <= self.M * (1 - self.variables['z_imp_lim'][t]))

        # limit regular grid export power
        if self.grid.p_max_exp is not None:
            for t in self.time_steps:
                self.problem += self.variables['e'][t] <= self.grid.p_max_exp * self.time_series.dt[t] / 3600
                self.problem += (self.grid.p_max_exp * self.time_series.dt[t] / 3600 - self.variables['e'][t]
                                 <= self.M * self.variables['z_exp_lim'][t])
                self.problem += (self.variables['e_exp_lim_exc'][t]
                                 <= self.M * (1 - self.variables['z_exp_lim'][t]))

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
                        # clip required charge to max charging power if needed
                        # and leave some air to breathe for the optimizer
                        p_demand = min(bat.c_max * self.time_series.dt[t] / 3600., bat.p_demand[t])
                        # two alternative constraints, only one is active:
                        # constraint option 1: charge energy tries to reach min charge energy parameter
                        self.problem += (self.variables['c'][i][t] + self.variables['p_demand_pen'][i][t]
                                         + self.M * self.variables['z_p_demand'][i][t] >= p_demand)
                        # constraint option 2: charge energy tries to reach energy to fill the battery to s_max
                        self.problem += (self.variables['c'][i][t] + self.variables['p_demand_pen'][i][t]
                                         + self.M * (1 - self.variables['z_p_demand'][i][t])
                                         - (self.batteries[i].s_max - self.variables['s'][i][t]) >= 0.)
                    elif bat.c_min > 0:
                        # in time steps without given charging demand, apply normal lower bound:
                        # Lower bound: either 0 or at least c_min
                        self.problem += (self.variables['c'][i][t] >= bat.c_min * self.time_series.dt[t] / 3600.
                                         * self.variables['z_c'][i][t])
                        self.problem += (self.variables['c'][i][t] <= self.M * self.variables['z_c'][i][t])

            # Constraint (7): Minimum charge power limits if there is not charge demand
            elif bat.c_min > 0:
                for t in self.time_steps:
                    # Lower bound: either 0 or at least c_min
                    self.problem += (self.variables['c'][i][t] >= bat.c_min * self.time_series.dt[t] / 3600.
                                     * self.variables['z_c'][i][t])
                    self.problem += (self.variables['c'][i][t] <= self.M * self.variables['z_c'][i][t])

            # control battery charging from grid
            if not bat.charge_from_grid:
                for t in self.time_steps:
                    self.problem += (self.variables['c'][i][t] <= self.M * self.variables['y'][t])

            # control battery discharging to grid
            if not bat.discharge_to_grid:
                for t in self.time_steps:
                    self.problem += (self.variables['d'][i][t] <= self.M * (1 - self.variables['y'][t]))

            # lock charging against discharging
            for t in self.time_steps:
                # Discharge constraint
                self.problem += self.variables['d'][i][t] <= self.M * self.variables['z_cd'][i][t]
                # Charge constraint
                self.problem += self.variables['c'][i][t] <= self.M * (1 - self.variables['z_cd'][i][t])

            # cost for depleting battery life by sitting at very high or very low SOC
            # figure out battery capacity, avoid div 0 in case s_amx is set to 0
            bat_capacity = self.batteries[i].s_max
            if self.batteries[i].s_capacity is not None and self.batteries[i].s_capacity > 0.0:
                bat_capacity = self.batteries[i].s_capacity
            bat_capacity = max(bat_capacity, 1.0)
            for t in self.time_steps:
                # scale the prices to the time step width
                prc_dpl_soc_high = self.batteries[i].prc_dpl_soc_high / 3600.0 * self.time_series.dt[t]
                prc_dpl_soc_low = self.batteries[i].prc_dpl_soc_low / 3600.0 * self.time_series.dt[t]
                # for all SOCs
                self.problem += self.variables['cst_bat_dpl'][i][t] >= 0
                # for high SOCs, linear cost ramp starting at 80% SOC, full cost at 100%
                self.problem += self.variables['cst_bat_dpl'][i][t] \
                    - ((prc_dpl_soc_high / (0.2 * bat_capacity)) * (self.variables['s'][i][t] - 0.8 * bat_capacity)) >= 0
                # for low SOCs, linear cost ramp starting at 20% SOC, full cost at 0%
                self.problem += self.variables['cst_bat_dpl'][i][t] \
                    - ((prc_dpl_soc_low) * (1.0 - (self.variables['s'][i][t] / (0.2 * bat_capacity)))) >= 0

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

        # Extract results
        status = pulp.LpStatus[self.problem.status]

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

        if status == 'Optimal':
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
