from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional

import numpy as np
import pulp

from .settings import OptimizerSettings


@dataclass
class OptimizationStrategy:
    """
    Defines how the optimizer should prefer charging and discharging.

    These are secondary objectives (tiny weights in the objective function)
    that break ties when the primary economic outcome is equivalent.

    Attributes:
        charging_strategy: 'charge_before_export' prefers filling batteries
            before exporting surplus to the grid. 'attenuate_grid_peaks'
            prefers charging when solar production is high.
            'none' applies no charging preference.
        discharging_strategy: 'discharge_before_import' prefers draining
            batteries before importing from the grid.
            'none' applies no discharging preference.
    """
    charging_strategy: str
    discharging_strategy: str


@dataclass
class GridConfig:
    """
    Grid connection configuration.

    Attributes:
        p_max_imp: Maximum grid import power [W]. None = unlimited.
            When prc_p_exc_imp is also set, this becomes the demand rate
            threshold instead of a hard limit.
        p_max_exp: Maximum grid export power [W]. None = unlimited.
        prc_p_exc_imp: Demand rate price [currency unit/W] applied to the
            peak power draw above p_max_imp. None = no demand rate.
    """
    p_max_imp: float
    p_max_exp: float
    prc_p_exc_imp: float


@dataclass
class BatteryConfig:
    """
    Configuration for a single battery (home storage or EV).

    Naming convention: 's' = state/storage (Wh), 'c' = charge, 'd' = discharge,
    'p' = price/power.

    Attributes:
        charge_from_grid: If False, battery can only charge from solar surplus
            (charging is blocked when grid flow direction is import).
        discharge_to_grid: If False, battery can only discharge to cover local
            consumption (discharging is blocked when grid flow direction is export).
        s_capacity: Total physical battery capacity [Wh]. Used as upper bound
            for SoC variables and as reference for solar_hold_pct.
        s_min: Minimum usable state of charge [Wh]. The optimizer uses a soft
            penalty to keep SoC above this value.
        s_max: Maximum usable state of charge [Wh]. The optimizer uses a soft
            penalty to keep SoC below this value. Typically s_max < s_capacity
            to preserve battery health (e.g. 95% of capacity).
        s_initial: Current state of charge at the start of optimization [Wh].
            May exceed s_max (e.g. battery was charged to 100% before
            optimization started).
        c_min: Minimum charge power [W]. When > 0, the battery must either
            charge at >= c_min or not charge at all (on/off behavior via
            binary variable z_c).
        c_max: Maximum charge power [W].
        d_max: Maximum discharge power [W]. Set to 0 for batteries that
            cannot discharge (e.g. EVs that only accept charge).
        p_a: Monetary value of stored energy [currency unit/Wh]. Applied to
            the final SoC to incentivize ending with a charged battery.
            Also represents battery aging cost per cycle.
        p_demand: Per-timestep minimum charge demand [Wh]. Used for EVs with
            a minimum charge target. None = no demand.
        s_goal: Per-timestep SoC goals [Wh]. Soft targets -- the optimizer
            tries to reach them but uses a penalty if it cannot. None = no goals.
        c_priority: Charging/discharging priority 0..2 relative to other
            batteries. Higher values get charged first.
    """
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
    p_demand: Optional[List[float]] = None
    s_goal: Optional[List[float]] = None
    c_priority: int = 0


@dataclass
class TimeSeriesData:
    """
    Time-discretized input data for the optimization horizon.

    Each list has T elements (one per timestep). Timesteps can have variable
    duration (dt[t] in seconds). Values represent energy [Wh] or power rates
    [W] for each timestep.

    Attributes:
        dt: Duration of each timestep [s]. Typically 900 (15 min). The first
            timestep may be shorter (partial interval from current time to
            the next 15-min boundary).
        gt: Forecasted total energy consumption per timestep [Wh]. Includes
            all household loads but excludes battery and EV charging.
        ft: Forecasted solar production per timestep [Wh]. Used to determine
            solar surplus (ft - gt) and to detect solar windows for the
            solar hold feature.
        p_N: Grid import price per timestep [currency unit/Wh]. Higher values
            incentivize battery discharge to avoid imports.
        p_E: Grid export price per timestep [currency unit/Wh]. Typically
            much lower than p_N (feed-in tariff vs retail price).
    """
    dt: List[int]
    gt: List[float]
    ft: List[float]
    p_N: List[float]
    p_E: List[float]


class Optimizer:
    """
    Mixed Integer Linear Programming (MILP) optimizer for battery scheduling.

    Builds and solves a MILP model that maximizes economic benefit by
    scheduling battery charging/discharging over a time horizon of T
    timesteps. The model considers grid import/export prices, battery
    aging costs, and configurable charging/discharging strategies.

    MILP variable naming convention (indices: i=battery, t=timestep):
        c[i][t]    - Charging energy [Wh] for battery i at timestep t
        d[i][t]    - Discharging energy [Wh] for battery i at timestep t
        s[i][t]    - State of charge [Wh] for battery i at end of timestep t
        n[t]       - Grid import energy [Wh] at timestep t
        e[t]       - Grid export energy [Wh] at timestep t
        y[t]       - Grid flow direction binary: 1=export, 0=import
        z_cd[i][t] - Charge/discharge lock binary: 1=discharging, 0=charging
        z_c[i][t]  - Charge activation binary (only if c_min > 0): 1=charging on

    Penalty variables (soft constraints that keep the model feasible):
        s_max_pen[i][t]    - SoC exceeding s_max [Wh]
        s_min_pen[i][t]    - SoC below s_min [Wh]
        s_goal_pen[i][t]   - Unmet SoC goal [Wh]
        p_demand_pen[i][t] - Unmet charge demand [Wh]
        e_imp_lim_exc[t]   - Grid import exceeding p_max_imp [Wh]
        e_exp_lim_exc[t]   - Grid export exceeding p_max_exp [Wh]

    Penalty weight naming convention (prc_ prefix = "price of"):
        prc_soc_exc_pen  - Weight for SoC limit violations (strongest)
        prc_e_goal_pen   - Weight for energy goal violations
        prc_p_goal_pen   - Weight for power demand violations
        prc_e_grid_imp_pen - Weight for grid import limit violations
        prc_e_grid_exp_pen - Weight for grid export limit violations
    """

    def __init__(self, strategy: OptimizationStrategy, grid: GridConfig, batteries: List[BatteryConfig], time_series: TimeSeriesData,
                 eta_c: float = 0.95, eta_d: float = 0.95, M: float = 1e6, optimizer_settings: OptimizerSettings | None = None):
        """
        Initialize the optimizer with input data and compute derived parameters.

        Args:
            strategy: Charging/discharging preference strategy.
            grid: Grid connection limits and demand rate configuration.
            batteries: List of battery configurations to optimize.
            time_series: Time-discretized forecast and price data.
            eta_c: Charging efficiency (0..1). Energy stored = eta_c * energy input.
            eta_d: Discharging efficiency (0..1). Energy output = eta_d * energy removed.
            M: Big-M constant for linearizing binary constraints. Must be larger
                than any variable value in the model.
            optimizer_settings: Server-side settings (thread count, time limit,
                default solar_hold_pct).
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
