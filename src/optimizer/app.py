import fcntl
import json
import os
import pathlib
import time
from datetime import datetime, timezone

import jwt
from flask import Flask, jsonify, request
from flask_restx import Api, Resource, fields
from werkzeug.exceptions import BadRequest

from .optimizer import CHARGING_STRATEGIES, DISCHARGING_STRATEGIES, BatteryConfig, GridConfig, OptimizationStrategy, Optimizer, TimeSeriesData
from .settings import OptimizerSettings

app = Flask(__name__)
settings = OptimizerSettings()


@app.before_request
def before_request_func():
    secret_key = os.environ.get('JWT_TOKEN_SECRET')
    if secret_key:
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({"message": "Missing authorization header"}), 401

        try:
            token_type, token = auth_header.split(' ')
            if token_type.lower() != 'bearer':
                return jsonify({"message": "Invalid token type"}), 401

            payload = jwt.decode(token, secret_key, algorithms=["HS256"])
            if settings.log_subject:
                print("subject:", payload.get('sub'))
        except jwt.ExpiredSignatureError:
            return jsonify({"message": "Token has expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"message": "Invalid token"}), 401
        except Exception as e:
            return jsonify({"message": str(e)}), 401


def dump_slow_request(payload, elapsed):
    """Persist requests that exhausted the solver time limit, they are the ones worth replaying.

    The elapsed time covers model building as well as solving, so a request that only exceeds
    the limit while building is caught too. That one is equally worth looking at.
    """
    path, limit = settings.dump_slow_requests, settings.time_limit
    if not path or limit is None or elapsed < limit:
        return

    # one line per request, carrying the same "request" key as test_cases/*.json so a line
    # can be replayed by the existing harness
    line = json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "elapsed": round(elapsed, 3), "request": payload}) + "\n"
    try:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            # every gunicorn worker appends to the same file, and a request is far larger than
            # the buffer size that would make the append atomic on its own
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(line)
    except OSError as e:
        # a full or read only disk must never turn a solved request into an error
        print("could not dump slow request:", e)


api = Api(app, version='1.0', title='EV Charging Optimization API',
          description='Mixed Integer Linear Programming model for EV charging optimization',
          validate=True)


@api.errorhandler(BadRequest)
def handle_validation_error(error):
    """Return JSON and log the cause without logging rejected request values."""
    data = getattr(error, 'data', None)
    reason = data.get('message', error.description).partition(':')[0] if data else 'Invalid request body'
    error.data = data or {'message': error.description}
    if 'errors' in error.data:
        error.data['details'] = error.data.pop('errors')

    print(json.dumps({'bad_request': {
        'path': request.path,
        'reason': reason,
        'fields': sorted(error.data.get('details', {})),
        'validator': getattr(error.__context__, 'validator', None),
    }}), flush=True)
    return error.data, 400


# Namespace for the API
ns = api.namespace('optimize', description='EV Charging Optimization Operations')

# Input models for API documentation
strategy_model = api.model('OptimizationStrategy', {
    'charging_strategy': fields.String(required=False, enum=list(CHARGING_STRATEGIES),
                                       description='Sets a strategy for charging in situations where choices are cost neutral.'),
    'discharging_strategy': fields.String(required=False, enum=list(DISCHARGING_STRATEGIES),
                                          description='Sets a strategy for discharging in situations where choices are cost neutral.')
})

grid_model = api.model('GridConfig', {
    'p_max_imp': fields.Float(required=False, description='Maximum grid import power in W'),
    'p_max_exp': fields.Float(required=False, description='Maximum grid export power in W'),
    'prc_p_exc_imp': fields.Float(required=False, description='price per W to consider in case the import limit is exceeded. ')
})

battery_config_model = api.model('BatteryConfig', {
    'charge_from_grid': fields.Boolean(required=False, description='Controls whether the battery can be charged from the grid.'),
    'discharge_to_grid': fields.Boolean(required=False, description='Controls whether the battery can discharge to grid.'),
    's_capacity': fields.Float(required=False, description='Capacity at 100% state of charge (Wh)'),
    's_min': fields.Float(required=True, description='Minimum state of charge (Wh)'),
    's_max': fields.Float(required=True, description='Maximum state of charge (Wh)'),
    's_initial': fields.Float(required=True, description='Initial state of charge (Wh)'),
    'p_demand': fields.List(fields.Float, required=False, description='Minimum charge demand per time step (Wh)'),
    's_goal': fields.List(fields.Float, required=False, description='Goal state of charge at each time step (Wh)'),
    'c_min': fields.Float(required=True, description='Minimum charge power (W)'),
    'c_max': fields.Float(required=True, description='Maximum charge power (W)'),
    'd_max': fields.Float(required=True, description='Maximum discharge power (W)'),
    'p_a': fields.Float(required=True, description='Monetary value per Wh at end of the optimization horizon'),
    'c_priority': fields.Integer(required=False, description='Charging and discharging priority compared to other batteries. 2 = highest priority.')
})

time_series_model = api.model('TimeSeries', {
    'dt': fields.List(fields.Float, required=True, description='duration in seconds for each time step (s)'),
    'gt': fields.List(fields.Float, required=True, description='Required energy for home consumption at each time step (Wh)'),
    'ft': fields.List(fields.Float, required=True, description='Forecasted solar generation at each time step (Wh)'),
    'p_N': fields.List(fields.Float, required=True, description='Price per Wh taken from grid at each time step'),
    'p_E': fields.List(fields.Float, required=True, description='Remuneration per Wh fed into grid at each time step'),
})

optimization_input_model = api.model('OptimizationInput', {
    'strategy': fields.Nested(strategy_model, required=False, description='Optimization strategy'),
    'grid': fields.Nested(grid_model, required=False, description='Grid import and export configuration'),
    'batteries': fields.List(fields.Nested(battery_config_model), required=True, description='Battery configurations'),
    'time_series': fields.Nested(time_series_model, required=True, description='Time series data'),
    'eta_c': fields.Float(required=False, default=0.95, description='Charging efficiency'),
    'eta_d': fields.Float(required=False, default=0.95, description='Discharging efficiency'),
})

# Output models
battery_result_model = api.model('BatteryResult', {
    'charging_power': fields.List(fields.Float, description='Optimal charging energy at each time step (Wh)'),
    'discharging_power': fields.List(fields.Float, description='Optimal discharging energy at each time step (Wh)'),
    'state_of_charge': fields.List(fields.Float, description='State of charge at each time step (Wh)')
})

limit_violation_result_model = api.model('LimitViolationResult', {
    'grid_import_limit_exceeded': fields.Boolean(description='The energy demand could only be satisfied by violating the grid import limit.'),
    'grid_export_limit_hit': fields.Boolean(description='The solar yield was reduced due to the limitation of grid export power.')
})

optimization_result_model = api.model('OptimizationResult', {
    'status': fields.String(description='Optimization status'),
    'objective_value': fields.Float(description='Optimal objective function value'),
    'limit_violations': fields.Nested(limit_violation_result_model, description='Collection of flags signalling the violation of defined limits'),
    'batteries': fields.List(fields.Nested(battery_result_model), description='Battery optimization results'),
    'grid_import': fields.List(fields.Float, description='Energy imported from grid at each time step (Wh)'),
    'grid_export': fields.List(fields.Float, description='Energy exported to grid at each time step (Wh)'),
    'flow_direction': fields.List(fields.Integer, description='Binary flow direction (1=export, 0=import)'),
    'grid_import_overshoot': fields.List(fields.Float, description='Energy above the power limit imported from grid at each time step (Wh)'),
    'grid_export_overshoot': fields.List(fields.Float, description='Energy not exported due to hitting the grid export power limit at each time step (Wh)')
})


@ns.route('/charge-schedule')
class OptimizeCharging(Resource):
    @api.expect(optimization_input_model, validate=True)
    @api.marshal_with(optimization_result_model)
    def post(self):
        """
        Optimize EV charging schedule using MILP

        This endpoint solves a Mixed Integer Linear Programming problem to optimize
        EV charging schedules considering battery constraints, grid prices, and energy demands.
        """
        try:
            data = api.payload

            # Parse strategy items with default values
            strat_data = data.get('strategy', {})
            strategy = OptimizationStrategy(
                charging_strategy=strat_data.get('charging_strategy', 'none'),
                discharging_strategy=strat_data.get('discharging_strategy', 'none')
            )

            # parse grid configuration
            grid_data = data.get('grid', {})
            grid = GridConfig(
                p_max_imp=grid_data.get('p_max_imp', None),
                p_max_exp=grid_data.get('p_max_exp', None),
                prc_p_exc_imp=grid_data.get('prc_p_exc_imp', None)
            )

            # Parse battery configurations
            batteries = []
            for bat_data in data['batteries']:
                batteries.append(BatteryConfig(
                    charge_from_grid=bat_data.get('charge_from_grid', False),
                    discharge_to_grid=bat_data.get('discharge_to_grid', False),
                    s_capacity=bat_data.get('s_capacity', bat_data['s_max']),
                    s_min=bat_data['s_min'],
                    s_max=bat_data['s_max'],
                    s_initial=bat_data['s_initial'],
                    p_demand=bat_data.get('p_demand'),
                    s_goal=bat_data.get('s_goal'),
                    c_min=bat_data['c_min'],
                    c_max=bat_data['c_max'],
                    d_max=bat_data['d_max'],
                    p_a=bat_data['p_a'],
                    c_priority=bat_data.get('c_priority', 0),
                ))

            # Parse time series data
            time_series = TimeSeriesData(
                dt=data['time_series']['dt'],
                gt=data['time_series']['gt'],
                ft=data['time_series']['ft'],
                p_N=data['time_series']['p_N'],
                p_E=data['time_series']['p_E'],
            )

            # Validate time series lengths
            lengths = [len(time_series.gt), len(time_series.ft),
                       len(time_series.p_N), len(time_series.p_E)]

            # Validate p_demand if provided
            for bat in batteries:
                if bat.p_demand is not None:
                    lengths.append(len(bat.p_demand))

            # Validate s_goal if provided
            for bat in batteries:
                if bat.s_goal is not None:
                    lengths.append(len(bat.s_goal))

            if len(set(lengths)) > 1:
                api.abort(400, "All time series must have the same length")

        except BadRequest:
            raise
        except Exception as e:
            api.abort(400, f"Invalid data format: {str(e)}")

        try:
            # Create and solve optimizer
            optimizer = Optimizer(
                strategy=strategy,
                grid=grid,
                batteries=batteries,
                time_series=time_series,
                eta_c=data.get('eta_c', 0.95),
                eta_d=data.get('eta_d', 0.95),
                M=1e6
            )

            started = time.perf_counter()
            result = optimizer.solve()
            elapsed = time.perf_counter() - started

            # one JSON line per request, so Log Analytics can attribute the response time to the
            # solve stages. The access log only carries the total.
            print(json.dumps({"solve": {
                "elapsed": round(elapsed, 3),
                "stages": optimizer.stage_seconds,
                "path": optimizer.solve_path,
                "preferences": optimizer.preference_stage,
                "status": result.get('status'),
                "steps": optimizer.T,
            }}), flush=True)

            dump_slow_request(data, elapsed)
            return result

        except Exception as e:
            api.abort(500, f"Optimization failed: {str(e)}")


@ns.route('/health')
class Health(Resource):
    def get(self):
        """Health check endpoint"""
        return {'status': 'healthy', 'message': 'EV Charging MILP API is running'}


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7050)
