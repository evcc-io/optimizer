import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

from optimizer.app import app

# help text
action_help = "create: create a new test case from a json request. \n \
                update: update an existing test case with the result of the current optimizer.\n \
                run: run to current test case, compare to the expected results and show the data."
file_help = "json file to process. For action 'create', a json request to the optimizer is expected.\
            for 'run' and 'update' a test case file. "

# disable warnings on div 0 during deviation calculation
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# parse the args
parser = argparse.ArgumentParser(prog="testbench")
parser.add_argument("action", choices=["create", "update", "run"], help=action_help)
parser.add_argument("file", type=str, default="", help=file_help)
parser.add_argument("-o", "--outfile", type=str, default="test_case.json", help="with action create: path to the test case file to write.")
parser.add_argument("-s", "--stacked", action="store_true", help="draw stacked chart where meaningful.")
args = parser.parse_args()
action = args.action
file_in = Path(args.file)
file_out = Path(args.outfile)
use_stacked = args.stacked

# checks
if not file_in.is_file():
    print(f"File not found: {file_in.name}")
    sys.exit(1)

if not file_out.parents[0].is_dir():
    print(f"Directory does not exist: {file_out.parents[0].name}")
    sys.exit(1)

client = app.test_client()

# create a new test case from a json request
if action == "create":
    request = json.loads(file_in.read_text())
    if "batteries" not in request:
        print(f"unexpected format in {file_in.name}")
        sys.exit(1)
    response = client.post("/optimize/charge-schedule", json=request)
    if response.status_code != 200:
        print(f"Request to optimizer returned with status {response.status_code}")
        sys.exit(1)
    test_case = {}
    test_case['request'] = request
    test_case['expected_response'] = response.get_json()
    json.dump(test_case, indent=4, fp=open(file_out, "w"))
    print(f"test case written to file {file_out}.")

# update an existing test case with the result of the current optimizer
if action == "update":
    test_case = json.loads(file_in.read_text())
    if "request" not in test_case or "expected_response" not in test_case:
        print(f"unexpected format in {file_in.name}")
        sys.exit(1)
    request = test_case["request"]
    response = client.post("/optimize/charge-schedule", json=request)
    if response.status_code != 200:
        print(f"Request to optimizer returned with status {response.status_code}")
        sys.exit(1)
    strict = test_case.get('strict', False)
    test_case = {}
    if strict:
        test_case['strict'] = strict
    test_case['request'] = request
    test_case['expected_response'] = response.get_json()
    json.dump(test_case, indent=4, fp=open(file_in, "w"))
    print(f"test case {file_in} updated.")

# run to current test case, compare to the expected results and show the data
if action == "run":
    test_case = json.loads(file_in.read_text())
    if "request" not in test_case or "expected_response" not in test_case:
        print(f"unexpected format in {file_in.name}")
        sys.exit(1)
    request = test_case["request"]
    expected_response = test_case["expected_response"]
    client = app.test_client()
    response = client.post("/optimize/charge-schedule", json=request)
    if response.status_code != 200:
        print(f"Request to optimizer returned with status {response.status_code}")
        sys.exit(1)
    else:
        # compare optimizer status
        df_status = pd.DataFrame({
            "current run": [response.json["status"]],
            "expected": [expected_response["status"]]
        })

        print("Otimizer Status: ")
        print(tabulate(df_status, headers='keys', tablefmt='psql'))

        if response.json["status"] != "Optimal" or expected_response["status"] != "Optimal":
            print("Non optimal optimizer status, stopping.")
            sys.exit(0)

        # compare objective values
        calc_obj_value = response.json["objective_value"]
        exp_obj_value = expected_response["objective_value"]
        obj_deviation = (calc_obj_value - exp_obj_value) / exp_obj_value

        df_objective = pd.DataFrame({
            "current run": [calc_obj_value],
            "expected": [exp_obj_value],
            "deviation": [obj_deviation]
        })
        print("Objective Values: ")
        print(tabulate(df_objective, headers='keys', tablefmt='psql',  floatfmt=".4f"))

        # gather data for table and diagram
        ts_input = request["time_series"]
        dt = ts_input["dt"]
        dt0 = dt.copy()
        dt0.insert(0, 0.)
        ts_time_ex = np.cumsum(dt0)
        dt0 = dt0[:-1]
        ts_time = np.cumsum(dt0)
        ts_period = np.array(dt)
        ts_prc_import = np.array(ts_input["p_N"])*1000
        ts_prc_export = np.array(ts_input["p_E"])*1000
        ts_solar_raw = ts_input["ft"]
        ts_demand_raw = ts_input["gt"]
        ts_grid_import_raw = response.json["grid_import"]
        ts_grid_export_raw = response.json["grid_export"]
        ts_solar = np.divide(ts_input["ft"], ts_input["dt"])
        ts_demand = np.negative(np.divide(ts_input["gt"], ts_input["dt"]))
        ts_grid = np.divide(np.subtract(response.json["grid_import"], response.json["grid_export"]), ts_input["dt"])
        ts_grid_exp = np.divide(np.subtract(expected_response["grid_import"], expected_response["grid_export"]), ts_input["dt"])
        ts_grid_dev = np.divide(np.subtract(ts_grid, ts_grid_exp), ts_grid_exp)

        # create the table dataframe
        df_table = pd.DataFrame({
            "period": ts_period,
            "prc_import": ts_prc_import,
            "prc_export": ts_prc_export,
            "E_solar": ts_solar_raw,
            "E_demand": ts_demand_raw
        })

        # add battery data for the table
        for i, bat in enumerate(response.json["batteries"]):
            df_table[f"E_bat{i}_c"] = bat["charging_power"]
            df_table[f"E_bat{i}_c_exp"] = expected_response["batteries"][i]["charging_power"]
            df_table[f"E_bat{i}_d"] = bat["discharging_power"]
            df_table[f"E_bat{i}_d_exp"] = expected_response["batteries"][i]["discharging_power"]
            if "s_goal" in request["batteries"][i]:
                df_table[f"SOC_bat{i}_goal"] = request["batteries"][i]["s_goal"]
            if "p_demand" in request["batteries"][i]:
                df_table[f"E_bat{i}_demand"] = request["batteries"][i]["p_demand"]
            df_table[f"SOC_bat{i}"] = bat["state_of_charge"]
            df_table[f"SOC_bat{i}_exp"] = expected_response["batteries"][i]["state_of_charge"]

        # print the table
        print(tabulate(df_table, headers='keys', tablefmt='psql',  floatfmt=".2f"))

        # Create diagram dataframe
        df_diagram = pd.DataFrame({
            "time": ts_time,
            "prc_import": ts_prc_import,
            "prc_export": ts_prc_export,
            "P_solar": ts_solar,
            "P_demand": ts_demand,
            "P_grid": ts_grid,
            "P_grid_exp": ts_grid_exp,
            "P_grid_dev": ts_grid_dev
        })

        # add battery data for the diagram
        for i, bat in enumerate(response.json["batteries"]):
            # figure out what value to take for 100% SOC
            bat_capacity = request["batteries"][i]["s_max"]
            if "s_capacity" in request["batteries"][i]:
                bat_capacity = request["batteries"][i]["s_capacity"]
            df_diagram[f"P_bat{i}"] = np.divide(np.subtract(bat["discharging_power"], bat["charging_power"]), ts_input["dt"])
            df_diagram[f"SOC_bat{i}"] = np.divide(bat["state_of_charge"], bat_capacity)*100
            df_diagram[f"P_bat{i}_exp"] = np.divide(np.subtract(expected_response["batteries"][i]["discharging_power"],
                                                                expected_response["batteries"][i]["charging_power"]),
                                                    ts_input["dt"])
            df_diagram[f"P_bat{i}_dev"] = np.divide(np.subtract(df_diagram[f"P_bat{i}"], df_diagram[f"P_bat{i}_exp"]), df_diagram[f"P_bat{i}_exp"])

        df_diagram['time'] = pd.to_datetime(df_diagram['time'], unit='s')
        ts_time_ex = pd.to_datetime(ts_time_ex, unit="s")

        fig, axs = plt.subplots(3, figsize=(16, 12), height_ratios=[1, 2, 1])

        axs[0].set_title("Battery SOCs and Prices")
        axs[0].set_ylabel("SOC [%]")
        axs[0].set
        for i, bat in enumerate(response.json["batteries"]):
            axs[0].stairs(df_diagram[f"SOC_bat{i}"], ts_time_ex, label=f"SOC_bat{i} [%]", linewidth=1.5)
        axs0_r = axs[0].twinx()
        axs0_r.set_ylabel("Price [€/kWh]")
        axs0_r.plot(df_diagram["time"], df_diagram["prc_import"], 'r+', label="prc_import [€/kWh]")
        axs0_r.plot(df_diagram["time"], df_diagram["prc_export"], 'g+', label="prc_export [€/kWh]")
        axs[0].xaxis.set_minor_locator(mdates.HourLocator())
        axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        axs[0].grid()
        axs[0].legend()

        axs[1].set_title("Power Balance")
        axs[1].set_ylabel("Power [kW]")
        if use_stacked:
            for i, bat in enumerate(response.json["batteries"]):
                axs[1].stackplot(df_diagram["time"], df_diagram[f"P_bat{i}"], labels={f"P_bat{i} [kW]"})
            axs[1].stackplot(df_diagram["time"], df_diagram["P_grid"], labels={"P_grid [kW]"})
            axs[1].stackplot(df_diagram["time"], df_diagram["P_solar"], labels={"P_solar [kW]"})
            axs[1].stackplot(df_diagram["time"], df_diagram["P_demand"], labels={"P_demand [kW]"})
        else:
            for i, bat in enumerate(response.json["batteries"]):
                axs[1].stairs(df_diagram[f"P_bat{i}"], ts_time_ex, label=f"P_bat{i} [kW]", linewidth=1.5)
            axs[1].stairs(df_diagram["P_grid"], ts_time_ex, label="P_grid [kW]", linewidth=1.5)
            axs[1].stairs(df_diagram["P_solar"], ts_time_ex, label="P_solar [kW]", linewidth=1.5)
            axs[1].stairs(df_diagram["P_demand"], ts_time_ex, label="P_demand [kW]", linewidth=1.5)
        axs[1].xaxis.set_minor_locator(mdates.HourLocator())
        axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        axs[1].grid()
        axs[1].legend()

        axs[2].set_title("Deviation from Expected Results")
        axs[2].set_ylabel("Deviation to Expected [1]")
        for i, bat in enumerate(response.json["batteries"]):
            axs[2].stairs(df_diagram[f"P_bat{i}_dev"], ts_time_ex, label=f"P_bat{i}_dev [1]", linewidth=1.5)
        axs[2].stairs(df_diagram["P_grid_dev"], ts_time_ex, label="P_grid_dev [1]", linewidth=1.5)
        axs[2].xaxis.set_minor_locator(mdates.HourLocator())
        axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        axs[2].grid()
        axs[2].legend()

        plt.show()
