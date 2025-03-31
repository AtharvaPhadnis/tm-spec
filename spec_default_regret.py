import argparse
import subprocess
import psutil
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ConfigSpace as CS
import mlos_core.optimizers
import subprocess
import pandas as pd
from io import StringIO
import csv

ITER_PER_RUN = 1

def get_total_rss(pid):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        total_rss = parent.memory_info().rss
        for child in children:
            total_rss += child.memory_info().rss
        return total_rss
    except psutil.NoSuchProcess:
        return 0

def monitor_memory_usage(pid, interval=1, unit='MB'):
    max_rss = 0
    rss_values = []
    process = psutil.Process(pid)
    while process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
        current_rss = get_total_rss(pid)
        rss_values.append(current_rss)
        time.sleep(interval)
        # print(f"Current total RSS: {current_rss / (1024 * 1024):.2f} MB")
        # print(f"Max total RSS: {max_rss / (1024 * 1024):.2f} MB")
        time.sleep(interval)
    process.wait()
    return rss_values

def launch_benchmark(env_dict):
    new_env = os.environ.copy()
    new_env.update({"LD_PRELOAD": "/usr/local/lib/libjemalloc.so"})

    stripped_dict = {key.removeprefix('env_'): str(value) for key, value in env_dict.items() if key.startswith('env_')}
    new_env.update(stripped_dict)

    je_malloc_conf_str = "" 
    for key, val in env_dict.items():
        if key.startswith('je_'):
            je_malloc_conf_str += f"{key.removeprefix('je_')}:{str(val)},"

    if len(je_malloc_conf_str) > 0:
        je_malloc_conf_str = je_malloc_conf_str[:-1]
    else:
        je_malloc_conf_str = "stats_print:true"
    new_env.update({"MALLOC_CONF": f"{je_malloc_conf_str}"})
    cmd = f"runcpu --config=ap-new.cfg --threads={args.threads} {args.benchmark}"
    print(f'Running command: {cmd}')
    try:
        with open('./dump.log', 'w') as dump_file:
            result = subprocess.Popen(
                    ['runcpu', '--config=ap-new.cfg', '--action', 'onlyrun', '--size=ref', '--tune=base', f'--threads={args.threads}', f'{args.benchmark}'],
                    # stdout=subprocess.DEVNULL,
                    # stderr=subprocess.DEVNULL,
                    stdout=dump_file,
                    stderr=dump_file,
                    env=new_env,
                    text=True,
                )
    except subprocess.CalledProcessError as e:
        print(f"Error while launching SPEC benchmark: {e.stderr}")
        raise
    
    return result

def launch_benchmark_tcmalloc():
    new_env = os.environ.copy()
    new_env.update({"LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"})
   
    cmd = f"runcpu --config=ap-new.cfg --threads={args.threads} {args.benchmark}"
    print(f'Running command: {cmd}')
    try:
        with open('./dump.log', 'w') as dump_file:
            result = subprocess.Popen(
                    ['runcpu', '--config=ap-new.cfg', '--action', 'onlyrun', '--size=ref', '--tune=base', f'--threads={args.threads}', f'{args.benchmark}'],
                    # stdout=subprocess.DEVNULL,
                    # stderr=subprocess.DEVNULL,
                    stdout=dump_file,
                    stderr=dump_file,
                    env=new_env,
                    text=True,
                )
    except subprocess.CalledProcessError as e:
        print(f"Error while launching SPEC benchmark: {e.stderr}")
        raise
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SPEC CPU benchmark and monitor memory usage")
    parser.add_argument("--threads", type=int, required=True, help="Number of threads to use")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark name")
    parser.add_argument("--outfile", type=str, required=False, default=None, help="Name of output file")
    global args
    args = parser.parse_args()

    if args.outfile == None:
        args.outfile = args.benchmark

    SPEC_CSV = f'{args.outfile.replace(".", "_")}_timeseries.csv'

    default_rss = 0
    interval = 1

    for _ in range(ITER_PER_RUN):
        process = launch_benchmark({})
        print(f'Launched SPEC benchmark : {args.benchmark} with PID: {process.pid}')
        rss_values = monitor_memory_usage(process.pid, interval=interval, unit='MB')
        process.wait()

    timestamps = np.arange(len(rss_values)) * interval

    with open(SPEC_CSV, mode='a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Timestamp', 'RSS'])  # Header row
        csv_writer.writerows(zip(timestamps, rss_values))  # Write each 
    

    # for _ in range(ITER_PER_RUN):
    #     process = launch_benchmark_tcmalloc()
    #     print(f'Launched SPEC benchmark : {args.benchmark} with PID: {process.pid}')
    #     curr_rss = monitor_memory_usage(process.pid, unit='MB')
    #     process.wait()
    #     default_rss += curr_rss / ITER_PER_RUN

    # print(f'Default RSS: {default_rss} MB')
    # with open(SPEC_CSV, mode='a', newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #     csv_writer.writerow(['tcmalloc', args.benchmark, default_rss])
    
    # max_memory = run_benchmark(args.threads, args.benchmark)
    # print(f"Maximum RSS memory usage: {max_memory  / (1024 * 1024):.2f} MB")
