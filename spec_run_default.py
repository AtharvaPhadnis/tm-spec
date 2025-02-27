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
    process = psutil.Process(pid)
    while process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
        current_rss = get_total_rss(pid)
        max_rss = max(max_rss, current_rss)
        # print(f"Current total RSS: {current_rss / (1024 * 1024):.2f} MB")
        # print(f"Max total RSS: {max_rss / (1024 * 1024):.2f} MB")
        time.sleep(interval)
    process.wait()
    match unit.upper():
        case 'KB':
            return round(max_rss / 1024, 5) 
        case 'MB':
            return round(max_rss / (1024 ** 2), 5) 
        case 'GB':
            return round(max_rss / (1024 ** 3), 5)
        case _:  # Default to bytes
            return max_rss

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

def run_benchmark(allocator_config):
    env_dict = allocator_config.iloc[0].to_dict()
    avg_rss = 0
    for _ in range(ITER_PER_RUN):
        process = launch_benchmark(env_dict)
        print(f'Launched SPEC benchmark : {args.benchmark} with PID: {process.pid}')
        curr_rss = monitor_memory_usage(process.pid, unit='MB')
        process.wait()
        print(f'Recorded max RSS: {curr_rss} MB for current run')
        avg_rss += curr_rss / ITER_PER_RUN

    optimizer_score = pd.DataFrame({'rss': [avg_rss]})
    
    combined_result = pd.concat([allocator_config, optimizer_score], axis=1)

    file_empty = False
    try:
        with open(SPEC_CSV, 'x') as f:  # Try creating the file
            combined_result.to_csv(f, index=False, header=True)
    except FileExistsError:
        if(os.stat(SPEC_CSV).st_size == 0):
            file_empty = True
        # If the file already exists, append without writing the header
        combined_result.to_csv(SPEC_CSV, mode='a', index=False, header=file_empty)

    return optimizer_score # This must be a dataframe with the score of the current trial

def run_optimization(optimizer: mlos_core.optimizers.BaseOptimizer):
    allocator_tunables = optimizer.suggest()
    allocator_config = pd.Series(allocator_tunables.config)
    allocator_config = pd.DataFrame([allocator_config])
    
    scores = run_benchmark(allocator_config).squeeze('rows')

    optimizer.register(allocator_tunables.complete(scores))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SPEC CPU benchmark and monitor memory usage")
    parser.add_argument("--threads", type=int, required=True, help="Number of threads to use")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark name")
    global args
    args = parser.parse_args()
    SPEC_CSV = 'default_values.csv'

    default_rss = 0

    for _ in range(ITER_PER_RUN):
        process = launch_benchmark({})
        print(f'Launched SPEC benchmark : {args.benchmark} with PID: {process.pid}')
        curr_rss = monitor_memory_usage(process.pid, unit='MB')
        process.wait()
        default_rss += curr_rss / ITER_PER_RUN

    print(f'Default RSS: {default_rss} MB')
    with open(SPEC_CSV, mode='a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['jemalloc', args.benchmark, default_rss])
    

    for _ in range(ITER_PER_RUN):
        process = launch_benchmark_tcmalloc()
        print(f'Launched SPEC benchmark : {args.benchmark} with PID: {process.pid}')
        curr_rss = monitor_memory_usage(process.pid, unit='MB')
        process.wait()
        default_rss += curr_rss / ITER_PER_RUN

    print(f'Default RSS: {default_rss} MB')
    with open(SPEC_CSV, mode='a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['tcmalloc', args.benchmark, default_rss])
    
    # max_memory = run_benchmark(args.threads, args.benchmark)
    # print(f"Maximum RSS memory usage: {max_memory  / (1024 * 1024):.2f} MB")
