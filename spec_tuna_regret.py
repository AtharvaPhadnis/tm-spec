import argparse
import subprocess
import psutil
import time
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ConfigSpace as CS
import mlos_core.optimizers
import subprocess
import pandas as pd
from io import StringIO
from scipy import integrate

ITER_PER_RUN = 1
SPEC_CSV = ''
TS_CSV = ''
N_ITERATIONS = 50

def define_config_space_jemalloc():
    """
    Defines which parameters we would like to tune and adds them to the config space for jemalloc
    
    Returns:
        The configured input space that will be passed to the optimizer
    """

    # input_space = CS.ConfigurationSpace(seed=1234)
    input_space = CS.ConfigurationSpace()
    
    # TODO: Remove hardcoded parameters and configure them dynamically, perhaps from constants.py
    input_space.add(CS.CategoricalHyperparameter(name='je_tcache', choices=['true', 'false']))
    input_space.add(CS.CategoricalHyperparameter(name='je_background_thread', choices=['true', 'false']))
    input_space.add(CS.CategoricalHyperparameter(name='je_percpu_arena', choices=['disabled', 'percpu', 'phycpu']))
    input_space.add(CS.CategoricalHyperparameter(name='je_metadata_thp', choices=['disabled', 'always', 'auto']))
    input_space.add(CS.CategoricalHyperparameter(name='je_thp', choices=['default', 'always', 'never']))
    input_space.add(CS.CategoricalHyperparameter(name='je_retain', choices=['true', 'false']))
    input_space.add(CS.CategoricalHyperparameter(name='je_cache_oblivious', choices=['true', 'false']))
    input_space.add(CS.CategoricalHyperparameter(name='je_trust_madvise', choices=['true', 'false']))
    input_space.add(CS.CategoricalHyperparameter(name='je_dss', choices=['disabled', 'primary', 'secondary']))
    input_space.add(CS.UniformIntegerHyperparameter(name='je_narenas', lower=1, upper=640)) 
    input_space.add(CS.UniformIntegerHyperparameter(name='je_muzzy_decay_ms', lower=-1, upper=50000)) 
    input_space.add(CS.UniformIntegerHyperparameter(name='je_dirty_decay_ms', lower=-1, upper=50000)) 
    input_space.add(CS.UniformIntegerHyperparameter(name='je_lg_tcache_max', lower=1, upper=30)) 
    input_space.add(CS.UniformIntegerHyperparameter(name='je_max_background_threads', lower=1, upper=80)) 
    input_space.add(CS.UniformIntegerHyperparameter(name='je_lg_extent_max_active_fit', lower=1, upper=24)) 
    input_space.add(CS.UniformIntegerHyperparameter(name='je_oversize_threshold', lower=0, upper=32*1024*1024)) # 32 MiB
    # input_space.add(CS.UniformIntegerHyperparameter(name='je_mutex_max_spin', lower=1, upper=2400)) 
    # input_space.add(CS.UniformIntegerHyperparameter(name='je_tcache_nslots_small_min', lower=1, upper=80)) 
    # input_space.add(CS.UniformIntegerHyperparameter(name='je_tcache_nslots_small_max', lower=100, upper=400)) 
    # input_space.add(CS.UniformIntegerHyperparameter(name='je_tcache_nslots_large', lower=1, upper=80))
    # input_space.add(CS.UniformIntegerHyperparameter(name='je_lg_tcache_nslots_mul', lower=1, upper=16)) 
    # input_space.add(CS.UniformIntegerHyperparameter(name='je_tcache_gc_incr_bytes', lower=16384, upper=262144))
    # input_space.add(CS.UniformIntegerHyperparameter(name='je_tcache_gc_delay_bytes', lower=0, upper=16384))
    # input_space.add(CS.UniformIntegerHyperparameter(name='je_lg_tcache_flush_small_div', lower=1, upper=16))
    # input_space.add(CS.UniformIntegerHyperparameter(name='je_lg_tcache_flush_large_div', lower=1, upper=16))
    # input_space.add(CS.UniformIntegerHyperparameter(name='je_remote_free_max', lower=1, upper=64))
    # input_space.add(CS.UniformIntegerHyperparameter(name='je_remote_free_max_batch', lower=1, upper=16))

    # input_space.add(CS.UniformIntegerHyperparameter(name='env_MALLOC_MMAP_THRESHOLD_', lower=32*1024, upper=512*1024))
    # input_space.add(CS.UniformIntegerHyperparameter(name='env_MALLOC_MMAP_MAX_', lower=0, upper=1024))

    return input_space

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
    unit = args.unit
    rss_values = []
    div = 1
    match unit.upper():
        case 'KB':
            div = 1024
        case 'MB':
            div = 1024*1024
        case 'GB':
            div = 1204*1024*1024
        case _:
            div = 1
    process = psutil.Process(pid)
    while process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
        current_rss = get_total_rss(pid)
        rss_values.append(current_rss / div)
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
    # je_malloc_conf_str += 'stats_print:true'
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

def calculate_rss_score(current_rss, default_rss, interval=0.1):
    # Ensure both lists have the same length
    min_length = min(len(current_rss), len(default_rss))
    current_rss = current_rss[:min_length]
    default_rss = default_rss[:min_length]
    
    # Calculate regret
    regret = np.array(current_rss) - np.array(default_rss)
    
    # Generate timestamps
    timestamps = np.arange(min_length) * interval

    df[f'run_{len(df.columns)-1}'] = pd.Series(current_rss)

    # print(f'Regret: {regret}')
    # print(f'Timestamp: {timestamps}')
    # Calculate area under the regret-time curve
    rss_score = integrate.trapezoid(regret, timestamps)

    print(f'RSS Score: {rss_score}')
    
    return rss_score


def run_benchmark(allocator_config):
    env_dict = allocator_config.iloc[0].to_dict()
    avg_rss_values = []
    rss_values = []
    for _ in range(ITER_PER_RUN):
        process = launch_benchmark(env_dict)
        print(f'Launched SPEC benchmark : {args.benchmark} with PID: {process.pid}')
        rss_values = monitor_memory_usage(process.pid, interval=interval, unit='MB')
        process.wait()
        if not avg_rss_values:
            avg_rss_values = rss_values
        else:
            avg_rss_values = [sum(x) for x in zip(avg_rss_values, rss_values)]

    avg_rss_values = [x / ITER_PER_RUN for x in avg_rss_values]
    # print(avg_rss_values)

    rss_score = calculate_rss_score(avg_rss_values, base_rss, interval=interval)
    optimizer_score = pd.DataFrame({'rss': [rss_score]})
    
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
    parser.add_argument("--unit", type=str, required=False, help="RSS unit (KB/MB/GB)")
    global args
    args = parser.parse_args()
    SPEC_CSV = f'{args.benchmark.replace(".", "_")}_regret_out.csv'
    TS_CSV = f'{args.benchmark.replace(".", "_")}_ts.csv'

    warm_up_rss = []
    rss_values = []
    global interval
    interval = 1

    for _ in range(ITER_PER_RUN):
        process = launch_benchmark({})
        print(f'Launched SPEC benchmark : {args.benchmark} with PID: {process.pid}')
        rss_values = monitor_memory_usage(process.pid, interval=interval, unit='MB')
        process.wait()
        if not warm_up_rss:
            warm_up_rss = rss_values
        else:
            warm_up_rss = [sum(x) for x in zip(warm_up_rss, rss_values)]
    
    warm_up_rss = [x / ITER_PER_RUN for x in warm_up_rss]
    
    global base_rss
    base_rss = warm_up_rss

    # print(f'Warm up RSS: {warm_up_rss} MB')
    timestamps = np.arange(len(rss_values)) * interval
    global df
    df = pd.DataFrame({'Timestamp': timestamps})
    
    df[f'run_warm_up'] = pd.Series(warm_up_rss)

    print('Starting optimizer')
    input_space = define_config_space_jemalloc()

    optimizer = mlos_core.optimizers.SmacOptimizer(parameter_space=input_space, optimization_targets=['rss'])

    with open(SPEC_CSV, 'w+') as file:
        file.truncate(0)

    for i in range(N_ITERATIONS):
        print(f'Run #{i+1} / {N_ITERATIONS}')
        run_optimization(optimizer)

    default_rss = []

    for _ in range(ITER_PER_RUN):
        process = launch_benchmark({})
        print(f'Launched SPEC benchmark: {args.benchmark} with PID: {process.pid}')
        default_rss_values = monitor_memory_usage(process.pid, interval=interval, unit='MB')
        process.wait()
        if not default_rss:
            default_rss = default_rss_values
        else:
            default_rss = [sum(x) for x in zip(default_rss_values, default_rss)]


    default_rss = [x / ITER_PER_RUN for x in default_rss]

    timestamps = np.arange(len(default_rss)) * interval
    
    df[f'run_default_jemalloc'] = pd.Series(default_rss)


    default_rss = []
    for _ in range(ITER_PER_RUN):
        process = launch_benchmark_tcmalloc()
        print(f'Launched SPEC benchmark: {args.benchmark} with PID: {process.pid}')
        default_rss_values = monitor_memory_usage(process.pid, interval=interval, unit='MB')
        process.wait()
        if not default_rss:
            default_rss = default_rss_values
        else:
            default_rss = [sum(x) for x in zip(default_rss_values, default_rss)]
    
    default_rss = [x / ITER_PER_RUN for x in default_rss]

    df[f'run_default_tcmalloc'] = pd.Series(default_rss)

    min_column_length = df.apply(lambda col: col.dropna().shape[0]).min()
    df = df.iloc[:min_column_length]
    df.to_csv(TS_CSV, index=False)



    # print(f'Default RSS for tcmalloc: {default_rss} MB')
    # max_memory = run_benchmark(args.threads, args.benchmark)
    # print(f"Maximum RSS memory usage: {max_memory  / (1024 * 1024):.2f} MB")
