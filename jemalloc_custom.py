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
from spec_tuna_regret_new import launch_benchmark_external, get_total_rss

def monitor_memory_usage_custom(pid, interval=1, unit='MB'):
    if args is not None:
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SPEC CPU benchmark and monitor memory usage")
    parser.add_argument("--threads", type=int, required=True, help="Number of threads to use")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark name")
    parser.add_argument("--unit", type=str, required=False, help="RSS unit (KB/MB/GB)")
    parser.add_argument("--size", type=str, required=False, help="Size: ref, test, train")
    args = parser.parse_args()
    SPEC_CSV = f'{args.benchmark.replace(".", "_")}_custom_regret_out.csv'
    TS_CSV = f'{args.benchmark.replace(".", "_")}_custom_ts.csv'

    with open(SPEC_CSV, 'w+') as file:
        file.truncate(0)
    
    with open(TS_CSV, 'w+') as file:
        file.truncate(0)

    interval = 1
    default_conf_dict = {'je_cache_oblivious': 'true', 'je_metadata_thp': 'disabled', 'je_trust_madvise': 'false', 'je_retain': 'true', 'je_dss': 'secondary', 'je_narenas': 160, 'je_oversize_threshold': 8388608, 'je_background_thread': 'false', 'je_max_background_threads': 40, 'je_dirty_decay_ms': 10000, 'je_muzzy_decay_ms': 10000, 'je_lg_extent_max_active_fit': 6, 'je_tcache': 'true', 'je_lg_tcache_max': 15, 'je_thp': 'default', 'je_percpu_arena': 'disabled', 'je_mutex_max_spin': '1600'}

    process = launch_benchmark_external(default_conf_dict, args)
    print(f'Launched SPEC benchmark : {args.benchmark} with PID: {process.pid}')
    rss_values = monitor_memory_usage_custom(process.pid, interval=interval, unit=f'{args.unit}')
    process.wait()

    conf_dict = {'je_cache_oblivious': 'true', 'je_metadata_thp': 'disabled', 'je_trust_madvise': 'false', 'je_retain': 'true', 'je_dss': 'secondary', 'je_narenas': 20, 'je_oversize_threshold': 8388608, 'je_background_thread': 'true', 'je_max_background_threads': 40, 'je_dirty_decay_ms': 5000, 'je_muzzy_decay_ms': 5000, 'je_lg_extent_max_active_fit': 6, 'je_tcache': 'true', 'je_lg_tcache_max': 12, 'je_thp': 'default', 'je_percpu_arena': 'disabled', 'je_mutex_max_spin': '1600'}
    process = launch_benchmark_external(conf_dict, args)
    print(f'Launched SPEC benchmark : {args.benchmark} with PID: {process.pid}')
    rss_values = monitor_memory_usage_custom(process.pid, interval=interval, unit=f'{args.unit}')
    process.wait()

    timestamps = np.arange(len(rss_values)) * interval
    df = pd.DataFrame({'Timestamp': timestamps})
    df[f'run_custom_jemalloc'] = pd.Series(rss_values)

    df.fillna(0).to_csv(TS_CSV, index=False)







