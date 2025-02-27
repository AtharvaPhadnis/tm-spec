import argparse
import subprocess
import psutil
import time
import os

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

def monitor_memory_usage(pid, interval=1):
    max_rss = 0
    while psutil.pid_exists(pid):
        if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
            break
        current_rss = get_total_rss(pid)
        max_rss = max(max_rss, current_rss)
        # print(f"Current total RSS: {current_rss / (1024 * 1024):.2f} MB")
        # print(f"Max total RSS: {max_rss / (1024 * 1024):.2f} MB")
        time.sleep(interval)
    return max_rss

def launch_benchmark():
    new_env = os.environ.copy()
    new_env.update({"LD_PRELOAD": "/usr/local/lib/libjemalloc.so"})
    new_env.update({"MALLOC_CONF": "confirm_conf:true,stats_print:true"})
    cmd = f"runcpu --config=ap-new.cfg --threads={args.threads} {args.benchmark}"
    print(f'Running command: {cmd}')
    try:
        with open('./dump.log', 'w') as dump_file:
            result = subprocess.Popen(
                    ['runcpu', '--config=ap-new.cfg', f'--threads={args.threads}', f'{args.benchmark}'],
                    # stdout=subprocess.DEVNULL,
                    # stderr=subprocess.DEVNULL,
                    stdout=dump_file,
                    stderr=subprocess.DEVNULL,
                    env=new_env,
                    text=True,
                )
    except subprocess.CalledProcessError as e:
        print(f"Error while launching SPEC benchmark: {e.stderr}")
        raise
    
    return result

def run_benchmark(threads, benchmark):
    process = launch_benchmark()
    print(f'Launched SPEC benchmark : {args.benchmark} with PID: {process.pid}')
    max_memory = monitor_memory_usage(process.pid)
    return max_memory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SPEC CPU benchmark and monitor memory usage")
    parser.add_argument("--threads", type=int, required=True, help="Number of threads to use")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark name")
    global args
    args = parser.parse_args()

    max_memory = run_benchmark(args.threads, args.benchmark)
    print(f"Maximum RSS memory usage: {max_memory  / (1024 * 1024):.2f} MB")
