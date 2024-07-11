#!/usr/bin/env python3
#
# Script to run HeCBench benchmarks and gather results

import re, time, datetime, sys, subprocess, multiprocessing, os
import argparse
import json
import logging

def await_input(prompt: str, is_valid_input) -> str:
    """ Wait the user for input until it is valid. """
    r = input(prompt)
    while not is_valid_input(r):
        r = input(prompt)
    return r

class Benchmark:
    def __init__(self, args, name, res_regex, run_args = [], binary = "main", invert = False):
        if name.endswith('sycl'):
            self.MAKE_ARGS = ['GCC_TOOLCHAIN="{}"'.format(args.gcc_toolchain)]
            if args.sycl_type == 'cuda':
                self.MAKE_ARGS.append('CUDA=yes')
                self.MAKE_ARGS.append('CUDA_ARCH=sm_{}'.format(args.nvidia_sm))
            elif args.sycl_type == 'hip':
                self.MAKE_ARGS.append('HIP=yes')
                self.MAKE_ARGS.append('HIP_ARCH={}'.format(args.amd_arch))
            elif args.sycl_type == 'opencl':
                self.MAKE_ARGS.append('CUDA=no')
                self.MAKE_ARGS.append('HIP=no')
            elif args.sycl_type == 'cpu':
                self.MAKE_ARGS.append('CUDA=no')
                self.MAKE_ARGS.append('HIP=no')
                self.MAKE_ARGS.append('GPU=no')
        elif name.endswith('cuda'):
            self.MAKE_ARGS = ['ARCH=sm_{}'.format(args.nvidia_sm)]
        else:
            self.MAKE_ARGS = []

        if args.compiler_name:
            self.MAKE_ARGS.append('CC={}'.format(args.compiler_name))

        if args.extra_compile_flags:
            flags = args.extra_compile_flags.replace(',',' ')
            self.MAKE_ARGS.append('EXTRA_CFLAGS={}'.format(flags))

        if args.bench_dir:
            self.path = os.path.realpath(os.path.join(args.bench_dir, name))
        else:
            self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', name)

        self.name = name
        self.binary = binary
        self.res_regex = res_regex
        self.args = run_args
        self.invert = invert
        self.clean = args.clean
        self.verbose = args.verbose

    def compile(self, shared_data):
        if self.clean:
            subprocess.run(["make", "clean"], cwd=self.path).check_returncode()
            time.sleep(1) # required to make sure clean is done before building, despite run waiting on the invoked executable

        out = subprocess.DEVNULL
        if self.verbose:
            out = subprocess.PIPE

        proc = subprocess.run(["make"] + self.MAKE_ARGS, cwd=self.path,
                              stdout=out, stderr=subprocess.STDOUT, encoding="utf-8")

        try:
            proc.check_returncode()
            shared_data[self.name] = "success"
        except subprocess.CalledProcessError as e:
            print(f'Failed compilation in {self.path}.\n{e}')
            if e.stderr:
                print(e.stderr, file=sys.stderr)

            print("*****************************************************************************************")
            print("Description of the compilation error:")
            cause = subprocess.run(["make"] + self.MAKE_ARGS, cwd=self.path,
                                   check=False, capture_output=True, encoding="utf-8")
            print(cause.stdout)
            print(cause.stderr)
            print("*****************************************************************************************")
            shared_data[self.name] = "failed"
            #raise(e)

        if self.verbose:
            print(proc.stdout)

    def run(self):
        cmd = ["./" + self.binary] + self.args
        proc = subprocess.run(cmd, cwd=self.path, timeout=600,
                              stdout=subprocess.PIPE, encoding="utf-8")
        out = proc.stdout
        if self.verbose:
            print(" ".join(cmd))
            print(out)
        try:
             res = re.findall(self.res_regex, out)
        except re.error as e:
             print("Regular expression error occurred:", e.msg)
             print("Pattern:", e.pattern)
             print("Position:", e.pos)
        logging.debug(f'Results of re.findall:\n {res}')
        if not res:
            raise Exception(self.path + ":\nno regex match for " + self.res_regex + " in\n" + out)
        res = sum([float(i) for i in res]) #in case of multiple outputs sum them (e.g. total time)
        if self.invert:
            res = 1/res
        return res


def comp(b, d):
    b.compile(d)


def main():
    parser = argparse.ArgumentParser(description='HeCBench runner')
    parser.add_argument("--log", choices=["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"],
                        default="INFO", type=str.upper, help="logging level")
    parser.add_argument('--output', '-o',
                        help='Output file for csv results')
    parser.add_argument('--summary', '-s',
                        help='Output file for a summary of the execution of selected benchmarks')
    parser.add_argument("--yes-prompt", action="store_true", help="If provided, automatically answer yes to the prompt.")
    parser.add_argument('--repeat', '-r', type=int, default=1,
                        help='Repeat benchmark run')
    parser.add_argument('--warmup', '-w', type=bool, default=True,
                        help='Run a warmup iteration')
    parser.add_argument('--sycl-type', '-t', choices=['cuda', 'hip', 'opencl', 'cpu'], default='cuda',
                        help='Type of SYCL device to use (default is cuda)')
    parser.add_argument('--nvidia-sm', type=int, default=60,
                        help='NVIDIA SM version (default is 60)')
    parser.add_argument('--amd-arch', default='gfx908',
                        help='AMD Architecture (default is gfx908)')
    parser.add_argument('--compiler-name', default='',
                        help='If a compiler is specified, use the specified one; otherwise, the default compiler \
                              in Makefiles will be used')
    parser.add_argument('--gcc-toolchain', default='',
                        help='GCC toolchain location (e.g. /path/to/gcc/x86_64/gcc-9.1.0)')
    parser.add_argument('--extra-compile-flags', '-e', default='',
                        help='Additional compilation flags (inserted before the predefined CFLAGS)')
    parser.add_argument("--overwrite", action="store_true",
                        help="If benchmark results are already available in the output file , \
                              then overwrite them. Default behavior is to skip existing results.")
    parser.add_argument('--clean', '-c', action='store_true',
                        help='Clean the builds')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose outputs from the builds')
    parser.add_argument('--bench-dir', '-b',
                        help='Benchmark directory')
    parser.add_argument('--bench-data', '-d',
                        help='Benchmark data')
    parser.add_argument('--bench-fails', '-f',
                        help='List of failing benchmarks to ignore')
    parser.add_argument('bench', nargs='+',
                        help='Either specific benchmark name or sycl, cuda, or hip')

    args = parser.parse_args()

    # setup logging
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: {}".format(args.log))
    logging.basicConfig(format="%(asctime)s [%(levelname)s] -- %(message)s", level=numeric_level)

    # warn user before continuing
    logging.warning("This script will compile and run selected benchmarks in HeCBench and gather results. " +
        "It is recommended that before you run this script, the dataset are available for certain benchmarks " +
        "and the compilers are in the PATH environment.") 

    if not args.yes_prompt:
        response = await_input("Continue? [y/n] ", lambda r: r.lower() in ["y", "n", "yes", "no"])
        if response.lower() in ["n", "no"]:
            logging.info("Exiting.")
            return

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load benchmark data
    if args.bench_data:
        bench_data = args.bench_data
    else:
        bench_data = os.path.join(script_dir, 'benchmarks', 'subset.json') 

    with open(bench_data) as f:
        benchmarks = json.load(f)

    logging.info(f"Loaded benchmark names and command-line arguments from {bench_data}.")

    # Load fail file
    if args.bench_fails:
        bench_fails = os.path.abspath(args.bench_fails)
    else:
        bench_fails = os.path.join(script_dir, 'benchmarks', 'subset-fails.txt')

    with open(bench_fails) as f:
        fails = f.read().splitlines()

    logging.info(f"Loaded benchmark skip file from {bench_fails}.")

    # Build benchmark list from the benchmark data file (e.g. subset.json)
    benches = []
    for b in args.bench:
        if b in ['sycl', 'cuda', 'hip']:
            benches.extend([Benchmark(args, k+'-'+b, *v)
                            for k, v in benchmarks.items()
                            if k+'-'+b not in fails])  # e.g. nbody-sycl
            continue
        # b is a specific benchmark instead
        ch_index = b.find('-') # find specific character '-'
        b_sub = b[:ch_index]
        benches.append(Benchmark(args, b, *benchmarks[b_sub]))

    summary = {}
    t0 = time.time()
    try:
        #with multiprocessing.Pool() as p:
        #    p.map(comp, benches)
        procs = []
        with multiprocessing.Manager() as m:
           d = m.dict() # a shared dictionary (nested dictionary not supported)
           for b in benches: 
             p = multiprocessing.Process(target = comp, args = (b, d)) 
             procs.append(p)
             p.start()

           for p in procs:
               p.join()

           for k, v in d.items():
               summary[k] = {}
               summary[k]["compile"] = v;
           
    except Exception as e:
        print("Error compiling the benchmarks:")
        print(e)

    t_compiled = time.time()

    outfile = sys.stdout
    if args.overwrite:
        filtered_benches = benches
        if args.output:
            outfile = open(args.output, 'w+t')
        logging.info(f"Overwrite the output file {args.output}.")
    else:
        recorded_benchmarks = set()
        if args.output:
            if os.path.isfile(args.output):
                outfile = open(args.output, 'r+t')
                for line in outfile:
                    bench, *rest = line.split(',')
                    recorded_benchmarks.add(bench)
                    # record the status only when it is in the input benchmark list
                    ch_index = bench.find('-')
                    if bench[:ch_index] in benchmarks.keys():
                        summary[bench]["run"] = "skipped"
                outfile.seek(0, 2) # seek to end of the file.
            else:
                outfile = open(args.output, 'w+t')

        filtered_benches = [b for b in benches if b.name not in recorded_benchmarks]
        num_filtered_benches = len(benches) - len(filtered_benches)
        if num_filtered_benches:
            print(f"Filtered out {num_filtered_benches} benchmarks."
                  " Results already exists in the output file.", flush=True)

    for i, b in enumerate(filtered_benches, 1):
        if b.name not in summary.keys():
            summary[b.name] = {}
        try:
            print(f"running {i}/{len(filtered_benches)}: {b.name}", flush=True)

            if args.warmup:
                b.run()

            res = []
            for i in range(args.repeat):
                res.append(str(b.run()))

            print(b.name + "," + ", ".join(res), file=outfile)
            summary[b.name]["run"] = "success"
        except Exception as e:
            print("Error running: ", b.name)
            print(e)
            summary[b.name]["run"] = "failed"

    if args.output:
        outfile.close()

    t_done = time.time()

    print("*****************************************************************************************")
    print(datetime.datetime.now())
    print("Summary of the benchmark execution:\n")
    print("Compilation took {} s, running took {} s.".format(t_compiled-t0, t_done-t_compiled))
    if args.summary:
        with open(args.summary, "w") as f:
            json.dump(summary, f, indent=4, sort_keys=True)
        logging.info(f"Wrote the summary to {args.summary}.")
    else:
        print(json.dumps(summary, indent=4, sort_keys=True))
    res = sum(('compile' in x.keys() and x['compile'] == 'failed' or
               'run' in x.keys() and x['run'] == 'failed') for x in summary.values())
    print(f'Number of benchmark compile or run failures: {res}');
    print("*****************************************************************************************")


if __name__ == "__main__":
    main()

