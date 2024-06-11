#!/usr/bin/env python3
#
# Script to run HeCBench benchmarks and gather results

import re, time, sys, subprocess, multiprocessing, os
import argparse
import json
from multiprocessing import Manager


class Benchmark:
    def __init__(self, args, name, res_regex, run_args = [], binary = "main", invert = False):
        if name.endswith('sycl'):
            self.MAKE_ARGS = ['GCC_TOOLCHAIN="{}"'.format(args.gcc_toolchain)]
            self.MAKE_ARGS.append('CC={}'.format(args.compiler_name))
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
            self.MAKE_ARGS = ['CUDA_ARCH=sm_{}'.format(args.nvidia_sm)]
            self.MAKE_ARGS.append('CC="{}"'.format(args.compiler_name)) # jenny added for cucc compiler
        else:
            self.MAKE_ARGS = []

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
        self.timeout = args.timeout

    def __eq__(self, other):
        return self.name == other.name

    def compile(self, benches, outfile_name, lock):
        if self.clean:
            subprocess.run(["make", "clean"], cwd=self.path).check_returncode()
            time.sleep(1) # required to make sure clean is done before building, despite run waiting on the invoked executable

        out = subprocess.DEVNULL
        if self.verbose:
            out = subprocess.PIPE

        proc = subprocess.run(["make"] + self.MAKE_ARGS, cwd=self.path, stdout=out, stderr=subprocess.STDOUT, encoding="utf-8")
        try:
            proc.check_returncode()
        except subprocess.CalledProcessError as e:
            print(f'Failed compilation in {self.path}.\n{e}')
            if e.stderr:
                print(e.stderr, file=sys.stderr)

            # write to error file
            print("%%%%%%%% " + self.name)
            #with lock:
            #    with open(outfile_name, 'a') as f:
            #        f.write(self.name + ",FAILED,N/A," + str(e).replace('\n', ' ').replace(',', ' ') + "\n")

            print("***write to file - failed compilation***")
            print("-----------------------------------------------------------------------------------------")
            print("Description of the compilation error:")
            cause = subprocess.run(["make"] + self.MAKE_ARGS, cwd=self.path,
                                   check=False, capture_output=True, encoding="utf-8")
            with lock:
                with open(outfile_name, 'a') as f:
                    f.write(self.name + ",FAILED,N/A," + str(cause.stdout).replace('\n', ' ').replace(',', ' ') + str(cause.stderr).replace('\n', ' ').replace(',', ' ') + "\n")
            print(cause.stdout)
            print(cause.stderr)
            print("-----------------------------------------------------------------------------------------")
            with lock:
                benches.remove(self)
                print("***remove from benches***")

        if self.verbose:
            print(proc.stdout)

    def run(self):
        cmd = ["./" + self.binary] + self.args
        proc = subprocess.run(cmd, cwd=self.path, stdout=subprocess.PIPE, encoding="utf-8", timeout=self.timeout)
        out = proc.stdout
        if self.verbose:
            print(" ".join(cmd))
            print(out)
        res = re.findall(self.res_regex, out)
        # Find in output for verification keywords
        verify_res = re.findall('not passed|pass|fail|succe|correct', out.lower())
        print("[VALIDATION]")
        if not verify_res:
            print("N/A")
        else:
            for r in verify_res:
                print(r)
        # Result line to be added in the output file
        res_l = []
        res_l.append("COMPILED")
        if out.strip() != "": res_l.append("SUCCESS")
        else: res_l.append("NO OUTPUT")
        # If no matching regex
        if not res:
            out = out.replace('\n', ' ').replace(',', ' ')
            res_l.append("[NO MATCH] " + out)
            # res_l[1] = "NO MATCH"
        else:
            print("[REGEX MATCHING RESULT]")
            for i in res:
                print(i)
            res_l.append(sum([float(i) for i in res])) #in case of multiple outputs sum them
            if self.invert:
                res_l[2] = 1/res_l[2]
        if verify_res:
            passing = True
            for v in verify_res:
                if ("pass" not in v.lower() and "succe" not in v.lower() and "correct" not in v.lower()) or v.lower() == "not passed":
                    passing = False
                    res_l.append("FAIL")
                    break
            if passing: res_l.append("PASS")
        else: res_l.append("N/A")
        
        res_l.append(str(self.res_regex).replace(',', ' '))

        return res_l


def comp(b, benches, outfile_name, lock):
    print("compiling: {}".format(b.name))
    b.compile(benches, outfile_name, lock)

def main():
    parser = argparse.ArgumentParser(description='HeCBench runner')
    parser.add_argument('--output', '-o',
                        help='Output file for csv results', default="out.csv")
    parser.add_argument('--repeat', '-r', type=int, default=1,
                        help='Repeat benchmark run')
    parser.add_argument('--warmup', '-w', type=bool, default=True,
                        help='Run a warmup iteration')
    parser.add_argument('--sycl-type', '-s', choices=['cuda', 'hip', 'opencl', 'cpu'], default='cuda',
                        help='Type of SYCL device to use (default is cuda)')
    parser.add_argument('--nvidia-sm', type=int, default=60,
                        help='NVIDIA SM version (default is 60)')
    parser.add_argument('--amd-arch', default='gfx908',
                        help='AMD Architecture (default is gfx908)')
    parser.add_argument('--compiler-name', default='clang++',
                        help='Name of a SYCL compiler (default is clang++)')
    parser.add_argument('--gcc-toolchain', default='',
                        help='GCC toolchain location (e.g. /path/to/gcc/x86_64/gcc-9.1.0)')
    parser.add_argument('--extra-compile-flags', '-e', default='',
                        help='Additional compilation flags (inserted before the predefined CFLAGS)')
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
    parser.add_argument('--timeout', '-t', type=int,
                        help='Timeout limit in seconds', default=120)

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load benchmark data
    if args.bench_data:
        bench_data = args.bench_data
    else:
        bench_data = os.path.join(script_dir, 'benchmarks', 'subset.json') 

    with open(bench_data) as f:
        benchmarks = json.load(f)

    # Load fail file
    if args.bench_fails:
        bench_fails = os.path.abspath(args.bench_fails)
    else:
        bench_fails = os.path.join(script_dir, 'benchmarks', 'subset-fails.txt')

    with open(bench_fails) as f:
        fails = f.read().splitlines()

    # Build benchmark list
    manager = Manager()
    benches = manager.list()
    for b in args.bench:
        if b in ['sycl', 'cuda', 'hip']:
            benches.extend([Benchmark(args, k, *v)
                            for k, v in benchmarks.items()
                            if k.endswith(b) and k not in fails])
            continue
        benches.append(Benchmark(args, b, *benchmarks[b]))

    # # Clear output file before appending
    # if (os.path.isfile(args.output)): os.system("rm " + args.output)

    # filter out benches that were already ran or failed to compile
    outfile = sys.stdout
    recorded_benchmarks = set()
    if args.output:
        if os.path.isfile(args.output):
            outfile = open(args.output, 'r+t')
            for line in outfile:
                bench, *rest = line.split(',')
                recorded_benchmarks.add(bench)
                print("RECORDDED BENCH: " + bench) # jenny added
            outfile.seek(0, 2) # seek to end of the file.
        else:
            outfile = open(args.output, 'w+t')

    filtered_benches = [b for b in benches if b.name not in recorded_benchmarks]
    num_filtered_benches = len(benches) - len(filtered_benches)
    if num_filtered_benches:
        print(f"Filtered out {num_filtered_benches} benchmarks."
              " Results already exists in the output file.", flush=True)
    benches = filtered_benches

    # Print out a list of benchmarks to be compiled
    print("*************FROM JSON FILE - START LIST*************")
    for b in benches:
        print(b.name)
    print("**************FROM JSON FILE - END LIST**************")

    # Create lock for synchronization
    lock = manager.Lock()

    t0 = time.time()

    try:
        # Compile with multiprocessing
        with multiprocessing.Pool() as p:
            p.starmap(comp, [(b, benches, args.output, lock) for b in benches])
    except Exception as e:
        print("Compilation failed, exiting")
        print(e)
        sys.exit(1)

    t_compiled = time.time()

    print("hello") # jenny added

    # Print out a list of benchmarks after compiling (failed-to-compile ones are removed)
    print("**********COMPILED BENCHMARKS (READY TO RUN) - START LIST**********")
    for b in benches:
        print(b.name)
    print("**********COMPILED BENCHMARKS (READY TO RUN) - END LIST**********")

    for i, b in enumerate(benches, 1):
        try:
            print(f"running {i}/{len(benches)}: {b.name}", flush=True)

            if args.warmup:
                print("***" + b.name + " warmup***")
                b.run()

            # Run args.repeat times (set the min to the first run, then compare) (if output is not float, directly attatch)
            print("***Running " + b.name + " (1/" + str(args.repeat) + ") ***")
            res = b.run()
            min_value = res[2]
            res[2] = str(res[2])
            print("***Done " + b.name + " (1/" + str(args.repeat) + ") ***")
            for i in range(2, args.repeat + 1):
                print("***Running " + b.name + " (" + str(i) + "/" + str(args.repeat) + ") ***")
                res_sub = b.run()
                if type(min_value) is float:
                    if res_sub[2] < min_value:
                        min_value = res_sub[2]
                        res_sub[2] = str(res_sub[2])
                        res = res_sub
                else:
                    res.append("\n")
                    for r in res_sub:
                        res.append(r)
                print("***Done " + b.name + " (" + str(i) + "/" + str(args.repeat) + ") ***")
            print(b.name + "," + ",".join(res), file=outfile)
            print("***write to file - success***")
        except Exception as err:
            print("Error running: ", b.name)
            print(err)
            outfile.write(b.name + ",COMPILED,ERR," + str(err).replace('\n', ' ').replace(',', ' ') + ",N/A," + str(b.res_regex).replace(',', ' ') + "\n")
            print("***write to file - error***")
        print("####DONE BENCHMARK - " + b.name + "####")

    if args.output:
        outfile.close()

    t_done = time.time()
    print("compilation took {} s, running took {} s.".format(t_compiled-t0, t_done-t_compiled))

if __name__ == "__main__":
    main()
