#!/usr/bin/env python

from optparse import OptionParser
import os
import subprocess
import os
this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
import sys
sys.path.insert(0,os.path.join(this_directory,"..","job_launching"))
import common
import re
import shutil
import glob
import datetime
import yaml
import common
import re
import datetime
import time

# We will look for the benchmarks 
parser = OptionParser()
parser.add_option("-B", "--benchmark_list", dest="benchmark_list",
                 help="a comma seperated list of benchmark suites to run. See apps/define-*.yml for " +\
                       "the benchmark suite names.",
                 default="rodinia_2.0-ft")
parser.add_option("-D", "--device_num", dest="device_num",
                 help="CUDA device number",
                 default="0")
parser.add_option("-n", "--norun", dest="norun", action="store_true", 
                 help="Do not actually run the apps, just create the dir structure and launch files")
parser.add_option("-l", "--limit_kernel_number", dest='kernel_number', default=-99, help="Sets a hard limit to the " +\
                        "number of traced limits")

(options, args) = parser.parse_args()

common.load_defined_yamls()

benchmarks = []
benchmarks = common.gen_apps_from_suite_list(options.benchmark_list.split(","))

cuda_version = common.get_cuda_version( this_directory )
now_time = datetime.datetime.now()
day_string = now_time.strftime("%y.%m.%d-%A")
time_string = now_time.strftime("%H:%M:%S")
logfile = day_string + "--" + time_string + ".csv"

nvbit_tracer_path = os.path.join(this_directory, "tracer_tool")

trace_dir = "/data/accel-sim_trace/hw_run/traces"
custom_dir = input(f"trace directory (default: {trace_dir}) : ")
run_time = time.time()
for bench in benchmarks:
    edir, ddir, exe, argslist = bench
    for argpair in argslist:
        args = argpair["args"]
        run_name = os.path.join( exe, common.get_argfoldername( args ) )

        if custom_dir != "":
            trace_dir = custom_dir
        this_run_dir = os.path.abspath(os.path.expandvars(
            os.path.join(trace_dir,"device-" + options.device_num, cuda_version, run_name)))
        this_trace_folder = os.path.join(this_run_dir, "traces")
        if not os.path.exists(this_run_dir):
            os.makedirs(this_run_dir)
        if not os.path.exists(this_trace_folder):
            os.makedirs(this_trace_folder)

        # link the data directory
        try:
            benchmark_data_dir = common.dir_option_test(os.path.join(ddir,exe,"data"),"",this_directory)
            if os.path.lexists(os.path.join(this_run_dir, "data")):
                os.remove(os.path.join(this_run_dir, "data"))
            os.symlink(benchmark_data_dir, os.path.join(this_run_dir,"data"))
        except common.PathMissing:
            pass

        all_data_link = os.path.join(this_run_dir,"data_dirs")
        if os.path.lexists(all_data_link):
            os.remove(all_data_link)
        top_data_dir_path = common.dir_option_test(ddir, "", this_directory)
        os.symlink(top_data_dir_path, all_data_link)

        if args == None:
            args = ""

        exec_path = common.file_option_test(os.path.join(edir, exe),"",this_directory)
        sh_contents = ""

        if('mlperf' in exec_path):
            exec_path = '. '+exec_path
            if(options.kernel_number > 0):
                os.environ['DYNAMIC_KERNEL_LIMIT_END'] = str(options.kernel_number)                
            else:
                os.environ['DYNAMIC_KERNEL_LIMIT_END'] = '1000'
        else:
            if(options.kernel_number > 0):
                os.environ['DYNAMIC_KERNEL_LIMIT_END'] = str(options.kernel_number)
            else:
                os.environ['DYNAMIC_KERNEL_LIMIT_END'] = '0'

	# first we generate the traces (.trace and kernelslist files)
	# then, we do post-processing for the traces and generate (.traceg and kernelslist.g files)
	# then, we delete the intermediate files ((.trace and kernelslist files files)
        sh_contents += "\nexport CUDA_VERSION=\"" + cuda_version + "\"; export CUDA_VISIBLE_DEVICES=\"" + options.device_num + "\" ; " +\
            "LD_PRELOAD=" + os.path.join(nvbit_tracer_path, "tracer_tool.so") + " " + exec_path +\
            " " + str(args) + " ; " + os.path.join(nvbit_tracer_path,"traces-processing", "post-traces-processing") + " " +\
            os.path.join(this_trace_folder, "kernelslist") + " ; rm -f " + this_trace_folder + "/*.trace ; rm -f " + this_trace_folder + "/kernelslist "

        print ("sh_contents: ", sh_contents)
        print ("this_run_dir: ", this_run_dir)
        print ("this_dir", this_directory)

        open(os.path.join(this_run_dir,"run.sh"), "w").write(sh_contents)
        if subprocess.call(['chmod', 'u+x', os.path.join(this_run_dir,"run.sh")]) != 0:
            exit("Error chmod runfile")

        if not options.norun:
            saved_dir = os.getcwd()
            os.chdir(this_run_dir)
            now = datetime.datetime.now()
            print("day {} {}:{}".format(now.day, now.hour, now.minute) )
            print ("Running {0}".format(exe))
            sys.stderr.write("Running {0}\n".format(exe))
            sys.stderr.write("day {} {}:{}\n".format(now.day, now.hour, now.minute))
            sys.stderr.write("runtime:{}\n".format(time.time() - run_time))
            run_time = time.time()

            if subprocess.call(["bash", "run.sh"]) != 0:
                print ("Error invoking nvbit on {0}".format(this_run_dir))
            os.chdir(saved_dir)
