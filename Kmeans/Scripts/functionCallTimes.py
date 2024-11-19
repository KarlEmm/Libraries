import csv
import matplotlib.pyplot as plt
import os
import subprocess

from io import StringIO
from os import listdir
from os.path import isfile, join

profiling_directory = "/home/karlemmb/Documents/VSCProjects/Libraries/Kmeans/ProfilingData/"

env_vars = os.environ.copy()
env_vars["XRAY_OPTIONS"] = "xray_naive_log=true patch_premain=true xray_mode=basic xray_logfile_base={}".format(profiling_directory)

for i in range(1000):
    try:
        subprocess.run(['./main'], capture_output=True, text=True, check=True, env=env_vars)
    except subprocess.CalledProcessError as e:
        print("Error:", e)
        print("stderr:", e.stderr)

column_names = {'funcid':0, '99ile':5, 'funcname':9}

profiled_function_name = "kMeansClustering"
data = []
profiling_files = [join(profiling_directory, f) for f in listdir(profiling_directory) if isfile(join(profiling_directory, f))]
for index, file in enumerate(profiling_files):
    try:
        #output_file = open(join(profiling_directory, 'account.' + str(index)), "w")
        result = subprocess.run(['llvm-xray', 'account', file, '--format=csv', '--sort=func', '--sortorder=dsc', '--instr_map=main'], text=True, check=True, capture_output=True)
        lines = result.stdout.splitlines()
        for line in lines:
            csvLine = StringIO(line)
            reader = csv.reader(csvLine)
            tokens = next(reader)
            if profiled_function_name in tokens[column_names['funcname']]:
                data.append(float(tokens[column_names['99ile']]))   
                break

    except subprocess.CalledProcessError as e:
        print("Error:", e)
        print("stderr:", e.stderr)

with open(join(profiling_directory, 'Timings' ,'kMeansClusteringPPSTTimings.data'), "w") as f:
    f.write(str(data))

plt.hist(data, bins=50)
plt.xlabel('Index')
plt.ylabel('Time (s)')
plt.title('FindNearestCentroid')
plt.show()
