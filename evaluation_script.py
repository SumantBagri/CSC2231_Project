import subprocess
import re
import csv
import os
import time

output_file = 'output.txt'
csv_file = 'evaluation_resnet.csv'

tegrastats_log_file_path = "tegrastats.txt"
power_output_file = "power_resnet.csv"

# Command to run the tensorrt engine with the tensorrt profiler
command = "/usr/src/tensorrt/bin/trtexec --loadEngine=resnet50_pytorch.trt --batch=1"

# Define the command to run the tegrastats utility to profile the power
command2 = "tegrastats --interval 250 --logfile "+tegrastats_log_file_path

# Check if the tegrastats.txt file already exists (so that the power stats are for the current run only)
if os.path.exists(tegrastats_log_file_path):
    os.remove(tegrastats_log_file_path)

# Run the tegrastats utility in the background
process = subprocess.Popen(command2.split())
# Take 3 seconds of power measurements before the model inference
time.sleep(0.5) # <----------------------------------------------------------------DECREASE TO 1 SECOND (OR LESS)?? TO DECREASE IMPACT ON AVERAGES

# Run the tensorrt engine with the tensorrt profiler and capture its output
output = subprocess.check_output(command.split())

# Take 3 seconds of power measurements after the model inference
time.sleep(0.5) # <----------------------------------------------------------------DECREASE TO 1 SECOND (OR LESS)?? TO DECREASE IMPACT ON AVERAGES
# Terminate the tegrastats utility in the background
process.terminate()

# Save the tensorrt profiler output to a file
with open('output.txt', 'w') as f:
    f.write(output.decode())
    
with open(output_file, 'r') as f:
    output = f.read()

# Extract the latency information
latency_pattern = r'End-to-End Host Latency: min = (\d+\.\d+) ms, max = (\d+\.\d+) ms, mean = (\d+\.\d+) ms, median = (\d+\.\d+) ms, percentile\(99%\) = (\d+\.\d+) ms'
latency_matches = re.search(latency_pattern, output)
latency_values = [float(val) for val in latency_matches.groups()]

# Extract the memory usage information
pattern = r"Init CUDA: CPU \+(\d+), GPU \+(\d+), now: CPU (\d+), GPU (\d+)"
match = re.search(pattern, output)

init_cpu = int(match.group(1))
init_gpu = int(match.group(2))
now_cpu = int(match.group(3))
now_gpu = int(match.group(4))

cpu_mem_usage = now_cpu - init_cpu
gpu_mem_usage = now_gpu - init_gpu

# Extract the power information

# Define the regex pattern to extract POM values
pattern = r'POM_5V_IN (\d+)/\d+ POM_5V_GPU (\d+)/\d+ POM_5V_CPU (\d+)/\d+'

# Open the log file and read its contents
with open(tegrastats_log_file_path, "r") as f:
    log_contents = f.read()

# Find all matches for the regex pattern
matches = re.findall(pattern, log_contents)

# Extract the POM values from each match
pom_values = [[int(match[0]), int(match[1]), int(match[2])] for match in matches]

# Write the POM values to the output CSV file
header = ["POM_5V_IN (mW)", "POM_5V_GPU (mW)", "POM_5V_CPU (mW)"]
mode = "w" if not os.path.isfile(power_output_file) else "a"
with open(power_output_file, mode) as f:
    writer = csv.writer(f)
    if mode == "w":
        writer.writerow(header)
    writer.writerows(pom_values)

# Peak POM values
max_values = [max(col) for col in zip(*pom_values)]

# Calculate the average POM values and round to nearest integer
avg_values = [round(sum(col)/len(col)) for col in zip(*pom_values)]
    
# Write the run stats to CSV

column_names = ['min latency (ms)', 'max latency (ms)', 'mean latency (ms)', 'median latency (ms)', '99% latency (ms)', 'cpu mem usage (MiB)', 'gpu mem usage (MiB)', 'POM_5V_IN max (mW)', 'POM_5V_GPU max (mW)', 'POM_5V_CPU max (mW)', 'POM_5V_IN avg (mW)', 'POM_5V_GPU avg (mW)', 'POM_5V_CPU avg (mW)']

column_values = latency_values.copy()
column_values.append(cpu_mem_usage)
column_values.append(gpu_mem_usage)
column_values = column_values + [max_values[i] for i in range(3)] + [avg_values[i] for i in range(3)]
row_data = dict(zip(column_names, column_values))

if not os.path.isfile(csv_file) or os.stat(csv_file).st_size == 0:
    # file doesn't exist or is empty
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=column_names)
        writer.writeheader()
        writer.writerow(row_data)
else:
    # file exists and has data
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=column_names)
        writer.writerow(row_data)
