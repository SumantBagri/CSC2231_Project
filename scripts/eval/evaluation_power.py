import subprocess
import time
import re
import csv
import os

tegrastats_log_file_path = "tegrastats.txt"
output_file = "power.csv"

# Check if the tegrastats.txt file already exists (so that the power stats are for the current run only)
if os.path.exists(tegrastats_log_file_path):
    os.remove(tegrastats_log_file_path)

# Define the command to run
command = "tegrastats --interval 250 --logfile "+tegrastats_log_file_path

# Run the command in the background
process = subprocess.Popen(command.split())

# Wait for 5 seconds
time.sleep(5)

# Terminate the process
process.terminate()

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
header = ["POM_5V_IN", "POM_5V_GPU", "POM_5V_CPU"]
mode = "w" if not os.path.isfile(output_file) else "a"
with open(output_file, mode) as f:
    writer = csv.writer(f)
    if mode == "w":
        writer.writerow(header)
    writer.writerows(pom_values)

# Print the maximum POM values
max_values = [max(col) for col in zip(*pom_values)]
print("Maximum POM_5V_IN value:", max_values[0])
print("Maximum POM_5V_GPU value:", max_values[1])
print("Maximum POM_5V_CPU value:", max_values[2])

# Calculate and print the average POM values
num_values = len(pom_values)
avg_values = [sum(col)/num_values for col in zip(*pom_values)]
print("Average POM_5V_IN value:", avg_values[0])
print("Average POM_5V_GPU value:", avg_values[1])
print("Average POM_5V_CPU value:", avg_values[2])
