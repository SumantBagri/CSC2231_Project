import subprocess
import re
import os


class PowerReader:
    def __init__(self) -> None:
        self.log_file_path = "tegrastats.txt"
        if os.path.exists(self.log_file_path):
            os.remove(self.log_file_path)

        self.cmd = "tegrastats --interval 250 --logfile " + self.log_file_path

        self.pattern = r"POM_5V_IN (\d+)/\d+ POM_5V_GPU (\d+)/\d+ POM_5V_CPU (\d+)/\d+"

    def _parse_data(self) -> None:
        # Open the log file and read its contents
        with open(self.log_file_path, "r") as f:
            log_contents = f.read()

        # Find all matches for the regex pattern
        matches = re.findall(self.pattern, log_contents)

        # Extract the POM values from each match
        pom_values = [
            [int(match[0]), int(match[1]), int(match[2])] for match in matches
        ]

        # Peak POM values
        self.max_values = [max(col) for col in zip(*pom_values)]

        # Calculate the average POM values and round to nearest integer
        self.avg_values = [round(sum(col) / len(col)) for col in zip(*pom_values)]

    def start(self) -> None:
        self.process = subprocess.Popen(self.cmd.split())

    def stop(self) -> None:
        self.process.terminate()
        self._parse_data()
        os.remove(self.log_file_path)

    def print_vals(self) -> None:
        "", "POM_5V_GPU max (mW)", "POM_5V_CPU max (mW)", "POM_5V_IN avg (mW)", "POM_5V_GPU avg (mW)", "POM_5V_CPU avg (mW)"
        print(f"POM_5V_IN  max \t =====> \t {self.max_values[0]} mW")
        print(f"POM_5V_CPU max \t =====> \t {self.max_values[1]} mW")
        print(f"POM_5V_GPU max \t =====> \t {self.max_values[2]} mW")
        print(f"POM_5V_IN  avg \t =====> \t {self.avg_values[0]} mW")
        print(f"POM_5V_CPU avg \t =====> \t {self.avg_values[1]} mW")
        print(f"POM_5V_GPU avg \t =====> \t {self.avg_values[2]} mW")
