import csv
import os
import re
import subprocess
import time

class Stat:
    def __init__(self) -> None:
        self.pattern    = None
        self.header     = list()
        self.maxvals    = list()
        self.avgvals    = list()
        self.fpath      = ''
        self.bval       = 0

class BaseReader:
    def __init__(self,
                 pwr_save_path: str,
                 mem_save_path: str,
                 logpath: str) -> None:
        self.log_file_path = logpath
        if os.path.exists(self.log_file_path):
            os.remove(self.log_file_path)
        
        self.cmd = None
        self.probe_cmd = None
        self.metrics = {'pwr': Stat(),
                        'mem': Stat()}
        self.units   = {'pwr': 'W',
                        'mem': 'MB'}

        # set save file paths
        self.metrics['pwr'].fpath = pwr_save_path
        self.metrics['mem'].fpath = mem_save_path

    def _parse_data(self) -> list:
        # Open the log file and read its contents
        with open(self.log_file_path, "r") as f:
            log_contents = f.read()
        
        return log_contents

    def start(self) -> None:
        self.probe()
        self.process = subprocess.Popen(self.cmd.split())

    def stop(self) -> None:
        self.process.terminate()
        time.sleep(2)
        self._parse_data()
        os.remove(self.log_file_path)

    def write_header(self, stat_name):
        stat = self.metrics[stat_name]
        # Write header
        with open(stat.fpath, mode='a') as file:
            writer = csv.writer(file)
            writer.writerow(stat.header)

    def write_row(self, num_it, stat_name) -> None:
        stat = self.metrics[stat_name]
        res = [num_it] + stat.maxvals + stat.avgvals
        with open(stat.fpath, mode='a') as file:
            writer = csv.writer(file)
            writer.writerow(res)
    
    def probe(self) -> list:
        result = subprocess.run(self.probe_cmd.split(' '), stdout=subprocess.PIPE)
        result = result.stdout.decode('utf-8')
        return result


class RTXReader(BaseReader):
    def __init__(self,
                 pwr_save_path: str,
                 mem_save_path: str,
                 logpath: str = "nvidiasmi.txt") -> None:
        super().__init__(pwr_save_path, mem_save_path, logpath)
        self.cmd = "nvidia-smi --query-gpu=memory.used,power.draw --format=csv,noheader -lms 250 -f " + self.log_file_path
        self.probe_cmd = "nvidia-smi --query-gpu=memory.used,power.draw --format=csv,noheader"
        
        # Set power attributes
        self.metrics['pwr'].pattern = r"([\d.]+) W"
        self.metrics['pwr'].header  = ["RUN", "PWR_MAX", "PWR_AVG"]

        # set memory attributes
        self.metrics['mem'].pattern = r"([\d]+) M"
        self.metrics['mem'].header = ["RUN", "MEM_MAX", "MEM_AVG"]
    
    def _parse_data(self) -> None:
        log_contents = super()._parse_data()
        
        for _, stat in self.metrics.items():
            # Find all matches for the regex pattern
            matches = re.findall(stat.pattern, log_contents)

            # Extract the values from each match
            values = [round(float(match)) for match in matches]

            # Peak values
            stat.maxvals = [max(values) - stat.bval]

            # Average values
            stat.avgvals = [round(sum(values) / len(values)) - stat.bval]
    
    def probe(self) -> list:
        result = super().probe()
        parsed_res = {
            'pwr' : round(float(re.findall(self.metrics['pwr'].pattern, result)[0])),
            'mem' : round(float(re.findall(self.metrics['mem'].pattern, result)[0]))
        }
        for name, stat in self.metrics.items():
            if name == 'mem':
                stat.bval = parsed_res[name]
            print(f"{name.upper()} : {stat.bval} {self.units[name]}")



class JetsonReader(BaseReader):
    def __init__(self,
                 pwr_save_path: str,
                 mem_save_path: str,
                 logpath: str = "tegrastat.txt") -> None:
        super().__init__(pwr_save_path, mem_save_path, logpath)
        self.cmd = "tegrastats --interval 250 --logfile " + self.log_file_path
        self.probe_cmd = "tegrastats --interval 0 | head -n 1"
        
        # Set power attributes
        self.metrics['pwr'].pattern = r"POM_5V_IN (\d+)/\d+ POM_5V_GPU (\d+)/\d+ POM_5V_CPU (\d+)/\d+"
        self.metrics['pwr'].header  = ["RUN", "PWR_IN_MAX", "PWR_CPU_MAX", "PWR_GPU_MAX",
                                       "PWR_IN_AVG", "PWR_CPU_AVG", "PWR_GPU_AVG"]

        # set memory attributes
        self.metrics['mem'].pattern = r"[^I]RAM (\d+)/"
        self.metrics['mem'].header = ["RUN", "MEM_MAX", "MEM_AVG"]
    
    def _parse_data(self) -> list:
        log_contents = super()._parse_data()

        pwr_stat = self.metrics['pwr']
        mem_stat = self.metrics['mem']

        # Find all matches for the regex pattern
        matches_pwr = re.findall(pwr_stat.pattern, log_contents)
        matches_mem = re.findall(mem_stat.pattern, log_contents)

        # Extract the POM and MEM values from each match
        pom_values = [
            [int(match[0]), int(match[1]), int(match[2])] for match in matches_pwr
        ]
        mem_values = [int(match) for match in matches_mem]

        # Peak POM/MEM values
        pwr_stat.maxvals = [max(col) for col in zip(*pom_values)]
        mem_stat.maxvals = [max(mem_values)]

        # Average POM/MEM values
        pwr_stat.avgvals = [round(sum(col) / len(col)) for col in zip(*pom_values)]
        mem_stat.avgvals = [round(sum(mem_values) / len(mem_values))]
    
    def probe(self) -> None:
        result = super().probe()

        match_pwr = re.findall(self.metrics['pwr'].pattern, result)
        match_mem = re.findall(self.metrics['mem'].pattern, result)

        parsed_res = {
            'pwr' : [int(m)//1000 for m in match_pwr],
            'mem' : int(match_mem[0])
        }
        for name, stat in self.metrics.items():
            if name == 'mem':
                stat.bval = parsed_res[name]
            print(f"{name.upper()} : {stat.pval} {self.units[name]}")
        

