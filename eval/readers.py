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
                 pwr_save_path: str = '',
                 mem_save_path: str = '',
                 lat_save_path: str = '',
                 logpath: str = '') -> None:
        self.log_file_path = logpath
        if os.path.exists(self.log_file_path):
            os.remove(self.log_file_path)
        
        self.cmd = None
        self.probe_cmd = None
        self.metrics = {'pwr': Stat(),
                        'mem': Stat(),
                        'lat': Stat()}
        self.units   = {'pwr': 'W',
                        'mem': 'MB',
                        'lat': 's'}

        # set save file paths
        self.metrics['pwr'].fpath = pwr_save_path
        self.metrics['mem'].fpath = mem_save_path
        self.metrics['lat'].fpath = lat_save_path

        self.metrics['lat'].maxvals = [-1.,-1.,-1.]
        self.metrics['lat'].header = ["RUN", "DEN_LAT", "DEC_LAT", "TOTAL_LAT"]

    def _parse_data(self) -> list:
        # Open the log file and read its contents
        with open(self.log_file_path, "r") as f:
            log_contents = f.read()
        
        return log_contents

    def start(self) -> None:
        self.process = subprocess.Popen(self.cmd.split())
        self.metrics['lat'].bval = time.time()

    def stop(self) -> None:
        self.metrics['lat'].maxvals[2] = round(time.time() - self.metrics['lat'].bval,4)
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
        result = subprocess.run(str.encode(self.probe_cmd), shell=True, stdout=subprocess.PIPE)
        result = result.stdout.decode('utf-8')
        return result


class RTXReader(BaseReader):
    def __init__(self,
                 pwr_save_path: str,
                 mem_save_path: str,
                 lat_save_path: str,
                 logpath: str = "nvidiasmi.txt") -> None:
        super().__init__(pwr_save_path, mem_save_path, lat_save_path, logpath)
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
            # for pwr and mem
            if stat.pattern:
                # Find all matches for the regex pattern
                matches = re.findall(stat.pattern, log_contents)
                # Extract the values from each match
                values = [round(float(match),4) for match in matches]
                # Peak values
                stat.maxvals = [max(values) - stat.bval]
                # Average values
                stat.avgvals = [round(sum(values) / len(values) - stat.bval,4)]
    
    def probe(self,print_stdout=False) -> list:
        result = super().probe()
        parsed_res = {
            'pwr' : round(float(re.findall(self.metrics['pwr'].pattern, result)[0]),4),
            'mem' : round(float(re.findall(self.metrics['mem'].pattern, result)[0]),4)
        }
        for name, stat in self.metrics.items():
            if name == 'mem':
                stat.bval = parsed_res[name]
            if print_stdout and name != 'lat':
                print(f"{name.upper()} : {stat.bval} {self.units[name]}")



class JetsonReader(BaseReader):
    def __init__(self,
                 pwr_save_path: str,
                 mem_save_path: str,
                 lat_save_path: str,
                 logpath: str = "tegrastat.txt") -> None:
        super().__init__(pwr_save_path, mem_save_path, lat_save_path, logpath)
        self.cmd = "/usr/bin/tegrastats --interval 250 --logfile " + self.log_file_path
        self.probe_cmd = "/usr/bin/tegrastats --interval 0 | head -n 1"
        
        # Set power attributes
        self.metrics['pwr'].pattern = r"POM_5V_IN (\d+)/\d+ POM_5V_GPU (\d+)/\d+ POM_5V_CPU (\d+)/\d+"
        self.metrics['pwr'].header  = ["RUN", "PWR_IN_MAX", "PWR_CPU_MAX", "PWR_GPU_MAX",
                                       "PWR_IN_AVG", "PWR_CPU_AVG", "PWR_GPU_AVG"]

        # set memory attributes
        self.metrics['mem'].pattern = r"\bRAM (\d+)/"
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
            [round(float(match[0])),
             round(float(match[1])),
             round(float(match[2]))] for match in matches_pwr
        ]
        mem_values = [round(float(match)) for match in matches_mem]

        # Peak POM/MEM values
        pwr_stat.maxvals = [max(col) - pwr_stat.bval for col in zip(*pom_values)]
        mem_stat.maxvals = [max(mem_values) - mem_stat.bval]

        # Average POM/MEM values
        pwr_stat.avgvals = [round(sum(col) / len(col) - pwr_stat.bval, 4) for col in zip(*pom_values)]
        mem_stat.avgvals = [round(sum(mem_values) / len(mem_values) - mem_stat.bval, 4)]
    
    def probe(self) -> None:
        result = super().probe()

        match_pwr = re.findall(self.metrics['pwr'].pattern, result)
        match_mem = re.findall(self.metrics['mem'].pattern, result)

        parsed_res = {
            'pwr' : [round(float(m)/1000,4) for m in match_pwr],
            'mem' : round(float(match_mem[0]),4)
        }
        for name, stat in self.metrics.items():
            if name == 'mem':
                stat.bval = parsed_res[name]
            print(f"{name.upper()} : {stat.pval} {self.units[name]}")
        

