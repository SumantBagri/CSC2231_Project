import csv

def csv_writer(filename):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            with open(filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(result)
            return result
        return wrapper
    return decorator

class Evaluator:
    def __init__(self, output_filename):
        self.output_filename = output_filename
    
    @csv_writer(output_filename)
    def get_latency(self):
        # code to get latency
        return [self.__class__.__name__, 'latency', latency_result]
    
    @csv_writer(output_filename)
    def get_memory(self):
        # code to get memory usage
        return [self.__class__.__name__, 'memory', memory_result]
    
    @csv_writer(output_filename)
    def get_power(self):
        # code to get power consumption
        return [self.__class__.__name__, 'power', power_result]
