import onnxruntime as ort
import os
import torch

from readers import JetsonReader, RTXReader

class BaseEvaluator:
    def __init__(self, 
                 output_dir = "output/eval_data/",
                 dev        = None, # ['rtx_3070', 'jetson_nano', 'android']
                 perf_cls   = "baseline" # ['baseline','optim1','optim2','optim3']
                ):
        self.out_dir = output_dir

        # define the device and performance class in child class
        self._dev       = dev
        self.perf_cls   = perf_cls

        # define the output file paths
        self._fpaths = {
            'stdout'    : self.out_dir+'_'.join([self._dev,self.perf_cls,'raw.stdout']),
            'latency'   : self.out_dir+'_'.join([self._dev,self.perf_cls,'latency.csv']),
            'memory'    : self.out_dir+'_'.join([self._dev,self.perf_cls,'memory.csv']),
            'power'     : self.out_dir+'_'.join([self._dev,self.perf_cls,'power.csv'])
        }

        # Define stat readers
        if dev == 'rtx_3070':
            self.reader = RTXReader(pwr_save_path=self._fpaths['power'],
                                    mem_save_path=self._fpaths['memory'],
                                    lat_save_path=self._fpaths['latency'])
        elif dev == 'jetson_nano':
            self.reader = JetsonReader(pwr_save_path=self._fpaths['power'],
                                       mem_save_path=self._fpaths['memory'],
                                       lat_save_path=self._fpaths['latency'])
        else:
            raise NotImplementedError

    def _clear_data(self):
        for fpath in self._fpaths.values():
            if os.path.exists(fpath):
                os.remove(fpath)

    def _stdout_writer(self, num_it, result):
        with open(self._fpaths['stdout'], mode='a') as file:
            file.write(f"\nRun number {num_it}:\n")
            file.write(str(result))

    def evaluate(self, model, inp, num_iters=10) -> None:
        self._clear_data()
        for sname in ['pwr','mem','lat']:
            self.reader.write_header(sname)

        if hasattr(model, "eval"): model.eval()
        if hasattr(model, "warmup"): model.warmup()

        for i in range(num_iters):
            print(f"Evaluation iter: {i}")
            # perform profiling
            self.reader.start()
            model(inp)
            torch.cuda.synchronize()
            self.reader.stop()
            # write to csv files
            self.reader.write_row(i, 'pwr') # write power to file
            self.reader.write_row(i, 'mem') # write memory to file
            self.reader.write_row(i, 'lat') # write latency to file

class ONNXEvaluator(BaseEvaluator):
    def __init__(self,
                 mpath,
                 output_dir="output/eval_data/",
                 dev=None,
                 perf_cls="baseline",
                 providers=["CUDAExecutionProvider"]):
        super().__init__(output_dir, dev, perf_cls)

        self.providers = providers

        self.pad = " "*100
        print("Creating Inference Session..."+self.pad+"\r",end='')
        self.reader.probe()
        self.sess = ort.InferenceSession(mpath, providers=self.providers)
        print("Inference Session Created!"+self.pad)

    def __del__(self):
        del self.sess
        torch.cuda.empty_cache()

    def evaluate(self, inp, num_iters=10) -> None:
        self._clear_data()
        self.reader.write_header('pwr')
        self.reader.write_header('mem')

        inp = inp.cpu().numpy()

        print("Warming up..."+self.pad+"\r",end='')
        self.sess.run([],{'input': inp})
        print("Warmup Completed!"+self.pad)

        for i in range(num_iters):
            print(f"Evaluation iter: {i}")
            # perform profiling
            self.reader.start()
            self.sess.run([],{'input': inp})
            torch.cuda.synchronize()
            self.reader.stop()
            # write to csv files
            self.reader.write_row(i, 'pwr') # write power to file
            self.reader.write_row(i, 'mem') # write memory to file
            self.reader.write_row(i, 'lat') # write latency to file