from threading import Lock

import numpy as np


class GpuResourceScheduler:
    def __init__(self, available_devices):
        self.available_devices = available_devices
        self.gpu_count = len(available_devices)
        self.used_gpu = np.zeros(self.gpu_count, dtype=np.int32)
        self.gpu_lock = Lock()

    def get_gpu_id(self):
        with self.gpu_lock:
            min_arg = np.argmin(self.used_gpu)
            self.used_gpu[min_arg] += 1
            return self.available_devices[min_arg]

    def return_gpu_id(self, gpu_id):
        with self.gpu_lock:
            idx = self.available_devices.index(gpu_id)
            self.used_gpu[idx] -= 1
