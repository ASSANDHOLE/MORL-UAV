from multiprocessing.shared_memory import SharedMemory

import numpy as np


class GpuResourceScheduler:
    def __init__(self, available_devices, lock, limit_per_device=None):
        self.available_devices = available_devices
        self.gpu_count = len(available_devices)
        arr = np.zeros(self.gpu_count, dtype=np.int32)
        self.used_gpu = SharedMemory(create=True, size=arr.nbytes)
        self.buffer_name = self.used_gpu.name
        arr_b = np.ndarray(arr.shape, dtype=arr.dtype, buffer=self.used_gpu.buf)
        arr_b[:] = arr[:]
        self.limit_per_device = int(limit_per_device) if limit_per_device is not None else 999999
        self.gpu_lock = lock

    def __del__(self):
        self.used_gpu.close()
        shm = SharedMemory(name=self.buffer_name)
        shm.close()
        shm.unlink()

    def get_gpu_id(self):
        with self.gpu_lock:
            shm = SharedMemory(name=self.buffer_name)
            used_gpu = np.ndarray((self.gpu_count,), dtype=np.int32, buffer=shm.buf)
            min_arg = np.argmin(used_gpu)
            if used_gpu[min_arg] >= self.limit_per_device:
                shm.close()
                return None
            used_gpu[min_arg] += 1
            shm.close()
            return self.available_devices[min_arg]

    def return_gpu_id(self, gpu_id):
        with self.gpu_lock:
            shm = SharedMemory(name=self.buffer_name)
            used_gpu = np.ndarray((self.gpu_count,), dtype=np.int32, buffer=shm.buf)
            idx = self.available_devices.index(gpu_id)
            used_gpu[idx] -= 1
            shm.close()
