try:
    import psutil
except ImportError:
    psutil = None

try:
    import GPUtil
except ImportError:
    GPUtil = None

try:
    import cpuinfo
except ImportError:
    cpuinfo = None


class HardwareProfiler:
    """
    A class to detect and profile system hardware components like CPU,
    GPU, RAM, and Storage.

    It gracefully handles missing libraries and ensures that methods
    return predictable data types.
    """
    def __init__(self):
        self.cpu_info = self._detect_cpu()
        self.memory_info = self._detect_memory()
        self.gpu_info = self._detect_gpu()
        self.storage_info = self._detect_storage()

    def _detect_gpu(self):
        """
        Detects GPU information.

        Returns:
            list: A list of dictionaries, one for each detected GPU.
                  Returns an empty list if GPUtil is not installed or no
                  GPU is found.
        """
        if not GPUtil:
            return [] 

        try:
            gpus = GPUtil.getGPUs()
            gpu_list = [{
                "name": gpu.name,
                "load": f"{gpu.load * 100:.1f}%",
                "vram_total_gb": round(gpu.memoryTotal / 1024, 2), 
                "vram_used_gb": round(gpu.memoryUsed / 1024, 2),
                "vram_free_gb": round(gpu.memoryFree / 1024, 2),
                "driver": gpu.driver,
                "temperature_c": gpu.temperature
            } for gpu in gpus]
            return gpu_list
        except Exception:
            return []


    def _detect_cpu(self):
        """
        Detects CPU information using both psutil and cpuinfo.

        Returns:
            dict: A dictionary of CPU details. Returns a dictionary
                  with 'N/A' values if libraries are missing.
        """
        if not psutil or not cpuinfo:
            return {
                "brand": "N/A (py-cpuinfo or psutil not installed)",
                "cores": "N/A", "threads": "N/A", "hz": "N/A"
            }

        info = {}
        try:
            cpu_details = cpuinfo.get_cpu_info()
            info["brand"] = cpu_details.get('brand_raw', 'N/A')
            info["hz"] = cpu_details.get('hz_friendly', 'N/A')
            info["cores"] = psutil.cpu_count(logical=False)
            info["threads"] = psutil.cpu_count(logical=True)

        except Exception:
            return {
                "brand": "Error during detection", "cores": "N/A",
                "threads": "N/A", "hz": "N/A"
            }
        return info

    def _detect_memory(self):
        """
        Detects system memory (RAM) information.

        Returns:
            dict: A dictionary of RAM details, with 'N/A' values
                  on failure or if psutil is not installed.
        """
        if not psutil:
            return {"total_gb": "N/A", "available_gb": "N/A", "used_gb": "N/A", "percentage": "N/A"}

        try:
            svmem = psutil.virtual_memory()
            return {
                "total_gb": round(svmem.total / (1024 ** 3), 2),
                "available_gb": round(svmem.available / (1024 ** 3), 2),
                "used_gb": round(svmem.used / (1024 ** 3), 2),
                "percentage": svmem.percent
            }
        except Exception:
            return {"total_gb": "N/A", "available_gb": "N/A", "used_gb": "N/A", "percentage": "N/A"}


    def _detect_storage(self):
        """
        Detects disk storage information for all partitions.

        Returns:
            list: A list of dictionaries, one for each partition.
                  Returns an empty list if psutil is not installed.
        """
        if not psutil:
            return [] 

        storage_list = []
        try:
            partitions = psutil.disk_partitions()
            for partition in partitions:
                if 'cdrom' in partition.opts or partition.fstype == '':
                    continue
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    storage_list.append({
                        "device": partition.device,
                        "mountpoint": partition.mountpoint,
                        "fstype": partition.fstype,
                        "total_gb": round(usage.total / (1024 ** 3), 2),
                        "used_gb": round(usage.used / (1024 ** 3), 2),
                        "free_gb": round(usage.free / (1024 ** 3), 2),
                        "percentage": usage.percent
                    })
                except (PermissionError, FileNotFoundError):
                    continue
        except Exception:
            return [] 
            
        return storage_list
