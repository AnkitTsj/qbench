import psutil
import subprocess
import pandas as pd
import time
from hp import HardwareProfiler
import os
import re
import pandas as pd
import threading
import uuid
import argparse
from datetime import datetime

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
import psutil
import socket
try:
    import cpuinfo
except ImportError:
    cpuinfo = None

try:
    import GPUtil
except ImportError:
    GPUtil = None

def get_system_profile():
    """
    Gathers a cross-platform summary of the system's hardware profile.

    This revised function avoids platform-specific calls and provides
    more reliable information about all available GPUs.
    """
    profile = {}

    try:
        profile['hostname'] = socket.gethostname()
    except Exception:
        profile['hostname'] = "Unknown"

    try:
        if cpuinfo:
            profile['cpu'] = cpuinfo.get_cpu_info().get('brand_raw', 'Unknown')
        else:
            profile['cpu'] = "Unknown (py-cpuinfo not installed)"
        
        profile['cpu_threads'] = psutil.cpu_count(logical=True)
    except Exception:
        profile['cpu'] = "Unknown"
        profile['cpu_threads'] = 0

    try:
        profile['ram_total_gb'] = round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        profile['ram_total_gb'] = 0

    profile['gpus'] = []
    if GPUtil:
        try:
            gpu_list = GPUtil.getGPUs()
            for gpu in gpu_list:
                profile['gpus'].append({
                    'name': gpu.name,
                    'vram_total_gb': round(gpu.memoryTotal / 1024, 2) 
                })
        except Exception:
            pass
            
    return profile


class ResourceMonitor:
    def __init__(self):
        self.running = False
        self.cpu = []
        self.mem = []
        self.gpu_util = []
        self.vram = []
        self.power = []
        if NVML_AVAILABLE:
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        else:
            self.handle = None

    def _print_status(self, message: str, status: str = "INFO"):
        """Print status message with consistent formatting"""
        status_symbols = {
            "INFO": "[*]",
            "SUCCESS": "[+]",
            "ERROR": "[-]",
            "WARNING": "[!]",
            "PROGRESS": "[>]"
        }
        symbol = status_symbols.get(status, "[*]")
        print(f"{symbol} {message}")

    def start(self, interval=0.1):
        self.running = True
        self.thread = threading.Thread(target=self._sample, args=(interval,), daemon=True)
        self.thread.start()
        self._print_status("Resource monitoring started", "SUCCESS")

    def stop(self):
        self.running = False
        self.thread.join()
        self._print_status("Resource monitoring stopped", "INFO")

    def _sample(self, interval):
        proc = psutil.Process(os.getpid()) if psutil else None
        while self.running:
            if psutil:
                self.cpu.append(psutil.cpu_percent(interval=None))
                self.mem.append(proc.memory_info().rss / (1024 * 1024))
            if NVML_AVAILABLE:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                pwr = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                self.gpu_util.append(util.gpu)
                self.vram.append(meminfo.used / (1024 * 1024))
                self.power.append(pwr)
            time.sleep(interval)

    def summary(self):
        def _stat(lst, agg):
            return round(agg(lst),2) if lst else 0.0
        return {
            "peak_memory_usage_mb": _stat(self.mem, max),
            "avg_memory_usage_mb": _stat(self.mem, lambda l: sum(l)/len(l)),
            "peak_vram_usage_mb": _stat(self.vram, max),
            "avg_gpu_usage_percent": _stat(self.gpu_util, lambda l: sum(l)/len(l)),
            "peak_gpu_usage_percent": _stat(self.gpu_util, max),
            "avg_cpu_usage_percent": _stat(self.cpu, lambda l: sum(l)/len(l)),
            "power_consumption_watts": _stat(self.power, lambda l: sum(l)/len(l)),
        }

class QuantizationAnalyzer:
    def __init__(self, llama_simple_path, llama_perplexity_path, dataset_path, profiler=None):
        self.llama_simple = llama_simple_path
        self.llama_perplexity = llama_perplexity_path
        self.dataset = dataset_path
        self.profiler = profiler or HardwareProfiler()

    def _print_header(self, title: str):
        """Print a consistent header format"""
        print("\n" + "="*80)
        print(f">>> {title.upper()}")
        print("="*80)

    def _print_step(self, step_name: str):
        """Print a step indicator"""
        print(f"\n[STEP] {step_name}")
        print("-" * 60)

    def _print_status(self, message: str, status: str = "INFO"):
        """Print status message with consistent formatting"""
        status_symbols = {
            "INFO": "[*]",
            "SUCCESS": "[+]",
            "ERROR": "[-]",
            "WARNING": "[!]",
            "PROGRESS": "[>]"
        }
        symbol = status_symbols.get(status, "[*]")
        print(f"{symbol} {message}")
    
    def memory_profile(self):
        self._print_step("System Memory Profile")
        mem = self.profiler.memory_info
        gpus = self.profiler.gpu_info
        
        self._print_status("RAM Configuration:", "INFO")
        print(f"    {mem}")
        self._print_status("GPU Configuration:", "INFO")
        print(f"    {gpus}")

    

    def check_compatibility(self, fmt, model_path=None):
        """
        Dynamic compatibility assessment based on actual system resources
        and model requirements rather than rigid format rules
        """
        # Get actual system specifications
        system_ram_gb = psutil.virtual_memory().total / (1024**3) if psutil else 8
        available_ram_gb = psutil.virtual_memory().available / (1024**3) if psutil else 4
        cpu_cores = psutil.cpu_count(logical=False) if psutil else 4
        cpu_threads = psutil.cpu_count(logical=True) if psutil else 8
        

        gpu_available = False
        gpu_vram_gb = 0
        gpu_free_vram_gb = 0
        gpu_name = "Unknown"
        
        if NVML_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_name = pynvml.nvmlDeviceGetName(handle).decode()
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_vram_gb = mem_info.total / (1024**3)
                gpu_free_vram_gb = mem_info.free / (1024**3)
                gpu_available = True
            except:
                pass
        
        # Estimate model memory requirements
        model_size_gb = 0
        if model_path and os.path.exists(model_path):
            model_size_gb = os.path.getsize(model_path) / (1024**3)
        else:
            size_estimates = {
                'fp16': 8.0, 'f16': 8.0,
                'fp32': 16.0, 'f32': 16.0,
                'q8_0': 4.5, 'q8': 4.5,
                'q6_k': 3.5, 'q6': 3.5,
                'q5_k_m': 3.0, 'q5': 3.0,
                'q4_k_m': 2.5, 'q4': 2.5,
                'q3_k_m': 2.0, 'q3': 2.0,
                'q2_k': 1.5, 'q2': 1.5,
            }
            
            fmt_lower = fmt.lower()
            for key, size in size_estimates.items():
                if key in fmt_lower:
                    model_size_gb = size
                    break
            
            if model_size_gb == 0:
                model_size_gb = 3.0  # Default estimate
        
        # Calculate actual resource requirements
        # Model loading requires ~1.2x model size in RAM (overhead)
        ram_required = model_size_gb * 1.2
        # Context processing requires additional memory (~0.5-1GB for 2K context)
        context_overhead = 0.7
        total_ram_needed = ram_required + context_overhead
        
        # GPU VRAM requirements (if using GPU)
        vram_required = model_size_gb * 1.1  # 10% overhead for GPU
        
        fmt_lower = fmt.lower()
        compatibility_score = 0
        issues = []
        recommendations = []
        
        # MEMORY COMPATIBILITY ASSESSMENT
        if total_ram_needed <= available_ram_gb:
            compatibility_score += 40
        elif total_ram_needed <= system_ram_gb * 0.9:
            compatibility_score += 25
            issues.append("May cause memory pressure")
        else:
            compatibility_score -= 20
            issues.append(f"Insufficient RAM ({total_ram_needed:.1f}GB needed, {available_ram_gb:.1f}GB available)")
        
        # GPU COMPATIBILITY ASSESSMENT
        if gpu_available:
            if any(x in fmt_lower for x in ['fp16', 'f16']):
                # FP16 benefits significantly from GPU
                if vram_required <= gpu_free_vram_gb:
                    compatibility_score += 30
                    recommendations.append("Excellent GPU acceleration possible")
                else:
                    compatibility_score += 10
                    issues.append("GPU VRAM insufficient for full offload")
            
            elif any(x in fmt_lower for x in ['q4', 'q5', 'q6']):
                # Medium quantization can benefit from GPU
                if vram_required <= gpu_free_vram_gb:
                    compatibility_score += 20
                    recommendations.append("Good GPU acceleration possible")
                else:
                    compatibility_score += 5
                    issues.append("Partial GPU offload possible")
        else:
            # No GPU available
            if any(x in fmt_lower for x in ['fp16', 'f16']):
                compatibility_score -= 15
                issues.append("FP16 without GPU is very slow")
            elif any(x in fmt_lower for x in ['q2', 'q3', 'q4']):
                compatibility_score += 10
                recommendations.append("Good CPU performance expected")
        
        # CPU PERFORMANCE ASSESSMENT
        if cpu_threads >= 8:
            compatibility_score += 10
        elif cpu_threads >= 4:
            compatibility_score += 5
        else:
            issues.append("Limited CPU threads may slow inference")
        
        # FORMAT-SPECIFIC ADJUSTMENTS
        if 'k_m' in fmt_lower or 'k_s' in fmt_lower:
            compatibility_score += 5  
        
        if any(x in fmt_lower for x in ['q2', 'q3']):
            compatibility_score += 15
            recommendations.append("Excellent for resource-constrained systems")
        
        if compatibility_score >= 70:
            rating = "EXCELLENT"
            color = "SUCCESS"
        elif compatibility_score >= 50:
            rating = "GOOD"
            color = "SUCCESS"
        elif compatibility_score >= 30:
            rating = "FAIR"
            color = "WARNING"
        elif compatibility_score >= 10:
            rating = "POOR"
            color = "WARNING"
        else:
            rating = "INCOMPATIBLE"
            color = "ERROR"
        
        details = []
        details.append(f"Model size: {model_size_gb:.1f}GB")
        details.append(f"RAM needed: {total_ram_needed:.1f}GB (available: {available_ram_gb:.1f}GB)")
        
        if gpu_available:
            details.append(f"GPU: {gpu_name} ({gpu_free_vram_gb:.1f}GB free)")
        else:
            details.append("GPU: Not available")
        
        details.append(f"CPU: {cpu_threads} threads")
        
        if issues:
            details.append(f"Issues: {'; '.join(issues)}")
        
        if recommendations:
            details.append(f"Notes: {'; '.join(recommendations)}")
        
        compatibility_string = f"{rating} (Score: {compatibility_score}/100)"
        
        self._print_status(f"Compatibility Analysis for {fmt}:", color)
        for detail in details:
            print(f"    {detail}")
        
        return compatibility_string

    def measure_memory_usage(self, model_path, prompt="Hello", n_predict=16, ngl=35):
        model_name = os.path.basename(model_path)
        self._print_status(f"Measuring memory usage for {model_name}", "PROGRESS")
        
        process = psutil.Process()
        mem_before = psutil.virtual_memory().used
        cmd = f'"{self.llama_simple}" -m "{model_path}" -n {n_predict} -ngl {ngl} "{prompt}"'
        p = subprocess.Popen(cmd, shell=True)
        p.wait()
        mem_after = psutil.virtual_memory().used
        memory_used_mb = (mem_after - mem_before) / (1024**2)
        
        self._print_status(f"RAM Delta: {memory_used_mb:.2f} MB", "INFO")
        return memory_used_mb / 1024  

    def test_model_load(self, model_path, timeout=30):
        model_name = os.path.basename(model_path)
        self._print_status(f"Testing model load: {model_name}", "PROGRESS")
        
        cmd = f'"{self.llama_simple}" -m "{model_path}" -n 1 "Hello"'
        try:
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
            if proc.returncode == 0:
                self._print_status("Model loaded successfully, generation test passed", "SUCCESS")
                return True
            self._print_status(f"Load failed with exit code {proc.returncode}", "ERROR")
            if proc.stderr:
                print(f"    Error details: {proc.stderr[:200]}...")
            return False
        except subprocess.TimeoutExpired:
            self._print_status(f"Load test timed out after {timeout}s", "ERROR")
            return False


    def benchmark_speed(self, model_path, n_threads=8, n_prompt_tokens=512, n_gen_tokens=128, timeout=90):
        model_name = os.path.basename(model_path)
        self._print_status(f"Benchmarking speed for {model_name}", "PROGRESS")
        
        bench_path = os.path.join(self.llama_dir, "build-cuda", "bin", "Release", "llama-bench.exe")
        if os.name != "nt":
            bench_path = bench_path.replace(".exe", "")
        
        if not os.path.exists(bench_path):
            alt_bench_path = os.path.join(self.llama_dir, "build-cuda", "bin", "Release", "llama-bench.exe")
            if os.name != "nt":
                alt_bench_path = alt_bench_path.replace(".exe", "")
            if os.path.exists(alt_bench_path):
                bench_path = alt_bench_path
            else:
                self._print_status(f"Benchmark tool not found at {bench_path}", "ERROR")
                return {}
    
        cmd = [
            bench_path,
            "-m", model_path,
            "-p", str(n_prompt_tokens),
            "-n", str(n_gen_tokens),
            "-t", str(n_threads)
        ]
        
        monitor = ResourceMonitor()
   
        try:
            monitor.start(0.1)
            tic = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                                timeout=timeout, check=True)
            toc = time.time()
            monitor.stop()
            
            output = proc.stdout
            
            res = {
                "total_inference_time_ms": int((toc-tic)*1000),
                "timestamp": datetime.now().isoformat(),
                "uuid": str(uuid.uuid4()),
                "model_filename": os.path.basename(model_path),
                "tokens_per_second_pp": 0.0,
                "tokens_per_second_pp_std": 0.0,
                "tokens_per_second_gen": 0.0,
                "tokens_per_second_gen_std": 0.0,
            }
            
            self._print_status("Parsing benchmark results", "PROGRESS")
            
            patterns = {
                'pp': [
                    rf'\|\s*pp{n_prompt_tokens}\s*\|\s*([^\|]+)\|',
                    r'\|\s*pp\d+\s*\|\s*([^\|]+)\|',
                    r'pp\s*\|\s*([^\|]+)',
                ],
                'tg': [
                    rf'\|\s*tg{n_gen_tokens}\s*\|\s*([^\|]+)\|',
                    r'\|\s*tg\d+\s*\|\s*([^\|]+)\|',
                    r'tg\s*\|\s*([^\|]+)',
                ]
            }
            # Try multiple patterns for prompt processing
            for pattern in patterns['pp']:
                tps_pp = re.search(pattern, output, re.IGNORECASE)
                if tps_pp:
                    pp_value = tps_pp.group(1).strip()
                    self._print_status(f"Prompt processing speed: {pp_value}", "INFO")
                    try:
                        if "±" in pp_value:
                            mean_val, std_val = pp_value.split("±")
                            res["tokens_per_second_pp"] = float(mean_val.strip())
                            res["tokens_per_second_pp_std"] = float(std_val.strip())
                        else:
                            res["tokens_per_second_pp"] = float(pp_value)
                            res["tokens_per_second_pp_std"] = 0.0
                    except ValueError as e:
                        self._print_status(f"Error parsing PP speed: {e}", "WARNING")
                    break
                    
            for pattern in patterns['tg']:
                tps_tg = re.search(pattern, output, re.IGNORECASE)
                if tps_tg:
                    tg_value = tps_tg.group(1).strip()
                    self._print_status(f"Text generation speed: {tg_value}", "INFO")
                    try:
                        if "±" in tg_value:
                            mean_val, std_val = tg_value.split("±")
                            res["tokens_per_second_gen"] = float(mean_val.strip())
                            res["tokens_per_second_gen_std"] = float(std_val.strip())
                        else:
                            res["tokens_per_second_gen"] = float(tg_value)
                            res["tokens_per_second_gen_std"] = 0.0
                    except ValueError as e:
                        self._print_status(f"Error parsing TG speed: {e}", "WARNING")
                    break
            
            monitor_data = monitor.summary()
            for key, value in monitor_data.items():
                res[key] = value
                
            self._print_status("Speed benchmark completed", "SUCCESS")
            return res
            
        except subprocess.CalledProcessError as e:
            monitor.stop()
            self._print_status(f"Benchmark failed with exit code {e.returncode}", "ERROR")
            return {
                "total_inference_time_ms": 0,
                "timestamp": datetime.now().isoformat(),
                "uuid": str(uuid.uuid4()),
                "model_filename": os.path.basename(model_path),
                "tokens_per_second_pp": 0.0,
                "tokens_per_second_pp_std": 0.0,
                "tokens_per_second_gen": 0.0,
                "tokens_per_second_gen_std": 0.0,
                "peak_memory_usage_mb": 0.0,
                "avg_memory_usage_mb": 0.0,
                "peak_vram_usage_mb": 0.0,
                "avg_gpu_usage_percent": 0.0,
                "peak_gpu_usage_percent": 0.0,
                "avg_cpu_usage_percent": 0.0,
                "power_consumption_watts": 0.0,
            }

        except Exception as e:
            monitor.stop()
            self._print_status(f"Benchmark error: {str(e)}", "ERROR")
            return {}

    def measure_perplexity(self, model_path, ctx=512, ngl=35, timeout=300):
        model_name = os.path.basename(model_path)
        self._print_status(f"Measuring perplexity for {model_name}", "PROGRESS")
        
        cmd = (
            f'"{self.llama_perplexity}" -m "{model_path}" -f "{self.dataset}" -c {ctx} -ngl {ngl}'
        )
        try:
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
            matches = re.findall(r"perplexity\s*=\s*([\d\.]+)", proc.stderr)
            if matches:
                perplexity = float(matches[-1])
                self._print_status(f"Perplexity: {perplexity:.4f}", "SUCCESS")
                return perplexity
            self._print_status("Could not parse perplexity from output", "WARNING")
            if proc.stderr:
                print(f"    Error output: {proc.stderr[:200]}...")
            return float('inf')
        except subprocess.TimeoutExpired:
            self._print_status(f"Perplexity measurement timed out after {timeout}s", "ERROR")
            return float('inf')
        
    def report(self, model_paths, n_threads=8, n_prompt_tokens=512, n_gen_tokens=128, ngl=35, csv_out="quantization_benchmark_results.csv"):
        self._print_header("Quantization Analysis Report")
        
        systeminfo = get_system_profile()
        self._print_status("System Information:", "INFO")
        for key, value in systeminfo.items():
            print(f"    {key}: {value}")
        
        results = []
        HEADER = [
            "timestamp", "uuid", "hostname",
            "Format", "model_filename",
            "cpu", "cpu_count", "ram_total_gb", "gpu", "gpu_vram_gb",
            "Model_Size_GB", "RAM Delta (GB)", "Perplexity", "Compat",
            "tokens_per_second_pp", "tokens_per_second_pp_std",
            "tokens_per_second_gen", "tokens_per_second_gen_std",
            "total_inference_time_ms",
            "peak_memory_usage_mb","avg_memory_usage_mb",
            "peak_vram_usage_mb","avg_gpu_usage_percent","peak_gpu_usage_percent",
            "avg_cpu_usage_percent","power_consumption_watts"
        ]
        
        self._print_status(f"Testing {len(model_paths)} model configurations", "INFO")
        for i, (name, path) in enumerate(model_paths.items(), 1):
            self._print_step(f"Model {i}/{len(model_paths)}: {name}")
            
            if not os.path.exists(path):
                self._print_status(f"File not found: {path}", "ERROR")
                continue
                
            # Test model loading
            loaded = self.test_model_load(path)
            if not loaded:
                self._print_status("Skipping due to load failure", "WARNING")
                continue
                
            # Check compatibility
            compatibility = self.check_compatibility(name)
            self._print_status(f"Hardware compatibility: {compatibility}", "INFO")
            
            # Measure memory usage
            memory_profile_gb = self.measure_memory_usage(path, n_predict=1, ngl=ngl)
            self._print_status(f"Memory usage delta: {memory_profile_gb:.3f}GB", "INFO")
            
            # Run speed benchmark
            bench = self.benchmark_speed(path, n_threads, n_prompt_tokens, n_gen_tokens)
            
            # Measure quality
            quality = self.measure_perplexity(path, ngl=ngl)
            
            # Calculate model size
            mem_gb = os.path.getsize(path)/(1024**3)
            self._print_status(f"Model file size: {mem_gb:.2f}GB", "INFO")
            
            row = {
                "timestamp": datetime.now().isoformat(),
                "uuid": str(uuid.uuid4()),
                "hostname": systeminfo.get("hostname", "unknown"),
                "Format": name,
                "model_filename": os.path.basename(path),
                "cpu": systeminfo.get("cpu", "unknown"),
                "cpu_count": systeminfo.get("cpu_threads", 0),
                "ram_total_gb": systeminfo.get("ram_total_gb", 0),
                "gpu": systeminfo.get("gpu", "unknown"),
                "gpu_vram_gb": systeminfo.get("gpus", [])[0].get("vram_total_gb", 0) if systeminfo.get("gpus", []) else 0,
                "Model_Size_GB": round(mem_gb, 4),
                "RAM Delta (GB)": memory_profile_gb,
                "Perplexity": quality if quality != float('inf') else -1,
                "Compat": compatibility,
                "tokens_per_second_pp": 0.0,
                "tokens_per_second_pp_std": 0.0,
                "tokens_per_second_gen": 0.0,
                "tokens_per_second_gen_std": 0.0,
                "total_inference_time_ms": 0,
                "peak_memory_usage_mb": 0.0,
                "avg_memory_usage_mb": 0.0,
                "peak_vram_usage_mb": 0.0,
                "avg_gpu_usage_percent": 0.0,
                "peak_gpu_usage_percent": 0.0,
                "avg_cpu_usage_percent": 0.0,
                "power_consumption_watts": 0.0,
            }
            

            if isinstance(bench, dict) and bench:
                for key, value in bench.items():
                    if key in row:  
                        row[key] = value
                        
            results.append(row)
            print("\n    Model Analysis Summary:")
            print("    " + "-"*40)
            print(f"    Format: {name}")
            print(f"    Size: {mem_gb:.2f}GB")
            print(f"    Compatibility: {compatibility}")
            print(f"    Perplexity: {quality:.4f}" if quality != float('inf') else "    Perplexity: FAILED")
            if bench and "tokens_per_second_gen" in bench:
                print(f"    Generation Speed: {bench['tokens_per_second_gen']:.1f} tok/s")
            print("    " + "-"*40)
            

        self._print_step("Generating Final Report")
        df = pd.DataFrame(results)
        
        if not os.path.exists(csv_out):
            df.to_csv(csv_out, columns=HEADER, index=False)
            self._print_status(f"Results saved to new file: {csv_out}", "SUCCESS")
        else:
            df.to_csv(csv_out, columns=HEADER, index=False, mode='a', header=False)
            self._print_status(f"Results appended to existing file: {csv_out}", "SUCCESS")
            
        print("\n" + "="*80)
        print(">>> BENCHMARK RESULTS SUMMARY")
        print("="*80)
        
        if not df.empty:
            summary_cols = ["Format", "Model_Size_GB", "Perplexity", "tokens_per_second_gen", "peak_vram_usage_mb", "Compat"]
            available_cols = [col for col in summary_cols if col in df.columns]
            
            if available_cols:
                print(df[available_cols].to_string(index=False))
            else:
                print(df.to_string(index=False))
        else:
            self._print_status("No successful benchmarks to display", "WARNING")
            
        print("="*80)
        
        return df

def get_models_from_dir(model_dir, ext='.gguf'):
    """
    Scans a directory and returns {quant_name: filepath} for all GGUF files.
    Enhanced to handle various naming conventions.
    """
    print(f"\n[*] Scanning directory: {model_dir}")
    print(f"[*] Looking for files with extension: {ext}")
    
    if not os.path.exists(model_dir):
        print(f"[-] Directory not found: {model_dir}")
        return {}
        
    models = {}
    files_found = 0
    
    for file in os.listdir(model_dir):
        if file.lower().endswith(ext.lower()):
            files_found += 1
            base = os.path.basename(file)
            
 
            name_without_ext = base.replace(ext, '').replace(ext.upper(), '')
            
            patterns = [
                r'.*-([qf]\d+[_k]*[_msl]*|fp\d+|f\d+|iq\d+[_k]*[_msl]*)$',  # Standard pattern
                r'.*\.([qf]\d+[_k]*[_msl]*|fp\d+|f\d+|iq\d+[_k]*[_msl]*)$',  # Dot separator
                r'.*_([qf]\d+[_k]*[_msl]*|fp\d+|f\d+|iq\d+[_k]*[_msl]*)$',   # Underscore separator
            ]
            
            quant = None
            for pattern in patterns:
                match = re.search(pattern, name_without_ext, re.IGNORECASE)
                if match:
                    quant = match.group(1).lower()
                    break
    
            if not quant:
                separators = ['-', '_', '.']
                for sep in separators:
                    if sep in name_without_ext:
                        quant = name_without_ext.rsplit(sep, 1)[-1].lower()
                        break
                
                if not quant:
                    quant = name_without_ext.lower()
            
            models[quant] = os.path.join(model_dir, file)
            print(f"[+] Found model: {quant} -> {file}")
    
    print(f"[*] Total models found: {files_found}")
    
    if not models:
        print(f"[!] No {ext} files found in {model_dir}")
    
    return models