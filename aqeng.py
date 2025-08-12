import os
import re
from datasets import load_dataset
from postp import pmain
import os
import shutil
import subprocess
import argparse
import sys
import os
import sys
from datetime import datetime
from pathlib import Path
import requests
import pandas as pd
from huggingface_hub import hf_hub_download
import os
import shutil
import subprocess
import requests
from qa import QuantizationAnalyzer
from huggingface_hub import hf_hub_download
import pandas as pd

class AutoQuantizer(QuantizationAnalyzer):
    """
    LLM quantization pipeline:
      - Clones/builds llama.cpp (CPU+CUDA)
      - Downloads pre-built kernels and pre-patched main.cpp
      - Downloads models from HuggingFace
      - Quantizes, benchmarks, recommends best config
    """
    def __init__(
    self,
    workspace_dir: str,
    llama_simple_path: str = None,
    llama_perplexity_path: str = None,
    dataset_path: str = None,
    gpu_layers_default: int = 36,
    kernels_repo_base: str = "https://raw.githubusercontent.com/AnkitTsj/qbench_kernels/main",
    run_main_dir: str = "https://raw.githubusercontent.com/AnkitTsj/qbench_kernels/main/run_main",
    *args, **kwargs
    ):
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.llama_dir = os.path.join(self.workspace_dir, "llama.cpp")
        self.kernels_dir = os.path.join(self.workspace_dir, "kernels")
        self.models_dir = os.path.join(self.workspace_dir, "models")
        
     
        exe_suffix = ".exe" if os.name == "nt" else ""
        
        if llama_simple_path is None:
            llama_simple_path = os.path.join(self.llama_dir, "build-cuda", "bin", "Release", f"llama-simple{exe_suffix}")
       
            if not os.path.exists(llama_simple_path):
                llama_simple_path = os.path.join(self.llama_dir, "build-cpu", "bin", "Release", f"llama-cli{exe_suffix}")
        
        if llama_perplexity_path is None:
            llama_perplexity_path = os.path.join(self.llama_dir, "build-cuda", "bin", "Release", f"llama-perplexity{exe_suffix}")

            if not os.path.exists(llama_perplexity_path):
                llama_perplexity_path = os.path.join(self.llama_dir, "build-cpu", "bin", "Release", f"llama-perplexity{exe_suffix}")
        
        if dataset_path is None:
            dataset_path = os.path.join(self.workspace_dir, "datasets", "wiki.test.short.txt")
        
        super().__init__(llama_simple_path, llama_perplexity_path, dataset_path, *args, **kwargs)
        
        self.gpu_layers_default = gpu_layers_default
        self.kernels_repo_base = kernels_repo_base
        self.run_main_dir = run_main_dir

        self.quantize_exe = None
        self.llama_cli_cpu = None
        self.llama_cli_cuda = None

        for d in [self.workspace_dir, self.models_dir, self.kernels_dir]:
            os.makedirs(d, exist_ok=True)

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

    def setup_complete_environment(self, force_rebuild: bool = False):
        self._print_header("Environment Setup")
        self._print_status(f"Workspace Directory: {self.workspace_dir}", "INFO")

        if not os.path.exists(self.llama_dir) or force_rebuild:
            self._clone_llama_cpp()

        self._download_patched_main_cpp()
        self._build_quantization_tools()
        self._build_cpu_engine()
        self._build_llama_perplexity_cpu()
        self._build_llama_simple_cpu()
        self._build_bench_cpu()
        # self._build_pplx_and_simple_cpu()
        if self._has_cuda_support():
            self._build_cuda_engine()
            self._build_llama_perplexity_cuda()
            self._build_llama_simple_cuda()
            self._build_bench_cuda()


        self._download_optimization_kernels()
        self._copy_kernels_to_builds()
        # self._build_benchmarkexe()
        self._print_status("Environment setup complete!", "SUCCESS")

    def _clone_llama_cpp(self):
        self._print_step("Cloning llama.cpp Repository")
        if os.path.exists(self.llama_dir):
            self._print_status("Removing existing llama.cpp directory", "WARNING")
            shutil.rmtree(self.llama_dir)
        
        self._print_status("Cloning llama.cpp from GitHub", "PROGRESS")
        r = subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp", self.llama_dir],
                           capture_output=True, text=True, cwd=self.workspace_dir)
        if r.returncode != 0:
            self._print_status(f"Clone failed: {r.stderr}", "ERROR")
            raise RuntimeError(f"llama.cpp clone failed: {r.stderr}")
        self._print_status("Repository cloned successfully", "SUCCESS")

    def _download_patched_main_cpp(self):
        self._print_step("Downloading Pre-patched main.cpp")
        main_cpp_local = os.path.join(self.llama_dir, "examples", "main", "main.cpp")
        os.makedirs(os.path.dirname(main_cpp_local), exist_ok=True)
        
        url = f"{self.run_main_dir}/main.cpp"
        self._print_status(f"Downloading from: {url}", "PROGRESS")
        
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(main_cpp_local, "w", encoding='utf-8') as f:
            f.write(r.text.strip())
        self._print_status("main.cpp replaced with optimized version", "SUCCESS")

    def _build_quantization_tools(self):
        self._print_step("Building Quantization Tools")
        
        self._print_status("Configuring CMake for quantization tools", "PROGRESS")
        configure_cmd = [
            "cmake", "-S", ".", "-B", "build",
            "-DLLAMA_BUILD_EXAMPLES=ON", "-DLLAMA_CURL=OFF", "-DCMAKE_BUILD_TYPE=Release"
        ]
        r = subprocess.run(configure_cmd, capture_output=True, text=True, cwd=self.llama_dir)
        if r.returncode != 0:
            self._print_status(f"CMake configure failed: {r.stderr}", "ERROR")
            raise RuntimeError(f"CMake configure failed: {r.stderr}")

        self._print_status("Building llama-quantize executable", "PROGRESS")
        build_cmd = ["cmake", "--build", "build", "--config", "Release", "--target", "llama-quantize"]
        r = subprocess.run(build_cmd, capture_output=True, text=True, cwd=self.llama_dir)
        if r.returncode != 0:
            self._print_status(f"Build failed: {r.stderr}", "ERROR")
            raise RuntimeError(f"Build failed: {r.stderr}")
            
        exe_suffix = ".exe" if os.name == "nt" else ""
        self.quantize_exe = os.path.join(self.llama_dir, "build", "bin", "Release", f"llama-quantize{exe_suffix}")
        self._print_status("Quantization tools built successfully", "SUCCESS")

    def _build_cpu_engine(self):
        self._print_step("Building CPU Inference Engine")
        
        self._print_status("Configuring CMake for CPU build", "PROGRESS")
        configure_cmd = [
            "cmake", "-S", ".", "-B", "build-cpu",
            "-DLLAMA_CURL=OFF", "-DCMAKE_BUILD_TYPE=Release"
        ]
        r = subprocess.run(configure_cmd, capture_output=True, text=True, cwd=self.llama_dir)
        if r.returncode != 0:
            self._print_status(f"CPU CMake configure failed: {r.stderr}", "ERROR")
            raise RuntimeError(f"CPU CMake configure failed: {r.stderr}")

        self._print_status("Building llama-cli for CPU", "PROGRESS")
        build_cmd = ["cmake", "--build", "build-cpu", "--config", "Release", "--target", "llama-cli"]
        r = subprocess.run(build_cmd, capture_output=True, text=True, cwd=self.llama_dir)
        if r.returncode != 0:
            self._print_status(f"CPU build failed: {r.stderr}", "ERROR")
            raise RuntimeError(f"CPU build failed: {r.stderr}")
            
        exe_suffix = ".exe" if os.name == "nt" else ""
        self.llama_cli_cpu = os.path.join(self.llama_dir, "build-cpu", "bin", "Release", f"llama-cli{exe_suffix}")
        self._print_status("CPU engine built successfully", "SUCCESS")

    def _build_cuda_engine(self):
        self._print_step("Building CUDA Inference Engine")
        
        self._print_status("Configuring CMake for CUDA build", "PROGRESS")
        if os.name == "nt":
            configure_cmd = [
                "cmake", "-S", ".", "-B", "build-cuda",
                "-G", "Visual Studio 17 2022", "-A", "x64",
                "-DGGML_CUDA=ON", "-DLLAMA_CURL=OFF", "-DCMAKE_BUILD_TYPE=Release"
            ]
        else:
            configure_cmd = [
                "cmake", "-S", ".", "-B", "build-cuda",
                "-DGGML_CUDA=ON", "-DLLAMA_CURL=OFF", "-DCMAKE_BUILD_TYPE=Release"
            ]
        r = subprocess.run(configure_cmd, capture_output=True, text=True, cwd=self.llama_dir)
        if r.returncode != 0:
            self._print_status(f"CUDA CMake configure failed: {r.stderr}", "ERROR")
            raise RuntimeError(f"CUDA CMake configure failed: {r.stderr}")
            
        self._print_status("Building llama-cli with CUDA support", "PROGRESS")
        build_cmd = ["cmake", "--build", "build-cuda", "--config", "Release", "--target", "llama-cli"]
        r = subprocess.run(build_cmd, capture_output=True, text=True, cwd=self.llama_dir)
        if r.returncode != 0:
            self._print_status(f"CUDA build failed: {r.stderr}", "ERROR")
            raise RuntimeError(f"CUDA build failed: {r.stderr}")
            
        exe_suffix = ".exe" if os.name == "nt" else ""
        self.llama_cli_cuda = os.path.join(self.llama_dir, "build-cuda", "bin", "Release", f"llama-cli{exe_suffix}")
        self._print_status("CUDA engine built successfully", "SUCCESS")

    def _build_llama_simple_cuda(self):
        """Builds the llama-simple executable with CUDA support."""
        self._print_status("Building llama-simple with CUDA support", "PROGRESS")
        build_cmd = ["cmake", "--build", "build-cuda", "--config", "Release", "--target", "llama-simple"]
        r = subprocess.run(build_cmd, capture_output=True, text=True, cwd=self.llama_dir)
        
        if r.returncode != 0:
            self._print_status(f"llama-simple build failed: {r.stderr}", "ERROR")
            raise RuntimeError(f"llama-simple failed: {r.stderr}")
        
        exe_suffix = ".exe" if sys.platform == "win32" else ""
        self.llama_simple_path = os.path.join(self.llama_dir, "build-cuda", "bin", "Release", f"llama-simple{exe_suffix}")
        self._print_status("llama-simple engine CUDA built successfully", "SUCCESS")
    

    def _build_llama_simple_cpu(self):
        """Builds the llama-simple executable with CPU support."""
        self._print_status("Building llama-simple with CPU support", "PROGRESS")
        build_cmd = ["cmake", "--build", "build-cpu", "--config", "Release", "--target", "llama-simple"]
        r = subprocess.run(build_cmd, capture_output=True, text=True, cwd=self.llama_dir)
        
        if r.returncode != 0:
            self._print_status(f"llama-simple build failed: {r.stderr}", "ERROR")
            raise RuntimeError(f"llama-simple failed: {r.stderr}")
        
        exe_suffix = ".exe" if sys.platform == "win32" else ""
        self.llama_simple_path = os.path.join(self.llama_dir, "build-cpu", "bin", "Release", f"llama-simple{exe_suffix}")
        self._print_status("llama-simple engine CPU built successfully", "SUCCESS")
    
    def _build_llama_perplexity_cpu(self):
        """Builds the llama-perplexity executable with CPU support."""
        self._print_status("Building llama-perplexity with CPU support", "PROGRESS")
        build_cmd = ["cmake", "--build", "build-cpu", "--config", "Release", "--target", "llama-perplexity"]
        r = subprocess.run(build_cmd, capture_output=True, text=True, cwd=self.llama_dir)
        
        if r.returncode != 0:
            self._print_status(f"llama-perplexity build failed: {r.stderr}", "ERROR")
            raise RuntimeError(f"llama-perplexity build failed: {r.stderr}")

        exe_suffix = ".exe" if sys.platform == "win32" else ""
        self.llama_perplexity_path = os.path.join(self.llama_dir, "build-cpu", "bin", "Release", f"llama-perplexity{exe_suffix}")
        self._print_status("llama-perplexity engine CPU built successfully", "SUCCESS")

    def _build_llama_perplexity_cuda(self):
        """Builds the llama-perplexity executable with CUDA support."""
        self._print_status("Building llama-perplexity with CUDA support", "PROGRESS")
        build_cmd = ["cmake", "--build", "build-cuda", "--config", "Release", "--target", "llama-perplexity"]
        r = subprocess.run(build_cmd, capture_output=True, text=True, cwd=self.llama_dir)
        
        if r.returncode != 0:
            self._print_status(f"llama-perplexity build failed: {r.stderr}", "ERROR")
            raise RuntimeError(f"llama-perplexity build failed: {r.stderr}")

        exe_suffix = ".exe" if sys.platform == "win32" else ""
        self.llama_perplexity_path = os.path.join(self.llama_dir, "build-cuda", "bin", "Release", f"llama-perplexity{exe_suffix}")
        self._print_status("llama-perplexity engine CUDA built successfully", "SUCCESS")

    def _build_bench_cuda(self):
        self._print_status("Building llama-bench with CUDA support", "PROGRESS")
        build_cmd = ["cmake", "--build", "build-cuda", "--config", "Release", "--target", "llama-bench"]
        r = subprocess.run(build_cmd, capture_output=True, text=True, cwd=self.llama_dir)
        if r.returncode != 0:
            self._print_status(f"llama-bench build failed: {r.stderr}", "ERROR")
            raise RuntimeError(f"llama-bench build failed: {r.stderr}")
            
        exe_suffix = ".exe" if os.name == "nt" else ""
        self.llama_bench_cuda = os.path.join(self.llama_dir, "build-cuda", "bin", "Release", f"llama-bench{exe_suffix}")
        self._print_status("llama-bench engine with CUDA built successfully", "SUCCESS")

    def _build_bench_cpu(self):
        self._print_status("Building llama-bench with CPU support", "PROGRESS")
        build_cmd = ["cmake", "--build", "build-cpu", "--config", "Release", "--target", "llama-bench"]
        r = subprocess.run(build_cmd, capture_output=True, text=True, cwd=self.llama_dir)
        if r.returncode != 0:
            self._print_status(f"llama-bench build failed: {r.stderr}", "ERROR")
            raise RuntimeError(f"llama-bench build failed: {r.stderr}")
            
        exe_suffix = ".exe" if os.name == "nt" else ""
        self.llama_bench_cuda = os.path.join(self.llama_dir, "build-cpu", "bin", "Release", f"llama-bench{exe_suffix}")
        self._print_status("llama-bench engine with CPU built successfully", "SUCCESS")



    def _has_cuda_support(self) -> bool:
            """Enhanced CUDA detection with specific version info"""
            try:
                result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
                if result.returncode == 0:
                    # Extract CUDA version
                    version_match = re.search(r'release (\d+\.\d+)', result.stdout)
                    if version_match:
                        cuda_version = version_match.group(1)
                        self._print_status(f"CUDA {cuda_version} detected", "SUCCESS")
                    else:
                        self._print_status("CUDA detected (version unknown)", "SUCCESS")
                    return True
                else:
                    self._print_status("NVCC not found in PATH", "WARNING")
                    return False
            except FileNotFoundError:
                self._print_status("CUDA toolkit not installed", "INFO")
                return False

    def _download_optimization_kernels(self):
        self._print_step("Downloading Optimization Kernels")
        
        kernel_types = ["fp16", "q4", "q8"]
        dll_suffix = ".dll" if os.name == "nt" else ".so"
        
        for kernel_type in kernel_types:
            fname = f"libsmolkernels_{kernel_type}{dll_suffix}"
            dest = os.path.join(self.kernels_dir, fname)
            
            if os.path.exists(dest):
                self._print_status(f"{kernel_type.upper()} kernel already exists", "INFO")
                continue
                
            url = f"{self.kernels_repo_base}/{kernel_type}_kernels/{fname}"
            self._print_status(f"Downloading {kernel_type.upper()} kernel", "PROGRESS")
            
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                with open(dest, "wb") as f:
                    f.write(r.content)
                self._print_status(f"{kernel_type.upper()} kernel downloaded", "SUCCESS")
            except Exception as e:
                self._print_status(f"Failed to download {kernel_type} kernel: {e}", "ERROR")

    def _copy_kernels_to_builds(self):
        self._print_step("Installing Kernels to Build Directories")
        
        dll_suffix = ".dll" if os.name == "nt" else ".so"
        kernel_files = [f"libsmolkernels_fp16{dll_suffix}", f"libsmolkernels_q4{dll_suffix}", f"libsmolkernels_q8{dll_suffix}"]
        
        for build_type in ["build-cpu", "build-cuda"]:
            target_dir = os.path.join(self.llama_dir, build_type, "bin", "Release")
            os.makedirs(target_dir, exist_ok=True)
            
            self._print_status(f"Installing kernels to {build_type}", "PROGRESS")
            for kfile in kernel_files:
                src = os.path.join(self.kernels_dir, kfile)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(target_dir, kfile))
            self._print_status(f"Kernels installed to {build_type}", "SUCCESS")

    def download_model_hf(self, repo_id, filename, local_filename=None):
        self._print_header("Model Download")
        
        target_filename = local_filename if local_filename else filename
        target_path = os.path.join(self.models_dir, target_filename)

        if os.path.exists(target_path):
            self._print_status(f"Model '{target_filename}' already exists", "INFO")
            self._print_status(f"Location: {target_path}", "INFO")
            return target_path
        
        self._print_status(f"Repository: {repo_id}", "INFO")
        self._print_status(f"Filename: {filename}", "INFO")
        self._print_status("Starting download...", "PROGRESS")
        
        try:
            path = hf_hub_download(
                repo_id=repo_id, 
                filename=filename, 
                local_dir=self.models_dir, 
                local_dir_use_symlinks=False
            )
            
            if local_filename and path != target_path:
                shutil.move(path, target_path)
                path = target_path
                
            self._print_status("Download completed successfully", "SUCCESS")
            self._print_status(f"Saved to: {path}", "INFO")
            return path
            
        except Exception as e:
            self._print_status(f"Download failed: {e}", "ERROR")
            raise

    def auto_quantize_model(self, fp16_path, target_formats=None, overwrite=False):
        self._print_header("Model Quantization")
        
        if not self.quantize_exe or not os.path.exists(self.quantize_exe):
            self._print_status("Quantizer not built", "ERROR")
            raise RuntimeError("Quantizer not built.")
            
        if target_formats is None:
            target_formats = ["q8_0", "q4_K_M"]
            
        self._print_status(f"Source model: {fp16_path}", "INFO")
        self._print_status(f"Target formats: {', '.join(target_formats)}", "INFO")
        
        base = fp16_path.replace(".gguf", "")
        results = {}
        
        for fmt in target_formats:
            out_file = f"{base}-{fmt}.gguf"
            
            if os.path.exists(out_file) and not overwrite:
                self._print_status(f"{fmt.upper()} quantization already exists", "INFO")
                results[fmt] = out_file
                continue
                
            self._print_status(f"Quantizing to {fmt.upper()}", "PROGRESS")
            cmd = [self.quantize_exe, fp16_path, out_file, fmt]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            
            if proc.returncode == 0:
                results[fmt] = out_file
                self._print_status(f"{fmt.upper()} quantization completed", "SUCCESS")
                self._print_status(f"Output: {out_file}", "INFO")
            else:
                self._print_status(f"{fmt.upper()} quantization failed: {proc.stderr}", "ERROR")
                
        return results

    def optimize_for_hardware(
        self,
        model_paths: dict,
        use_case: str = "balanced",
        cpu_only: bool = False,
    ) -> dict:
        """
        Benchmark quantized models and recommend optimal configuration
        
        Based on performance results from blog post Table 4-1
        """
        self._print_header("Hardware Optimization Analysis")
        
        if not self.llama_cli_cpu or not os.path.exists(self.llama_cli_cpu):
            self._print_status("Inference engines not built", "ERROR")
            raise RuntimeError("Inference engines not built. Run setup_complete_environment() first.")
        
        self._print_status(f"Use case: {use_case}", "INFO")
        self._print_status(f"Models to benchmark: {len(model_paths)}", "INFO")
        self._print_status("Starting comprehensive benchmarks", "PROGRESS")
        
        benchmark_df = self.report(model_paths, csv_out=f"{self.workspace_dir}/quantization_benchmark_results.csv")
     
        recommendations = self._analyze_tradeoffs(benchmark_df, use_case)
        
        self._print_status("Benchmark analysis completed", "SUCCESS")
        self._print_status(f"Recommended format: {recommendations['recommended']['format']}", "SUCCESS")
        self._print_status(f"Reasoning: {recommendations['recommended']['reasoning']}", "INFO")
        
        return recommendations

    def _print_build_summary(self):
        """Print summary of built components"""
        self._print_header("Build Summary")
        
        self._print_status(f"Workspace: {self.workspace_dir}", "INFO")
        
        components = [
            ("Quantize Tool", self.quantize_exe),
            ("CPU Engine", self.llama_cli_cpu),
            ("CUDA Engine", self.llama_cli_cuda)
        ]
        
        for name, path in components:
            if path and os.path.exists(path):
                self._print_status(f"{name}: READY", "SUCCESS")
            else:
                self._print_status(f"{name}: NOT AVAILABLE", "WARNING")
        
        dll_suffix = ".dll" if os.name == "nt" else ".so"
        kernel_types = ["fp16", "q4", "q8"]
        
        print("\nKernel Status:")
        print("-" * 30)
        for kernel_type in kernel_types:
            kernel_path = os.path.join(self.kernels_dir, f"libsmolkernels_{kernel_type}{dll_suffix}")
            if os.path.exists(kernel_path):
                self._print_status(f"{kernel_type.upper()} Kernel: READY", "SUCCESS")
            else:
                self._print_status(f"{kernel_type.upper()} Kernel: NOT FOUND", "WARNING")

    def _analyze_tradeoffs(self, benchmark_df: pd.DataFrame, use_case: str) -> dict:
        """Analyze benchmark results and recommend optimal configuration"""
        self._print_step("Analyzing Performance Tradeoffs")
        
        configs = []
        
        for _, row in benchmark_df.iterrows():
            score = self._calculate_optimization_score(row, use_case)
            configs.append({
                "format": row["Format"],
                "model_file": row["model_filename"],
                "score": score,
                "speed_tokens_per_sec": row.get("tokens_per_second_gen", 0),
                "memory_usage_gb": row.get("peak_vram_usage_mb", 0) / 1024,
                "quality_score": 1.0 / max(row.get("Perplexity", float('inf')), 1e-6),
                "reasoning": self._explain_recommendation(row, use_case)
            })
        
        configs.sort(key=lambda x: x["score"], reverse=True)
        
        self._print_status(f"Analyzed {len(configs)} configurations", "INFO")
        
        return {
            "recommended": configs[0],
            "alternatives": configs[1:3],
            "analysis": self._generate_analysis_summary(configs)
        }

    def _calculate_optimization_score(self, row: pd.Series, use_case: str) -> float:
        """Calculate multi-objective optimization score based on blog post benchmarks"""
        speed = row.get("tokens_per_second_gen", 0)
        memory = row.get("peak_vram_usage_mb", float('inf'))
        quality = 1.0 / max(row.get("Perplexity", float('inf')), 1.0)
        
        # Weights based on use case 
        weights = {
            "speed_priority": (0.6, 0.2, 0.2),      
            "quality_priority": (0.2, 0.6, 0.2),    
            "memory_constrained": (0.3, 0.3, 0.4),  
            "balanced": (0.4, 0.4, 0.2)             
        }
        
        w_speed, w_quality, w_memory = weights.get(use_case, weights["balanced"])
        
        memory_score = 1.0 / max(float(memory), 1.0)
        
        return w_speed * speed + w_quality * quality + w_memory * memory_score

    def _explain_recommendation(self, row: pd.Series, use_case: str) -> str:
        """Generate human-readable explanation for recommendation"""
        fmt = row["Format"]
        speed = row.get("tokens_per_second_gen", 0)
        memory = row.get("peak_vram_usage_mb", 0) / 1024
        
        explanations = {
            "speed_priority": f"{fmt} delivers {speed:.1f} tok/s generation speed, optimal for real-time applications",
            "quality_priority": f"{fmt} maintains excellent model quality while using {memory:.1f}GB VRAM",
            "memory_constrained": f"{fmt} fits efficiently in {memory:.1f}GB memory budget with good performance",
            "balanced": f"{fmt} offers optimal balance of {speed:.1f} tok/s speed and {memory:.1f}GB memory usage"
        }
        
        return explanations.get(use_case, f"{fmt} recommended for your configuration")

    def _generate_analysis_summary(self, configs: list) -> str:
        """Generate comprehensive analysis summary"""
        if not configs:
            return "No valid configurations found"
        
        top = configs[0]
        alternatives = [c["format"] for c in configs[1:3] if c]
        
        summary = f"""
            Performance Analysis Results:
            {"-" * 50}
            PRIMARY RECOMMENDATION: {top['format']}
            Speed: {top['speed_tokens_per_sec']:.1f} tokens/sec
            Memory: {top['memory_usage_gb']:.1f}GB
            Reasoning: {top['reasoning']}

            ALTERNATIVE OPTIONS: {', '.join(alternatives) if alternatives else 'None'}

            ANALYSIS SUMMARY: Based on {len(configs)} tested configurations, 
            the analysis considered generation speed, memory efficiency, and model quality 
            to determine the optimal setup for your hardware and use case.
                    """.strip()
        
        return summary

def download_data(output_filename):
    """
    Downloads a subset of the wikitext dataset and saves it to a file,
    ensuring the directory structure exists beforehand.
    """
    print("Loading the wikitext dataset from Hugging Face...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    test_data = dataset['test']

    output_dir = os.path.dirname(output_filename)

    if output_dir:
        
        os.makedirs(output_dir, exist_ok=True)

    print(f"Writing test data to '{output_filename}'...")
    with open(output_filename, "w", encoding="utf-8") as f:
        non_empty_lines = [line for line in test_data['text'][:100] if line.strip()]
        f.write("\n".join(non_empty_lines))

    print(f"\nSuccessfully created '{os.path.abspath(output_filename)}'")



def main(args):
    """Main execution function."""
 
    workspace = args.workspace_dir
    llama_simple_path = os.path.join(workspace, "llama.cpp", "build-cuda", "bin", "Release", "llama-simple.exe")
    llama_pplx_path = os.path.join(workspace, "llama.cpp", "build-cuda", "bin", "Release", "llama-perplexity.exe")
    data_path = os.path.join(workspace, "data", "wiki.test.short.txt")

    print("="*80)
    print(">>> LLAMA.CPP QUANTIZATION PIPELINE")
    print("="*80)

    aq = AutoQuantizer(
        workspace_dir=workspace,
        llama_simple_path=llama_simple_path,
        llama_perplexity_path=llama_pplx_path,
        dataset_path=data_path,
        gpu_layers_default=args.gpu_layers,
        kernels_repo_base=args.kernels_repo
    )

    try:
        aq.setup_complete_environment(force_rebuild=args.force_rebuild)

        fp16_model = aq.download_model_hf(
            repo_id=args.repo_id,
            filename=args.filename,
            local_filename=args.local_filename
        )

        quantized_models = aq.auto_quantize_model(
            fp16_model,
            target_formats=args.target_formats
        )

        recommendations = aq.optimize_for_hardware(
            quantized_models,
            use_case=args.use_case
        )

        print("\n" + "="*80)
        print(">>> PIPELINE COMPLETION SUMMARY")
        print("="*80)
        print(f"[+] Recommended Model: {recommendations['recommended']['model_file']}")
        print(f"[+] Format: {recommendations['recommended']['format']}")
        print(f"[*] Performance: {recommendations['recommended']['speed_tokens_per_sec']:.1f} tok/s")
        print(f"[*] Memory Usage: {recommendations['recommended']['memory_usage_gb']:.1f}GB")
        print("\nAnalysis Details:")
        print(recommendations['analysis'])
        print("="*80)

    except Exception as e:
        print(f"[-] Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated GGUF Quantization Pipeline.")


    parser.add_argument("--workspace-dir", type=str, default="C:/llm_quantization_workspace", help="Root directory for the quantization workspace.")

    parser.add_argument("--repo-id", type=str, default="unsloth/Qwen3-4B-Instruct-2507-GGUF", help="Hugging Face repository ID for the model.")
    parser.add_argument("--filename", type=str, default="Qwen3-4B-Instruct-2507-F16.gguf", help="The F16 model filename in the repository.")
    parser.add_argument("--local-filename", type=str, default="qwen3-4b-f16.gguf", help="The local filename to save the downloaded model as.")

    parser.add_argument("--gpu-layers", type=int, default=36, help="Default number of GPU layers to offload.")
    parser.add_argument("--target-formats", nargs='+', default=["q4_k_m"], help="List of quantization formats to target (e.g., q4_k_m q5_k_s).")
    parser.add_argument("--use-case", type=str, default="balanced", choices=["balanced", "performance", "memory"], help="Optimization preference.")
    
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild of the environment.")
    parser.add_argument("--kernels-repo", type=str, default="https://raw.githubusercontent.com/AnkitTsj/qbench_kernels/main", help="Base URL for qbench kernels.")

    args = parser.parse_args()
    main(args)
    workspace = args.workspace_dir
    local_filename = args.local_filename
    pmain(f"{workspace}/quantization_benchmark_results.csv", f"{workspace}/llama.cpp", f"{workspace}/model/{local_filename}")


