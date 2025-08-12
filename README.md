# QBench - LLM Quantization & Hardware Optimization

A CLI tool for automated quantization, benchmarking, and hardware-specific optimization of Large Language Models. QBench eliminates guesswork by profiling your system, testing multiple quantization schemes, and providing data-driven recommendations.

## Overview

QBench automates the model optimization workflow:
- Hardware profiling and compatibility analysis
- Automated quantization to multiple formats
- Performance benchmarking (speed, memory, quality)
- Hardware-aware configuration recommendations
- Integration with custom optimization kernels

## Installation

```bash
git clone https://github.com/AnkitTsj/qbench.git
cd qbench
pip install -r requirements.txt
```

## Usage

### 1. System Analysis
```bash
python preprocess.py
```
Profiles your hardware configuration and generates compatibility reports.

![Hardware Profile](https://github.com/user-attachments/assets/1823c3c7-e719-4493-9f8c-1c50976412f4)

### 2. Model Optimization
the model files should be from huggingface - https://huggingface.co/models
```bash
python aqeng.py \
    --workspace-dir "/media/user/fast_ssd/llm_projects" \
    --repo-id "NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF" \
    --filename "Hermes-2-Pro-Llama-3-8B-F16.gguf" \
    --local-filename "hermes-l3-8b-f16.gguf" \
    --gpu-layers 45 \
    --target-formats "q4_k_m" "q6_k" "q8_0" \
    --use-case "performance" \
    --kernels-repo "https://my.custom.repo/kernels/main" \
    --force-rebuild
```

### 3. Results Analysis
The tool generates comprehensive benchmark reports with performance metrics, memory usage, and quality scores.

![Benchmark Results](https://github.com/user-attachments/assets/701d0997-6297-4252-bab4-3f0b2eccbc69)

### 4. Deployment
Ready-to-use commands for optimal configurations:

![Run Command](https://github.com/user-attachments/assets/72b430be-6118-4374-b47a-837af64a5f0e)

## Command Reference

### quantize.py

| Parameter | Type | Description |
|-----------|------|-------------|
| `--workspace-dir` | string | Working directory for models and outputs |
| `--repo-id` | string | HuggingFace repository identifier |
| `--filename` | string | Source model filename |
| `--local-filename` | string | Local filename for downloaded model |
| `--gpu-layers` | int | Number of layers to offload to GPU |
| `--target-formats` | list | Quantization formats to benchmark |
| `--use-case` | string | Optimization target: performance/quality/memory/balanced |
| `--kernels-repo` | string | Custom kernel repository URL (optional) |
| `--force-rebuild` | flag | Force rebuild existing quantizations |

## Quantization Formats

| Format | Bits | Description | Use Case |
|--------|------|-------------|----------|
| `q4_k_m` | 4-bit | K-means quantization, mixed precision | Balanced performance/quality |
| `q5_k_m` | 5-bit | Higher precision K-means | Quality-focused deployments |
| `q6_k` | 6-bit | High quality quantization | Quality-critical applications |
| `q8_0` | 8-bit | Minimal quality loss | High-quality baseline |

## Optimization Strategies

### Performance Mode
```bash
--use-case "performance"
```
Maximizes inference speed. Weights: 60% speed, 25% quality, 15% memory efficiency.

### Quality Mode
```bash
--use-case "quality"
```
Prioritizes model accuracy. Weights: 20% speed, 60% quality, 20% memory efficiency.

### Memory Mode
```bash
--use-case "memory"
```
Minimizes memory footprint. Weights: 30% speed, 30% quality, 40% memory efficiency.

### Balanced Mode
```bash
--use-case "balanced"
```
Equal weighting across all metrics. Weights: 40% speed, 40% quality, 20% memory efficiency.

## Benchmarking Metrics

QBench measures and reports:

**Performance Metrics:**
- Prompt processing speed (tokens/sec)
- Text generation speed (tokens/sec)
- Model loading time
- First token latency

**Resource Utilization:**
- Peak memory usage (RAM/VRAM)
- Average GPU utilization
- Power consumption (when available)

**Quality Assessment:**
- Perplexity scores on standard datasets
- Quality retention vs baseline

**System Compatibility:**
- Hardware compatibility scoring
- Memory requirement validation
- Performance prediction confidence

## Architecture

### Core Components

- **HardwareProfiler**: System analysis and capability detection
- **QuantizationAnalyzer**: Model quantization and benchmarking engine  
- **ResourceMonitor**: Real-time performance monitoring
- **OptimizationEngine**: Multi-objective configuration scoring

### Integration Points

- llama.cpp for quantization and inference
- HuggingFace Hub for model management
- Custom kernel repositories for hardware-specific optimizations

## Advanced Configuration

### Custom Kernels
```bash
--kernels-repo "https://github.com/username/optimized-kernels"
```
Integrates hardware-specific optimization kernels (CUDA, Nim, etc.)

### Batch Processing
```bash
# Process multiple models from file
python quantize.py --batch-file models.txt
```

### Continuous Optimization
```bash
# Monitor and re-optimize based on usage patterns  
python quantize.py --continuous --monitor-interval 24h
```

## Output Formats

Results are saved in multiple formats:
- CSV: Machine-readable benchmark data
- JSON: Structured configuration recommendations  
- Markdown: Human-readable analysis reports

## Requirements

**System:**
- Python 3.8+
- CUDA toolkit (for GPU acceleration)
- 8GB+ RAM recommended

**Dependencies:**
- psutil (system monitoring)
- pandas (data analysis)
- huggingface-hub (model downloads)
- pynvml (NVIDIA GPU monitoring)

## Contributing

Contributions are welcome. Please ensure:
- Code follows existing patterns
- New features include appropriate tests
- Documentation is updated for API changes
- Performance impact is measured and documented

## Issues

Report bugs or feature requests via GitHub Issues. Include:
- Hardware specifications
- Command executed
- Error output or unexpected behavior
- System environment details
