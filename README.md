# QBench – Automated LLM Quantization & Hardware Optimization

Stop wasting hours guessing which quantization format runs best on your machine.
QBench profiles your hardware, tests multiple formats, benchmarks speed/memory/quality, and tells you exactly which configuration gives you the best results for your needs.

---

## Why QBench

Optimizing an LLM for your hardware is usually trial-and-error:

* Which quantization should I use?
* How many GPU layers can I offload without running out of memory?
* What’s the speed trade-off if I go for higher quality?

QBench removes the guesswork:

1. **Profiles your system** – detects GPU, memory limits, and capabilities.
2. **Tests multiple quantization formats** – in one automated run.
3. **Benchmarks performance, memory, and quality** – using real metrics.
4. **Recommends the optimal setup** – based on your chosen priority: speed, quality, memory, or balance.
5. **Generates ready-to-use commands** – so you can deploy immediately.

---

## Quick Start

```bash
git clone https://github.com/AnkitTsj/qbench.git
cd qbench
pip install -r requirements.txt
```

---

## Example Workflow

### 1. Profile your hardware

```bash
python preprocess.py
```

Generates a compatibility report with GPU specs, available memory, and performance estimates.

---

### 2. Optimize a model

```bash
python aqeng.py \
    --workspace-dir "/path/to/workspace" \
    --repo-id "NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF" \
    --filename "Hermes-2-Pro-Llama-3-8B-F16.gguf" \
    --local-filename "hermes-l3-8b-f16.gguf" \
    --gpu-layers 45 \
    --target-formats "q4_k_m" "q6_k" "q8_0" \
    --use-case "performance"
```

QBench will:

* Download the model
* Quantize it into the specified formats
* Benchmark speed, memory use, and quality
* Recommend the best performing setup

---

### 3. View results

Outputs include:

* **Markdown report** – readable summary of results and recommendations
* **CSV** – detailed benchmark metrics for analysis
* **JSON** – structured data for automation

Example (performance mode):

| Format   | Tokens/sec | VRAM (GB) | Quality Score | Recommendation   |
| -------- | ---------- | --------- | ------------- | ---------------- |
| q4\_k\_m | 42.1       | 4.8       | 87.2          | Best for speed   |
| q6\_k    | 32.7       | 6.3       | 93.5          |                  |
| q8\_0    | 25.4       | 7.9       | 98.2          |                  |

---

## Command Options

| Parameter          | Type   | Description                                       |
| ------------------ | ------ | ------------------------------------------------- |
| `--workspace-dir`  | string | Directory for models and outputs                  |
| `--repo-id`        | string | Hugging Face model repo                           |
| `--filename`       | string | Source model filename                             |
| `--local-filename` | string | Local filename                                    |
| `--gpu-layers`     | int    | Layers to offload to GPU                          |
| `--target-formats` | list   | Quant formats to test                             |
| `--use-case`       | string | `performance` / `quality` / `memory` / `balanced` |
| `--kernels-repo`   | string | Optional custom kernels repo                      |
| `--force-rebuild`  | flag   | Rebuild quantizations even if cached              |

---

## Supported Quantization Formats

| Format   | Bits  | Description               | Use Case        |
| -------- | ----- | ------------------------- | --------------- |
| q4\_k\_m | 4-bit | K-means mixed precision   | Balanced        |
| q5\_k\_m | 5-bit | Higher-precision K-means  | Higher quality  |
| q6\_k    | 6-bit | High quality quantization | Quality-focused |
| q8\_0    | 8-bit | Minimal quality loss      | High fidelity   |

---

## Optimization Modes

| Mode        | Focus                              |
| ----------- | ---------------------------------- |
| performance | Max speed, small quality trade-off |
| quality     | Highest possible accuracy          |
| memory      | Lowest memory footprint            |
| balanced    | Good across all metrics            |

---

## Metrics Collected

* **Performance** – tokens/sec, load time, first-token latency
* **Memory** – peak RAM/VRAM usage
* **Quality** – perplexity vs. original model
* **Compatibility** – passes/fails based on system limits

---

## Requirements

* Python 3.8+
* CUDA toolkit (for GPU acceleration)
* 8GB+ RAM recommended

Dependencies: `psutil`, `pandas`, `huggingface-hub`, `pynvml`

---

## Contributing

Pull requests and issues are welcome. Please include:

* Hardware specs
* Command run
* Output/error details
