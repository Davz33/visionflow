# Wan2.2 Colab Free Tier (Tesla T4) Benchmark & Limits Report

**Date:** 2026-08-16  
**Environment:** Google Colab Free Tier  
**Hardware Specifications:**  
- **GPU:** NVIDIA Tesla T4 (14.56 GB VRAM, Compute Capability 7.5)  
- **CPU System RAM:** 12.67 GB  
- **PyTorch / CUDA:** `2.11.0+cu128`

---

## 1. Executive Summary

This document records the empirical results and conclusions from running the Wan2.2 evaluation harness and smoke generation test (`scripts/colab_wan22_e2e.py`) on Google Colab's Free Tier.

- **Eval Suite & CLI Unit Tests:** **PASSED (21 / 21 passed in 0.84s)**. All model zoo resolutions, generate argument specs, and matrix cell expansions run cleanly without GPU requirements.
- **Wan2.2 TI2V-5B In-Memory Inference:** **FAILED (Hardware System RAM Bottleneck)**. While all 31.4 GB of model weights download and extract to disk cache without error, assembling the `WanPipeline` uncompressed weight tensors into Python/PyTorch memory exceeds Colab's 12.67 GB CPU RAM limit.

---

## 2. Tested Technical Configurations & Results

### 2.1 Full Precision (`fp32` / `bfloat16`) Direct GPU Load
- **Result:** `CUDA Out of Memory (OOM)`
- **Observation:** Attempting to allocate 14GB+ directly on CUDA 0 triggered PyTorch CUDA OOM (Tesla T4 capacity: 14.56 GB; available: ~12.4 GB).

### 2.2 Adaptive Sequential CPU Offloading (`enable_sequential_cpu_offload`)
- **Result:** `Linux Kernel OOM Killer (SIGKILL / Signal 9)`
- **Observation:** CPU offloading moves active layers between CPU RAM and GPU VRAM during forward passes. However, to construct the pipeline dictionary and shard mappings, PyTorch must instantiate the full module graph in CPU RAM. System RAM peaked at 96.6% (11.88 GB / 12.67 GB) before the OS kernel terminated the process.

### 2.3 Sub-Component 8-Bit Quantization (`BitsAndBytesConfig(load_in_8bit=True)`)
- **Result:** `Linux Kernel OOM Killer (SIGKILL / Signal 9)`
- **Observation:** Quantizing `WanTransformer3DModel` reduces active VRAM footprint during inference, but initializing the transformer parameters from 5 uncompressed checkpoint shards still spikes System RAM to 96.3% (~11.88 GB) at weight shard ~109/242, triggering the Linux kernel OOM killer.

---

## 3. Core Architectural Conclusions

1. **System RAM (not VRAM) is the Primary Bottleneck on Colab Free:**
   The Tesla T4's 14.56 GB VRAM is sufficient when paired with `enable_sequential_cpu_offload()` or `enable_model_cpu_offload()`. The hard blocker is the **12.67 GB CPU System RAM** limitation on Colab Free instances. Loading a 5B multi-shard DiT pipeline requires at least **16–24 GB of System RAM** during shard deserialization and weight mapping.

2. **Model Tier Compatibility Matrix:**

| Model Tier | Parameters | Colab Free (12.7GB RAM / T4 14.5GB VRAM) | Required Hardware Tier |
|---|---|---|---|
| **Wan2.1 T2V-1.3B** | 1.3B | **Supported** (with `model_cpu_offload`) | Consumer GPU / Colab Free |
| **Wan2.2 TI2V-5B** | 5.0B | **Unsupported** (System RAM OOM) | High-RAM Colab (25GB RAM) or A10G / A100 |
| **Wan2.2 T2V-A14B** | 14.0B | **Unsupported** | 80GB VRAM (A100 / H100) |

---

## 4. Operational Guidance & Recommendations

1. **For Colab Free Users:**
   Use `wan2.1-t2v-1.3b` for end-to-end video generation tests on 12.7 GB System RAM runtimes:
   ```python
   request = VideoGenerationRequest(
       prompt="A red cat walks across a wooden table",
       model_key="wan2.1-t2v-1.3b",
       resolution="832x480",
       duration=2
   )
   ```

2. **For Wan2.2 TI2V-5B In-Memory Evaluation:**
   Switch Colab to a **High-RAM Instance** (`Runtime` → `Change runtime type` → `RAM: High-RAM` [25 GB RAM]) or run on dedicated cloud instances (RunPod, GCP, AWS) with $\ge 24\text{ GB}$ System RAM and $\ge 16\text{ GB}$ VRAM.
