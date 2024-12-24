# LLM Quantization Framework with LoRA

[English](README.md) | [中文](README_zh-CN.md)

> A lightweight framework for training Llama-3.2-1B-Instruct using 4-bit quantization and LoRA (r=32)
> 
> :rocket: Only 2.8GB VRAM required for training (batch_size=1, max_length=1024) :rocket:


A lightweight framework for efficient-LLMs, combining quantization(4bit/8bit) with LoRA. This framework enables fine-tuning of large models on consumer-grade hardware by leveraging:
- **Quantization**: Supports both 4-bit and 8-bit quantization using BitsAndBytes
- **PEFT**: Implements LoRA for parameter-efficient fine-tuning
- **Memory Efficiency**: Optimized for low VRAM usage through quantization and mainstream optimizations
- **Evaluation**: Integrated with lm-eval-harness for model performance assessment
 
## 1.Features

### 1.1 Core Capabilities

- **Advanced Quantization**
  - 8-bit quantization with configurable threshold and skip modules
  - 4-bit quantization with NF4/FP4 data types and nested quantization support


- **LoRA Fine-tuning**
  - Configurable target modules (`q_proj`, `k_proj`, `v_proj`, etc.)
  - Adjustable LoRA rank (r) and alpha scaling

- **Performance Optimizations**
  - Integration with BitsAndBytes for efficient low-precision training
  - Flash Attention 2.0 for reduced memory usage and faster computation
  - Gradient checkpointing for memory-efficient training
  - KV cache management for optimized inference (disabled during training, enabled during inference)
  - Paged optimizer (paged_adamw_32bit) for memory efficiency

### 1.2 Fine-Tuning Datasets

- Using `alpaca_data_cleaned.json` for instruction fine-tuning
- To use other datasets, modify the `preprocess_fn()` in `llama_bnb_peft.py`

### 1.3 Evaluation Pipeline

- Built-in support for various benchmarks: WikiText, ARC_Challenge, MMLU, HellaSwag, IFEval, GPQA
- Evaluation datasets are included in the repository

### 1.4 Llama Prompt Format

The framework uses a standardized prompt template:

```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{Usr_Prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{Response}<|eot_id|>
```

## 2.File Structure
- **Main Training Module**
  - `llama_bnb_peft.py`: Implementation for quantization and LoRA fine-tuning, contains lm-eval evaluation

- **Utility Modules**
  - `print_and_save.py`
    - `print_count_para_mem()`: Monitor model parameters and gradient requirements
    - `inspect_data_once()`: Inspect training data details and formatting

- **Others**
  - `llama_bnb_load.py`: Load models with adapters for evaluation
  - `load_LEH.py`: Direct model loading and evaluation
  - `netCheck.py`: Network connectivity test for users requiring proxy settings

## 3.Quick Start

### Hardware Requirements:
*CUDA Version: 12.2 (Recommendation)*

### Software Dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
python llama_bnb_peft.py \
  --pretrained "path/to/model" \  # Path to the pre-trained model
  --load_in_4bit True \           # Enable 4-bit quantization
  --lora_r 32 \                   # LoRA rank (set to 0 to disable LoRA)
  --data_size 1000 \              # Number of training samples
  --lm_eval_flag False            # Disable evaluation
```

## License and Version
- **Version**: 2024.10.24
- **License**: MIT License

Copyright (c) 2024 [KingdalfGoodman]

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
