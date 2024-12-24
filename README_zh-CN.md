# LLM Quantization Framework with LoRA

[English](README.md) | [中文](README_zh-CN.md)

> 案例: 使用本框架训练 Llama-3.2-1B-Instruct 时, 采用 4-bit 量化和 LoRA (r=32) 
> :rocket: 仅需 2.8GB 显存 (batch_size=1, max_length=1024) :rocket:

这是通用的 LLM 轻量级框架, 结合了量化(4bit/8bit) 和 LoRA 技术。

- **量化(Quantization)**: 使用 BitsAndBytes 支持 4-bit 和 8-bit 量化
- **PEFT**: 实现 LoRA 进行参数高效微调
- **内存效率**: 通过量化和主流优化方法实现低显存使用
- **评估**: 集成 lm-eval-harness 进行模型性能评估

## 1. 功能特性

### 1.1 核心功能

- **量化**
  - 8-bit 量化, 支持可配置阈值和skip模块
  - 4-bit 量化, 支持 NF4/FP4 数据类型和嵌套量化

- **LoRA 微调**
  - 可配置目标模块 (`q_proj`, `k_proj`, `v_proj` 等)
  - 可调节的 LoRA rank (r) 和 alpha 缩放

- **性能优化**
  - 集成 BitsAndBytes 实现高效的低精度训练
  - 使用 Flash Attention 2.0 降低内存使用并加速计算
  - 使用 Gradient checkpointing 实现内存高效训练
  - KV cache 管理优化推理 (训练期间禁用,推理时启用)
  - 使用分页优化器 (paged_adamw_32bit), 提高内存效率

### 1.2 微调数据集

- 使用 `alpaca_data_cleaned.json` 进行指令微调
- 如需使用其他数据集,请修改 `llama_bnb_peft.py` 中的 `preprocess_fn()`

### 1.3 评估流程

- 内置支持多种基准测试: WikiText, ARC_Challenge, MMLU, HellaSwag, IFEval, GPQA
- 评估数据集已包含在代码库中, 可根据数据集的readme进行下载

### 1.4 Llama Prompt 格式

框架使用标准化的 prompt 模板:

```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{Usr_Prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{Response}<|eot_id|>
```

## 2. 文件结构

- **主训练模块**
  - `llama_bnb_peft.py`: 实现量化和 LoRA 微调,包含 lm-eval 评估

- **工具模块**
  - `print_and_save.py`
    - `print_count_para_mem()`: 监控模型参数和梯度需求
    - `inspect_data_once()`: 检查训练数据细节和格式

- **其他**
  - `llama_bnb_load.py`: 加载带适配器的模型进行评估
  - `load_LEH.py`: 直接加载模型进行评估
  - `netCheck.py`: 网络连接测试, 用于需要代理设置的用户

## 3. 快速使用

### 硬件要求:
*CUDA Version: 12.2 (推荐)*

### 软件依赖:
```bash
pip install -r requirements.txt
```

### 基本用法

```bash
python llama_bnb_peft.py \
  --pretrained "path/to/model" \  # 预训练模型路径
  --load_in_4bit True \           # 启用 4-bit 量化
  --lora_r 32 \                   # LoRA rank (设为 0 则禁用 LoRA)
  --data_size 1000 \              # 训练样本数量
  --lm_eval_flag False            # 禁用评估
```

## 许可证和版本
- **版本**: 2024.10.24
- **许可证**: MIT License

Copyright (c) 2024 [KingdalfGoodman]

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件