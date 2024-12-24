import os
import json
from datetime import datetime


def inspect_data_once(trainer,
                      batch_index=0,
                      num_samples=2,
                      token_level=True,
                      show_padding_labels=True):
    """
    一个多合一的检查函数，结合原先4个函数的核心逻辑。

    Args:
        trainer: Huggingface Trainer instance
        batch_index: 只检查哪一个batch（默认为第0个）
        num_samples: batch里最多检查多少个样本
        token_level: 是否进行 token-by-token 的分析 (类似 inspect_tokens_in_batch)
        show_padding_labels: 是否打印 padding 位的labels检查 (类似 inspect_padding_labels)
    """

    # 1. 全局数据信息
    train_dataset = trainer.train_dataset
    print("=== Dataset Basic Info ===")
    print(f"Dataset size: {len(train_dataset)}")
    # 只打印一次全局分布统计
    lengths = [len(sample['input_ids']) for sample in train_dataset]
    print(f"Min length: {min(lengths)}, Max length: {max(lengths)}, Avg length: {sum(lengths) / len(lengths):.2f}")

    # 2. 打印batch（collated后）的信息
    dataloader = trainer.get_train_dataloader()
    tokenizer = trainer.tokenizer  # 如果警告烦人，可升级transformers或改用 processing_class

    # 确保 batch_index 合法
    for i, batch in enumerate(dataloader):
        if i == batch_index:
            print(f"\n=== Inspect Batch {i} ===")
            print(f"Keys in batch: {list(batch.keys())}")
            for k, v in batch.items():
                print(f"{k} shape: {v.shape}, dtype: {v.dtype}, device: {v.device}")
            # 只检查这个batch就跳出
            break

    # 3. 从该 batch 里检查若干 sample
    #    注意 trainer.get_train_dataloader() 每次迭代都会返回一个 batch，如果需要反复访问，
    #    可以先 list(dataloader) 或者写自定义函数保证可重复迭代。
    batch_samples = list(dataloader)[batch_index]
    input_ids = batch_samples['input_ids']
    labels = batch_samples['labels']
    attention_mask = batch_samples['attention_mask']

    # 若 num_samples > batch大小，则截断
    max_samples = min(num_samples, input_ids.shape[0])

    for sample_idx in range(max_samples):
        print(f"\n--- Sample {sample_idx} in Batch {batch_index} ---")
        sample_input = input_ids[sample_idx]
        sample_label = labels[sample_idx]
        sample_attn = attention_mask[sample_idx]

        # [A] 简要打印：非padding长度和文本预览
        non_pad = (sample_input != tokenizer.pad_token_id).sum().item()
        text_preview = tokenizer.decode(sample_input[:50])  # decode前50个token
        print(f"Non-padding length: {non_pad}")
        print(f"Text preview (first 50 tokens decoded): {text_preview}")

        # [B] 如果需要看 padding labels
        if show_padding_labels:
            padding_positions = (sample_input == tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
            if len(padding_positions) > 0:
                padding_labels = sample_label[padding_positions]
                unique_padding_labels = padding_labels.unique()
                print(f"Padding token positions: {padding_positions[:10].tolist()}...(total {len(padding_positions)})")
                print(f"Unique labels at padding positions: {unique_padding_labels.tolist()}")
            else:
                print("No padding tokens in this sample.")

        # [C] 如果要看 token-level
        if token_level:
            print("\nToken-level analysis:")
            print(f"{'Pos':^5} | {'Token':^12} | {'ID':^6} | {'Label':^6} | {'Attn':^5}")
            print("-" * 50)
            for pos, (tid, lab, att) in enumerate(zip(sample_input, sample_label, sample_attn)):
                tok = tokenizer.decode([tid]).replace('\n', '\\n').replace(' ', '▢')
                tok = (tok[:9] + "...") if len(tok) > 12 else tok
                print(f"{pos:^5} | {tok:^12} | {tid:^6} | {lab:^6} | {att:^5}")
        print("-" * 50)


def print_count_para_mem(model):
    """打印模型的参数数量和内存占用统计"""
    total_params = 0  # 总参数数量
    total_memory = 0  # 总内存占用(bytes)
    grad_params = 0   # 需要梯度的参数数量
    grad_memory = 0   # 需要梯度的参数内存占用(bytes)
    lora_params = 0

    for name, param in model.named_parameters():
        if "layers.0" in name or "layers." not in name:
            print(f"{name}: {param.numel()} {param.dtype} {param.shape} {param.requires_grad}")
        num_params = param.numel()
        total_params += num_params
        if str(param.dtype) == "torch.uint8":
            total_params += num_params  # uint8 contain 2 parameters

        dtype_to_bytes = {"torch.float32": 4, "torch.bfloat16": 2, "torch.int8": 1, "torch.uint8": 1}
        if str(param.dtype) not in dtype_to_bytes:
            raise ValueError(f"未处理的数据类型: {param.dtype}")
        mem_size = num_params * dtype_to_bytes[str(param.dtype)]
        total_memory += mem_size

        if param.requires_grad:
            grad_params += num_params  # uint8 not used in grad.
            grad_memory += mem_size
        if "lora" in name:
            lora_params += num_params

    print("=== 参数统计 ===")
    print(f"总参数量: {total_params:,}", f"LoRA_params_count: {lora_params}" if lora_params != 0 else "")
    print(f"需要梯度的参数量: {grad_params:,} ({100 * grad_params / total_params:.2f}%)")

    print("=== 内存占用统计 ===")
    print(f"总内存占用: {total_memory / 1024 / 1024 / 1024:.2f} GB")
    print(f"需要梯度的参数内存占用: {grad_memory / 1024 / 1024 / 1024:.2f} GB ({100 * grad_memory / total_memory:.2f}%)\n\n")


def results_print(task_results):
    mmlu_tasks_list = [task for task in task_results if task.startswith('mmlu')]
    if mmlu_tasks_list:
        print(f"\nTask: mmlu")
        print(f"  acc,none: {task_results['mmlu']['acc,none']:.6f}")
        print(f"  acc_stderr,none: {task_results['mmlu']['acc_stderr,none']:.6f}")
        # 计算macro_avg/acc
        mmlu_tasks = [task for task in mmlu_tasks_list if task.startswith('mmlu_')]
        macro_avg_acc = sum(task_results[task]['acc,none'] for task in mmlu_tasks) / len(mmlu_tasks)
        print(f"  Macro Average Accuracy on MMLU: {macro_avg_acc:.6f}")

    for task_name, metrics in task_results.items():
        if task_name in mmlu_tasks_list:
            continue
        print(f"\nTask: {task_name}")
        for metric_name, value in metrics.items():
            if metric_name in ['alias', '_len_'] or not isinstance(value, (float, int)):
                continue  # 跳过非指标项
            if isinstance(value, float):
                print(f"  {metric_name}: {value:.6f}")
            else:
                print(f"  {metric_name}: {value}")


def save_results(results, output_path, args):
    os.makedirs(output_path, exist_ok=True)
    model_name = os.path.basename(args.pretrained)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")                            # 生成时间戳
    results_without_samples = {k: v for k, v in results.items() if k != 'samples'}  # 过滤掉 samples

    full_results = {
        "model_name": model_name,
        "timestamp": timestamp,
        "arguments": vars(args),  # 将参数对象转换为字典
        "evaluation_results": results_without_samples
    }
    results_file = os.path.join(output_path, f"{model_name}_{timestamp}_results.json")
    with open(results_file, 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")
