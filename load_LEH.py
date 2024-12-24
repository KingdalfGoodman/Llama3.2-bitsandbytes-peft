import argparse
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

import os
import json
from datetime import datetime
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 单卡
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ["HF_HOME"] = "/home/ubuntu/lib/huggingface"
os.environ["XDG_CACHE_HOME"] = "/home/ubuntu/lib/cache"


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


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model with specified tasks.")
    # "wikitext", "arc_challenge", "mmlu; 5", "hellaswag", "ifeval", "gpqa"
    parser.add_argument("--tasks", nargs="+", default=["wikitext"])
    parser.add_argument("--num_fewshot", type=int, default=0, help="Number of few-shot examples to use")

    # Llama-3.2-1B-Instruct; 3B
    parser.add_argument("--pretrained", type=str, default="../Llama-3.2-1B-Instruct", help="Path to the pretrained model or model name")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for evaluation")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device_map", type=str, default="auto", help="Device map configuration (e.g., 'auto', 'balanced')")
    parser.add_argument("--trust_remote_code", default=True, help="Trust remote code in the model")

    parser.add_argument("--output_path", type=str, default="./results", help="Path to save evaluation results")
    args = parser.parse_args()

    model_args = {
        "pretrained": args.pretrained,
        "dtype": args.dtype,
        "device": args.device,
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code
    }
    lm = HFLM(**model_args)

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
    )
    results_print(results['results'])

    if args.output_path:
        save_results(results, args.output_path, args)


if __name__ == "__main__":
    main()
