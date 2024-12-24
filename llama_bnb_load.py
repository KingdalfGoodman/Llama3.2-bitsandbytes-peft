from transformers import AutoModelForCausalLM, HfArgumentParser, BitsAndBytesConfig
from peft import PeftModel

from typing import List, Any, Dict
from dataclasses import dataclass, field

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from print_and_save import *

import os
import logging
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 单卡
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ["HF_HOME"] = "/home/ubuntu/lib/huggingface"
os.environ["XDG_CACHE_HOME"] = "/home/ubuntu/lib/cache"


@dataclass
class ModelConfig:
    pretrained: str = field(default="../Llama-3.2-1B-Instruct", metadata={"help": "Path to pretrained model"})
    # "./results/modelAda_r32_1222_131303"
    adapter_path: str = field(default="./results/modelAda_r32_1222_131303", metadata={"help": "Path to LoRA adapter"})

    dtype: str = field(default="bfloat16", metadata={"help": "Data type for model"})
    flash_attn: str = field(default="flash_attention_2")
    gradient_checkpointing: bool = field(default=False, metadata={"help": "Enable gradient checkpointing"})

    device: str = field(default="cuda", metadata={"help": "Device to use"})
    device_map: str = field(default="auto", metadata={"help": "Device map configuration"})
    trust_remote_code: bool = field(default=True, metadata={"help": "Trust remote code"})

    # "wikitext", "arc_challenge", "mmlu; 5", "hellaswag", "ifeval", "gpqa"
    tasks: List[str] = field(default_factory=lambda: ["wikitext"], metadata={"help": "Tasks to evaluate"})
    num_fewshot: int = field(default=0, metadata={"help": "Number of few-shot examples"})
    eval_batch_size: int = field(default=2, metadata={"help": "Batch size for evaluation"})
    output_path: str = field(default="./results", metadata={"help": "Path to save results"})

    # bnb setting
    load_in_8bit: bool = field(default=False, metadata={"help": "Enable 8-bit quantization"})
    load_in_4bit: bool = field(default=True, metadata={"help": "Enable 4-bit quantization"})
    bnb_4bit_quant_type: str = field(default="nf4", metadata={"choices": ["fp4", "nf4"]})
    bnb_4bit_compute_dtype: str = field(default="bfloat16", metadata={"help": "Computation type for 4-bit"})


class ModelConfigManager:
    def __init__(self, args: ModelConfig):
        self.args = args

    def get_model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "torch_dtype": self.args.dtype,
            "device_map": self.args.device_map,
            "trust_remote_code": self.args.trust_remote_code,
            "attn_implementation": self.args.flash_attn
        }
        quant_config = self._get_quantization_config()
        if quant_config:
            base_kwargs["quantization_config"] = quant_config
        return base_kwargs

    def _get_quantization_config(self) -> BitsAndBytesConfig or None:
        if self.args.load_in_8bit:
            return BitsAndBytesConfig(load_in_8bit=True)
        elif self.args.load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.args.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=self.args.bnb_4bit_compute_dtype
            )
        return None


def main():
    # 参数设置
    parser = HfArgumentParser(ModelConfig)
    args = parser.parse_args_into_dataclasses()[0]
    config_manager = ModelConfigManager(args)

    # 模型加载
    model = AutoModelForCausalLM.from_pretrained(args.pretrained, **config_manager.get_model_kwargs())
    print_count_para_mem(model)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
        print_count_para_mem(model)

    lm = HFLM(pretrained=model, dtype=args.dtype, device=args.device, trust_remote_code=args.trust_remote_code)
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.eval_batch_size,
    )
    results_print(results['results'])

    if args.output_path:
        save_results(results, args.output_path, args)


if __name__ == "__main__":
    main()
