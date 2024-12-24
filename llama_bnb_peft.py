from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer
from transformers import TrainingArguments, HfArgumentParser, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

from typing import List, Any, Dict
from dataclasses import dataclass, field
from datasets import load_dataset, DatasetDict
from transformers import PreTrainedTokenizer, DataCollatorForSeq2Seq

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from print_and_save import *

import os
import logging
from datetime import datetime
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 单卡
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ["HF_HOME"] = "/home/ubuntu/lib/huggingface"
os.environ["XDG_CACHE_HOME"] = "/home/ubuntu/lib/cache"
IGNORE_INDEX = -100


@dataclass
class ModelConfig:
    pretrained: str = field(default="../model/Llama-3.2-1B-Instruct", metadata={"help": "Path to pretrained model"})
    dtype: str = field(default="bfloat16", metadata={"help": "Data type for model"})
    flash_attn: str = field(default="flash_attention_2")
    device: str = field(default="cuda", metadata={"help": "Device to use"})
    device_map: str = field(default="auto", metadata={"help": "Device map configuration"})
    trust_remote_code: bool = field(default=True, metadata={"help": "Trust remote code"})
    output_path: str = field(default="./results", metadata={"help": "Path to save results"})

    # 8位量化参数
    load_in_8bit: bool = field(default=False, metadata={"help": "Enable 8-bit quantization"})
    llm_int8_threshold: float = field(default=5.0, metadata={"help": "Int8 threshold (smaller = more stable)"})
    llm_int8_skip_modules: List[str] = field(default_factory=lambda: ["lm_head", "embed_tokens", "norm", "rotary_emb",
                                                                      "input_layernorm", "post_attention_layernorm"])
    # parser.add_argument("--llm_int8_enable_fp32_cpu_offload", action="store_true", help="GPU不够用，会把部分模型放在CPU上用FP32计算")
    # parser.add_argument("--llm_int8_has_fp16_weight", action="store_true", help="保存了fp16，方便反向传播; 但是微调结果不就适应fp16了吗;可能有细节或历史问题")

    # 4位量化参数
    load_in_4bit: bool = field(default=False, metadata={"help": "Enable 4-bit quantization"})
    bnb_4bit_quant_type: str = field(default="nf4", metadata={"choices": ["fp4", "nf4"]})
    bnb_4bit_use_double_quant: bool = field(default=True, metadata={"help": "Use nested quantization"})
    bnb_4bit_compute_dtype: str = field(default="bfloat16", metadata={"help": "Computation type for 4-bit"})
    group_size: int = field(default=32, metadata={"help": "Group size for quantization"})

    # "o_proj", "gate_proj", "up_proj", "down_proj"; LoRA参数: W+(alpha/r)*BA; 更大的 alpha 意味着 LoRA 的更新有更大的影响力
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj"])
    lora_r: int = field(default=32, metadata={"help": "LoRA attention dimension"})
    lora_alpha: int = field(default=128, metadata={"help": "LoRA alpha scaling"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})

    # 训练参数
    data_path: str = field(default="./data/alpaca/alpaca_data_cleaned.json")
    data_size: int = field(default=1000, metadata={"help": "Number for training"})
    max_length: int = field(default=1024, metadata={"help": "max_length for inps or labels"})
    train_batch_size: int = field(default=1, metadata={"help": "Training batch size"})
    gradient_accumulation_steps: int = field(default=8, metadata={"help": "Gradient accumulation steps"})
    num_train_epochs: int = field(default=1, metadata={"help": "Number of training epochs"})

    learning_rate: float = field(default=2e-4, metadata={"help": "Learning rate"})
    warmup_steps: int = field(default=100, metadata={"help": "Warmup steps"})

    logging_steps: int = field(default=10, metadata={"help": "Logging interval"})
    save_strategy: str = field(default="no", metadata={"help": "The checkpoint save strategy to use"})
    optim: str = field(default="paged_adamw_32bit", metadata={"help": "The optimizer to use"})  # adamw_torch
    gradient_checkpointing: bool = field(default=True, metadata={"help": "Enable gradient checkpointing"})

    lm_eval_flag: bool = field(default=False, metadata={"help": "Enable evaluate model"})
    # "wikitext", "arc_challenge", "mmlu; 5", "hellaswag", "ifeval", "gpqa"
    tasks: List[str] = field(default_factory=lambda: ["wikitext"], metadata={"help": "Tasks to evaluate"})
    num_fewshot: int = field(default=0, metadata={"help": "Number of few-shot examples"})
    eval_batch_size: int = field(default=2, metadata={"help": "Batch size for evaluation"})


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
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=self.args.llm_int8_threshold,
                llm_int8_skip_modules=self.args.llm_int8_skip_modules,
            )
        elif self.args.load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.args.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.args.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=self.args.bnb_4bit_compute_dtype,
                group_size=self.args.group_size
            )
        return None

    def get_lora_config(self) -> LoraConfig or None:
        if self.args.lora_r == 0:
            return None

        return LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            target_modules=self.args.lora_target_modules,
            lora_dropout=self.args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",  # for Llama
        )

    def get_training_args(self) -> TrainingArguments:
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        return TrainingArguments(
            output_dir=os.path.join(self.args.output_path, f"modelAda_r{self.args.lora_r}_{timestamp}"),
            per_device_train_batch_size=self.args.train_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            num_train_epochs=self.args.num_train_epochs,
            learning_rate=self.args.learning_rate,
            warmup_steps=self.args.warmup_steps,
            bf16=(self.args.dtype == "bfloat16"),
            logging_steps=self.args.logging_steps,
            save_strategy=self.args.save_strategy,
            optim=self.args.optim,
            gradient_checkpointing=self.args.gradient_checkpointing,
        )


def prepare_dataset(data_path: str, tokenizer: PreTrainedTokenizer, train_ratio=0.8, max_length=512, data_size=10000) -> DatasetDict:
    dataset = load_dataset('json', data_files=data_path, split=f"train[:{data_size}]", num_proc=16)

    # 定义一个数据处理函数 , tokenizer=tokenizer
    def preprocess_fn(example):
        user_prompt = tokenizer("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
                                + "\n".join([example["instruction"], example["input"]]).strip()
                                + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                                , add_special_tokens=False)
        response = tokenizer(example["output"] + "<|eot_id|>", add_special_tokens=False)

        # 合并输入和标签; -100 is necessary
        input_ids = user_prompt["input_ids"] + response["input_ids"]
        labels = [IGNORE_INDEX] * len(user_prompt["input_ids"]) + response["input_ids"]
        attention_mask = user_prompt["attention_mask"] + response["attention_mask"]
        input_ids, attention_mask, labels = map(lambda x: x[:max_length], [input_ids, attention_mask, labels])

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    # 应用数据处理函数; remove_columns=dataset.column_names will be warning
    tokenized_dataset = dataset.map(preprocess_fn, remove_columns=list(dataset.column_names), num_proc=8)

    split_datasets = tokenized_dataset.train_test_split(train_size=train_ratio, seed=42)
    # split_datasets.set_format(type='torch') cause of DataCollatorForSeq2Seq, don't need set_format.

    return DatasetDict({'train': split_datasets['train'], 'validation': split_datasets['test']})


def main():
    # 参数设置
    parser = HfArgumentParser(ModelConfig)
    args = parser.parse_args_into_dataclasses()[0]
    config_manager = ModelConfigManager(args)

    # 模型加载
    model = AutoModelForCausalLM.from_pretrained(args.pretrained, **config_manager.get_model_kwargs())
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()   # Not compatible with KV_cache, as it modifies the computation graph structure
    # print_count_para_mem(model)
    if lora_config := config_manager.get_lora_config():
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<|finetune_right_pad_id|>"})
        if args.load_in_8bit or args.load_in_4bit:
            model = prepare_model_for_kbit_training(model)  # if not quantFormat will be float32 by kbit_training()
        model = get_peft_model(model, lora_config)
        # print_count_para_mem(model)

        # 创建数据集和整理器
        dataset = prepare_dataset(data_path=args.data_path,
                                  tokenizer=tokenizer, max_length=args.max_length, data_size=args.data_size)

        training_args = config_manager.get_training_args()
        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=dataset['train'],
            # eval_dataset=dataset['validation'],
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, pad_to_multiple_of=8)
        )
        inspect_data_once(trainer, batch_index=0, num_samples=2, token_level=True, show_padding_labels=True)

        # 模型训练
        model.config.use_cache = False  # 训练时禁用KV cache
        trainer.train()
        trainer.save_model()

    if args.lm_eval:
        model.gradient_checkpointing_disable()
        model.config.use_cache = True  # 推理时启用KV cache

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
