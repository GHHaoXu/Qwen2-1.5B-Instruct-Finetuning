import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback  # 添加这行导入
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import argparse
from tqdm.auto import tqdm
import logging

# 配置日志格式
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# 硬编码配置
MODEL_PATH = "/root/autodl-tmp/qwen/Qwen2-1.5B-Instruct"  # 也可改为 1.5B/0.5B 等小模型
DATA_PATH = "./dataset/qwen2_sft_format.jsonl"  # 你的数据文件
OUTPUT_DIR = "./qwen2_finetuned"  # 模型保存路径

class ProgressCallback(TrainerCallback):
    """自定义进度条回调"""
    def __init__(self):
        self.progress_bar = None
        self.current_epoch = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.progress_bar = tqdm(total=state.max_steps, desc="Training")

    def on_step_end(self, args, state, control, **kwargs):
        self.progress_bar.update(1)
        if state.log_history and "loss" in state.log_history[-1]:
            loss_str = f"{state.log_history[-1]['loss']:.4f}"
        else:
            loss_str = "NaN"

        self.progress_bar.set_postfix({
            "loss": loss_str,
            "epoch": f"{state.epoch:.1f}/{args.num_train_epochs}"
        })

    def on_train_end(self, args, state, control, **kwargs):
        self.progress_bar.close()

def load_model_and_tokenizer():
    """加载模型和分词器"""
    logger.info(f"Loading model {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")
    return model, tokenizer

def apply_lora(model):
    """应用LoRA适配器"""
    logger.info("Applying LoRA...")
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    return get_peft_model(model, config)

def preprocess_function(examples, tokenizer):
    """格式化数据为模型输入"""
    inputs = [
        f"Instruction: {inst}\nInput: {inp}\nOutput: {out}"
        for inst, inp, out in zip(
            examples["instruction"],
            examples["input"],
            examples["output"]
        )
    ]
    return tokenizer(inputs, truncation=True, max_length=512, padding="max_length")

def train(use_lora=False):
    # 1. 加载模型
    model, tokenizer = load_model_and_tokenizer()
    if use_lora:
        model = apply_lora(model)
        logger.info("✅ LoRA enabled")

    # 2. 加载数据集
    logger.info(f"Loading data from {DATA_PATH}...")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )

    # 3. 训练参数
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        fp16=torch.cuda.is_available(),
        report_to="none",  # 禁用wandb等外部记录
        logging_dir="./logs",
    )

    # 4. 开始训练（添加进度条回调）
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[ProgressCallback()],  # 关键：添加进度条
    )
    
    logger.info("🚀 Starting training...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    logger.info(f"Training complete! Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", action="store_true", help="Enable LoRA")
    args = parser.parse_args()
    train(use_lora=args.lora)