import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# 1. 模型 & 数据路径
model_name = "Qwen/Qwen1.5-1.8B-Chat"
dataset_path = "./girl_cleaned.jsonl"

# 2. 4bit量化，量化配置：4bit 加载模型，能显著减少显存占用（1.8B 模型从十几GB降到约5GB）。
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# 3. 加载模型和 Tokenizer，加载模型和 Tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",      # 自动分配模型到 GPU。
    trust_remote_code=True # 禁用缓存，否则 LoRA 训练会报错
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)       # 把文本转为模型可读的 token 序列。
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 4. LoRA 配置，LoRA：只训练一小部分可学习矩阵（低秩分解），其余权重冻结。
lora_config = LoraConfig(
    r=16,       # 低秩维度。
    lora_alpha=32,      # 缩放系数。
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]        # 指定对哪些层注入 LoRA（常用于注意力层和前馈层）
)
model = get_peft_model(model, lora_config)

# 5. 训练参数
training_arguments = TrainingArguments(
    output_dir="./results_jiangjian",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    learning_rate=1e-4,
    bf16=True,      # bf16：比 fp16 稳定。
    max_grad_norm=0.3,
    num_train_epochs=15,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_steps=50,      # 每 50 step 保存一次 checkpoint。
)

# 6. 【终极核心修正】为兼容旧版trl，格式化函数改为处理单个样本，用于把每条数据转成“对话格式”，符合 Qwen Chat 模型风格。
# 非常关键：这个版本的 SFTTrainer 期望函数返回字符串而不是 token 序列，所以用了 tokenize=False。
def formatting_chat_template_single(example):
    # 'example' 在这里是一个单独的字典, e.g., {'instruction': '你好', 'output': '你好啊'}
    # 我们不再需要 for 循环
    messages = [
        {"role": "system", "content": "You are a helpful assistant that mimics the user's friend."},
        {"role": "user", "content": example['instruction']}, # 直接访问字符串
        {"role": "assistant", "content": example['output']}  # 直接访问字符串
    ]
    # SFTTrainer (batched=False) 期望这个函数返回一个格式化好的字符串
    return tokenizer.apply_chat_template(messages, tokenize=False)

# 7. 加载数据集
dataset = load_dataset("json", data_files=dataset_path, split="train")

# 8. 创建 SFTTrainer 并训练：SFTTrainer 是 Hugging Face 的高级封装：
# 自动处理 tokenization、padding、梯度裁剪。自动应用 LoRA、量化配置。
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    formatting_func=formatting_chat_template_single, # 使用修正后的单样本处理函数，指定如何把原始数据转换为模型输入。
    args=training_arguments,
)

# 9. 开始训练
trainer.train()

# 10. 保存模型，保存 LoRA 适配权重 和 Tokenizer。
save_path = "./gir_qwen_chat_1.8b"
trainer.model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"✅ 姜简风格模型训练完成！已保存到 {save_path}")