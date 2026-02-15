import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import json
import os

# --- 配置 ---
base_model_path = "Qwen/Qwen1.5-1.8B-Chat"
# 选择你想聊天的AI朋友
lora_path = "./girl_qwen_chat_1.8b"
# 定义“日记本”的文件名，用 .jsonl（一行一条 JSON）格式存储每一轮对话。
history_file = f"chat_history_{os.path.basename(lora_path)}.jsonl"
# 定义“短期记忆”的长度（只保留最近的N轮对话作为上下文）
MEMORY_TURNS = 10  # 你可以根据需要调整这个数字

# --- 加载模型 ---
print("正在加载模型...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, lora_path)
print("模型加载完成！")
print("=" * 50)


# --- 长期记忆管理 ---
def load_history():
    """从文件加载完整的聊天历史"""
    if not os.path.exists(history_file):
        # 如果文件不存在，返回一个包含系统指令的初始历史
        return [{"role": "system", "content": "You are a helpful assistant that mimics the user's friend."}]

    with open(history_file, 'r', encoding='utf-8') as f:
        # 逐行读取JSONL文件
        return [json.loads(line) for line in f]


def save_message(role, content):
    """将一条新消息追加到历史文件"""
    with open(history_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps({"role": role, "content": content}) + '\n')


# --- 聊天核心函数 ---
def chat_with_long_term_memory(user_message):
    # 1. 加载完整的历史记录
    full_history = load_history()

    # 2. 将新消息加入完整历史并保存
    full_history.append({"role": "user", "content": user_message})
    save_message("user", user_message)

    # 3. 【关键】构建用于本次推理的“短期记忆”
    # 我们只取系统指令和最近的几轮对话
    system_prompt = [full_history[0]]
    recent_history = full_history[-(MEMORY_TURNS * 2):]  # 取最近10轮对话（10问+10答）
    context_for_inference = system_prompt + recent_history

    # 4. 模型推理
    text = tokenizer.apply_chat_template(context_for_inference, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=200,
        temperature=0.7
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 5. 将模型的回答也保存到长期记忆中
    save_message("assistant", response_text)

    return response_text


# --- 启动聊天循环 ---
print(f"你好！开始和 {os.path.basename(lora_path)} 聊天吧。")
print(f"聊天记录将保存在 {history_file} 中。(输入 '退出' 来结束对话)")
print("-" * 50)

while True:
    user_input = input("你: ")
    if user_input.lower() in ["退出", "exit", "quit"]:
        print("AI: 好的，下次再见！")
        break

    ai_response = chat_with_long_term_memory(user_input)
    print(f"AI: {ai_response}")