import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

# --- 配置 ---
base_model_path = "Qwen/Qwen1.5-1.8B-Chat"
# 【重要】确保这里的路径是你最新训练的 "jiangjian" 模型的路径
lora_path = "./girl_qwen_chat_1.8b"

# --- 加载分词器 ---让 transformers 能正确加载 Qwen 自定义代码逻辑。
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)


# --- 定义生成函数 ---
def generate_response(model, instruction):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": instruction}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)      # 用 apply_chat_template() 自动生成 Qwen 风格的 prompt（带 <|im_start|>user 等标记）
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")        # tokenizer([text], return_tensors="pt") → 转成 tensor。并送入gpu

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=100,     # 最多生成 100 个新 token。
        temperature=0.9  # 让我们用那个效果最好的“黄金温度”
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


# --- 问题列表 ---
test_prompts = [
    "考研报过名了？",
    "今天英语单词背了吗？",
    "评价一下我今天的穿搭。",
    "我看星穹铁道策划是想死了",
    "如何评价你学校被称为“上海小清华”？",
    "你对白厄怎么看？"
]

# --- 存储结果的字典 ---
results = {prompt: {} for prompt in test_prompts}

# --- 严格分离的测试流程 ---

# 1. 加载并测试纯净的原始模型
print("=" * 50)
print("正在加载和测试 🤖 纯净的原始模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,     # 使用 bfloat16 减少显存占用。
    device_map="auto",
    trust_remote_code=True
)
for prompt in test_prompts:
    response = generate_response(base_model, prompt)
    results[prompt]["base"] = response

# 2. 【核心】在测试完原始模型后，再进行“改装”
print("\n正在将LoRA适配器应用到模型上...")
tuned_model = PeftModel.from_pretrained(base_model, lora_path)      # 把 LoRA 参数注入到原模型，得到一个改装模型。
print("模型改装完成！")

# 3. 测试改装后的模型
print("\n正在测试 🧑‍🎨 您的微调模型...")
for prompt in test_prompts:
    response = generate_response(tuned_model, prompt)
    results[prompt]["tuned"] = response

# --- 卸载模型，释放显存 ---
del base_model
del tuned_model
gc.collect()
torch.cuda.empty_cache()

# --- 最终结果对比展示 ---
print("\n\n" + "=" * 25 + " 最终对比 " + "=" * 25)
for prompt in test_prompts:
    print("\n" + "-" * 50)
    print(f"🤔 问题: {prompt}")
    print("-" * 50)
    print(f"🤖 原始模型的回答:\n{results[prompt]['base']}")
    print(f"\n🧑‍🎨 你的模型的回答:\n{results[prompt]['tuned']}")

print("\n" + "=" * 50)
print("所有对比测试完成！")