from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 加载预训练的 GPT-2 模型和 tokenizer
model_name = "gpt2"  # 你也可以使用其他模型如 "gpt2-medium", "gpt2-large", "gpt2-xl"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# 输入提示
prompt_text = "I love you"

# 将输入提示转换为模型可以处理的格式
inputs = tokenizer.encode(prompt_text, return_tensors="pt")

# 生成文本
outputs = model.generate(
    inputs,
    max_length=200,  # 设置最大生成长度
    num_return_sequences=1,  # 生成文本的数量
    no_repeat_ngram_size=2,  # 防止重复的 n-grams
    temperature=0.7,  # 控制生成文本的多样性
    top_k=50,  # 只从最有可能的 k 个单词中采样
    top_p=0.95,  # 采样的累积概率阈值
    do_sample=True,  # 启用采样来生成文本
)

# 解码生成的文本
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("生成的文本：")
print(generated_text)
