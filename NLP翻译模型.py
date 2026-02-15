from transformers import MarianMTModel, MarianTokenizer

# 加载预训练的翻译模型和tokenizer
model_name = "Helsinki-NLP/opus-mt-en-zh"  # 这是中文到英文的翻译模型
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)


def translate(text, model, tokenizer):
    # 使用tokenizer编码输入文本
    encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # 使用模型进行翻译
    translated = model.generate(**encoded)

    # 解码生成的翻译结果
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

    return translated_text


# 输入中文文本
text = "I love you, but I don't know if it's going to be you or me. I just love what you do. You're like my friend, so it doesn't feel like I've been there. It's just something that's happened. I know you want to see me again, that you're going through something with me, and you can't take it anymore. But I'm sorry. There's no way I can do it to you. If you don. Because it has to. And I know that. That I love and I want you to know. This is what I wanted to do to me when I was 12 years old. So please, please don' t be upset. Don't be angry. Please don t do what needs to happen. We'll see"
translated_text = translate(text, model, tokenizer)
print("翻译结果：", translated_text)
