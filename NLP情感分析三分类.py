import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 使用已有的二分类模型
MODEL_NAME = "uer/roberta-base-finetuned-jd-binary-chinese"

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@torch.no_grad()
def predict(texts, tokenizer, model, device, max_len=128):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    ).to(device)

    logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)

    results = []
    for i, t in enumerate(texts):
        p_neg = probs[i, 0].item()
        p_neu = probs[i, 1].item()
        p_pos = probs[i, 2].item()

        # 根据概率最大值预测标签
        if p_pos >= p_neg and p_pos >= p_neu:
            label = "正向👍"
        elif p_neg >= p_pos and p_neg >= p_neu:
            label = "负向👎"
        else:
            label = "中性😐"

        results.append((t, label, p_pos, p_neg, p_neu))
    return results

def token_saliency(text, tokenizer, model, device, max_len=128, topk=12):
    """
    一个“够用且直观”的解释：对预测类别logit求 embedding 梯度，
    用 |grad * emb| 的 L2 范数当作每个 token 的重要性分数。
    """
    model.eval()

    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    base = getattr(model, "bert", None) or getattr(model, "roberta", None) or model.base_model
    emb_layer = base.embeddings.word_embeddings

    inputs_embeds = emb_layer(input_ids)
    inputs_embeds.requires_grad_(True)
    inputs_embeds.retain_grad()

    out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    logits = out.logits[0]
    pred = int(torch.argmax(logits).item())

    model.zero_grad(set_to_none=True)
    logits[pred].backward()

    grads = inputs_embeds.grad[0]              # [seq_len, hidden]
    embs  = inputs_embeds.detach()[0]          # [seq_len, hidden]
    scores = (grads * embs).abs().norm(p=2, dim=-1)  # [seq_len]

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    pairs = []
    for tok, sc in zip(tokens, scores.tolist()):
        if tok in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        pairs.append((tok, sc))

    pairs.sort(key=lambda x: x[1], reverse=True)
    top = pairs[:topk]

    def pretty(tok):
        return tok.replace("##", "")

    return pred, [(pretty(t), float(s)) for t, s in top]

def main():
    device = pick_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 加载预训练模型，并修改输出层为三分类（num_labels=3）
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3, ignore_mismatched_sizes=True)

    # 重新初始化输出层
    model.classifier = torch.nn.Linear(model.classifier.in_features, 3)

    # 将模型加载到指定设备
    model.to(device)

    demo_texts = [
        "这手机续航太顶了，屏幕也很舒服，真香！",
        "客服态度很差，物流慢得离谱，太失望了。",
        "一般般吧，没想象中好，但也不算差。",
        "做工很差，按键松动，还发热，别买。",
        "包装精致，手感不错，性价比很高。",
        "我在网上搜了半天都没找到，问ai问出来了,到底想干啥",
        "怎么还多了1.12,收了五分钟的利息",
        "我室友打游戏连麦到两点,他也考研,哈哈,我想下去铜丝他",
        "你知道吗,切比雪夫不等式里藏着亲亲😘,每天都有新发现",
        "我真想玩鸣潮了,有空买个号",
        "你没学第四章,快学,学了就能做24年第十题了",
        "这家伙在说什么呢,真会求吗？来个当场练习",
        "我是真得给你免打扰了,算了我给手机开个免打扰吧",
        "惊天大op,要是我一天中最惬意的时刻是打开原神我看我得在6楼攻击水泥地面了",
        "你看看太好玩了",
        "整个凉皮肉夹馍吃吃,还有冰峰",
        "耍了三年导致的,我正儿八经耍了三年,有点放纵了,属于是叛逆期延长到大学来了,但现在我过去了"
    ]

    print("\n=== 情感预测 ===")
    results = predict(demo_texts, tokenizer, model, device)
    for t, label, p_pos, p_neg, p_neu in results:
        print(f"- {t}\n  预测: {label} | P(pos)={p_pos:.3f}, P(neg)={p_neg:.3f}, P(neu)={p_neu:.3f}")

    # 选一个句子看“模型主要盯着哪些 token”
    text = "客服态度很差，物流慢得离谱，太失望了。"
    pred, top = token_saliency(text, tokenizer, model, device)
    pred_label = "正向👍" if pred == 2 else "负向👎" if pred == 0 else "中性😐"

    print("\n=== 简易解释（token 重要性 Top） ===")
    print(f"句子: {text}")
    print(f"预测: {pred_label}")
    for tok, sc in top:
        print(f"  {tok:>6s}  score={sc:.4f}")

    print("\n你也可以改 demo_texts，放进去你自己的句子试试。")

if __name__ == "__main__":
    main()
