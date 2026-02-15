import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 一个常用的中文电商评论二分类情感模型（HuggingFace）
# 标签通常是：0=negative, 1=positive
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

    # probs[..., 1] = positive prob
    results = []
    for i, t in enumerate(texts):
        p_neg = probs[i, 0].item()
        p_pos = probs[i, 1].item()
        label = "正向👍" if p_pos >= p_neg else "负向👎"
        results.append((t, label, p_pos, p_neg))
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

    # 取 base model 的词向量层 (适配 BERT/RoBERTa 这类结构)
    base = getattr(model, "bert", None) or getattr(model, "roberta", None) or model.base_model
    emb_layer = base.embeddings.word_embeddings

    # (1) 先把 input_ids -> embeddings，并让它可求导
    inputs_embeds = emb_layer(input_ids)
    inputs_embeds.requires_grad_(True)  # 关键：确保启用梯度

    # 在这里，显式告诉 PyTorch 保留梯度
    inputs_embeds.retain_grad()  # 保留梯度

    # (2) 用 inputs_embeds 前向（多数 BERT/RoBERTa 结构支持）
    out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    logits = out.logits[0]
    pred = int(torch.argmax(logits).item())

    # (3) 反向：对预测类别 logit 求梯度
    model.zero_grad(set_to_none=True)
    logits[pred].backward()

    grads = inputs_embeds.grad[0]              # [seq_len, hidden]
    embs  = inputs_embeds.detach()[0]          # [seq_len, hidden]
    scores = (grads * embs).abs().norm(p=2, dim=-1)  # [seq_len]

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    # 去掉特殊 token
    pairs = []
    for tok, sc in zip(tokens, scores.tolist()):
        if tok in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        pairs.append((tok, sc))

    pairs.sort(key=lambda x: x[1], reverse=True)
    top = pairs[:topk]

    # 友好打印：把 WordPiece 前缀 ## 合并显示得更像“词”
    def pretty(tok):
        return tok.replace("##", "")

    return pred, [(pretty(t), float(s)) for t, s in top]


def main():
    device = pick_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)

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
    for t, label, p_pos, p_neg in results:
        print(f"- {t}\n  预测: {label} | P(pos)={p_pos:.3f}, P(neg)={p_neg:.3f}")

    # 选一个句子看“模型主要盯着哪些 token”
    text = "客服态度很差，物流慢得离谱，太失望了。"
    pred, top = token_saliency(text, tokenizer, model, device)
    pred_label = "正向👍" if pred == 1 else "负向👎"

    print("\n=== 简易解释（token 重要性 Top） ===")
    print(f"句子: {text}")
    print(f"预测: {pred_label}")
    for tok, sc in top:
        print(f"  {tok:>6s}  score={sc:.4f}")

    print("\n你也可以改 demo_texts，放进去你自己的句子试试。")

if __name__ == "__main__":
    main()
