import torch
from transformers import AutoTokenizer
from bert_kan_3classifier import BertWithChebyshevKAN

# 定义情感类别映射（逆映射）
target_map = {'悲伤': 0, '快乐': 1, '厌恶': 2, '愤怒': 3, '喜欢': 4, '惊讶': 5, '恐惧': 6}
emotion_map = {v: k for k, v in target_map.items()}  # 创建逆向映射

# 加载模型和标记器
model = BertWithChebyshevKAN.from_pretrained('./training_bert-kan7/checkpoint-46350', num_labels=7, degree=3)
tokenizer = AutoTokenizer.from_pretrained('model/bert-base-chinese', local_files_only=True)

# 准备输入文本
text = "这只是一个穷孩子做的一个吃糖的梦，请不要讥笑它吧。"
encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

# 预测
with torch.no_grad():
    outputs = model(**encoded_input)

# 从输出字典中获取 logits
logits = outputs['logits'] if 'logits' in outputs else outputs[0]

# 应用Softmax来计算概率
probs = torch.nn.functional.softmax(logits, dim=-1)

# 获取最大概率的类别和分数
max_prob, predicted_label = torch.max(probs, dim=-1)

# 使用情感类别映射将数字标签转换为具体情感类别
predicted_emotion = emotion_map[predicted_label.item()]  # 使用逆向映射

# 打印预测结果
print(f"预测的情绪: {predicted_emotion}，概率为 {max_prob.item():.4f}")