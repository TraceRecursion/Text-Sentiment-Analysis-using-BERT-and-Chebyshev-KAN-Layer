import torch
from transformers import AutoTokenizer
from bert_kan_3classifier import BertWithChebyshevKAN

# 加载模型和标记器
model = BertWithChebyshevKAN.from_pretrained('./training_bert-kan3/checkpoint-6690', num_labels=3, degree=3)
tokenizer = AutoTokenizer.from_pretrained('model/bert-base-chinese', local_files_only=True)

# 准备输入文本
text = "把海弄干的鱼在海干前上了陆地，从一片黑暗森林奔向另一片黑暗森林。"
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

# 创建和pipeline相同的输出格式
results = [{'label': f'LABEL_{predicted_label.item()}', 'score': max_prob.item()}]

print(results)
