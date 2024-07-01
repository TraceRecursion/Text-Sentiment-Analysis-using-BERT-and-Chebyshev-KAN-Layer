from transformers import pipeline

# 定义情感类别映射（逆映射）
target_map = {'悲伤': 0, '快乐': 1, '厌恶': 2, '愤怒': 3, '喜欢': 4, '惊讶': 5, '恐惧': 6}
# 构建逆向映射字典
target_map_inv = {v: k for k, v in target_map.items()}

# 加载模型
model = pipeline('text-classification', model='./training_basic-bert7/checkpoint-46350')

# 预测文本情感
result = model('这只是一个穷孩子做的一个吃糖的梦，请不要讥笑它吧。')

# 输出可读结果
if result:
    label_id = result[0]['label'].replace('LABEL_', '')
    if label_id.isdigit():
        label_id = int(label_id)
        predicted_emotion = target_map_inv.get(label_id, '未知')
        score = result[0]['score']
        print(f"预测的情绪: {predicted_emotion}，概率为 {score:.4f}")
    else:
        print("无效的标签ID")
else:
    print("未作出预测。")