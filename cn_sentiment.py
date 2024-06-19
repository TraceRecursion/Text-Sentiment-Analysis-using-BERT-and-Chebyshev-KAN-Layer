import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

plt.switch_backend('agg')
from sklearn.metrics import f1_score

# 读取文件
df = pd.read_csv('test.csv', encoding='gbk')

# 定义转换字典
target_map = {
    'positive': 1,
    'negative': 0,
    'neutral': 2
}

# 将文件中的对应单词转换为数字 单独列出一列
df['target'] = df['sentiment'].map(target_map)

# 将文本和标签提取出来  这里导出为新的csv文件 方便后续load_data
df2 = df[['text', 'target']]
df2.columns = ['sentence', 'label']
df2.to_csv('data.csv', index=None)

raw_datasets = load_dataset('csv', data_files='data.csv')  # 加载训练的数据格式

split = raw_datasets['train'].train_test_split(test_size=0.3, seed=42)  # 分隔为数据集和测试集 测试集占百分之30

tokenizer = AutoTokenizer.from_pretrained('model/bert-base-chinese', local_files_only=True)


def tokenize_fn(batch):
    return tokenizer(batch['sentence'], truncation=True)  # 将数据集中的句子使用标记器进行标记化处理，以便后续在自然语言处理任务中使用


tokenized_datasets = split.map(tokenize_fn, batched=True)

model = AutoModelForSequenceClassification.from_pretrained('model/bert-base-chinese', num_labels=3,
                                                           local_files_only=True)

training_args = TrainingArguments(
    output_dir='training_dir',  # 输出文件夹
    eval_strategy='epoch',  # 使用新的参数名称
    save_strategy='epoch',
    num_train_epochs=3,  # 训练轮次
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    logging_dir='logs',  # 定义日志目录
    logging_strategy='epoch'  # 将日志设置为每个epoch打印一次
)


def compute_metrics(logits_and_labels):
    logits, labels = logits_and_labels
    predictions = np.argmax(logits, axis=-1)
    acc = np.mean(predictions == labels)
    f1 = f1_score(labels, predictions, average='macro')
    return {'accuracy': acc, 'f1': f1}


trainer = Trainer(
    model=model,  # 模型实例，用于训练
    args=training_args,  # 训练参数，包括学习率、批大小、训练轮数等
    train_dataset=tokenized_datasets["train"],  # 训练数据集
    eval_dataset=tokenized_datasets["test"],  # 验证数据集
    tokenizer=tokenizer,  # 分词器，用于对输入进行分词
    compute_metrics=compute_metrics  # 用于计算性能指标的函数
)

trainer.train()
