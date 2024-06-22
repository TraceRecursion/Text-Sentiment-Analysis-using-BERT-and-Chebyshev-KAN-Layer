import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# 设置 matplotlib 后端
plt.switch_backend('agg')


def load_and_prepare_data(filename):
    """加载和预处理数据集"""
    df = pd.read_csv(filename, encoding='gbk')
    target_map = {'positive': 1, 'negative': 0, 'neutral': 2}
    df['target'] = df['sentiment'].map(target_map)
    prepared_df = df[['text', 'target']]
    prepared_df.columns = ['sentence', 'label']
    prepared_df.to_csv('data.csv', index=False)
    return load_dataset('csv', data_files='data.csv')


def tokenize_data(dataset, tokenizer):
    """使用tokenizer对数据集进行处理"""

    def tokenize_fn(batch):
        # 添加max_length参数并确保启用截断
        return tokenizer(batch['sentence'], truncation=True, max_length=512)  # 这里假设最大长度设置为512

    return dataset.map(tokenize_fn, batched=True)


def compute_metrics(eval_pred):
    """计算模型性能指标"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = np.mean(predictions == labels)
    f1 = f1_score(labels, predictions, average='macro')
    return {'accuracy': acc, 'f1': f1}


def main():
    raw_datasets = load_and_prepare_data('test.csv')
    train_test_split = raw_datasets['train'].train_test_split(test_size=0.3, seed=42)

    tokenizer = AutoTokenizer.from_pretrained('model/bert-base-chinese', local_files_only=True)
    tokenized_datasets = tokenize_data(train_test_split, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        'model/bert-base-chinese', num_labels=3, local_files_only=True)

    training_args = TrainingArguments(
        output_dir='training_dir',
        eval_strategy='epoch',
        save_strategy='epoch',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        logging_dir='logs',
        logging_strategy='epoch'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()


if __name__ == "__main__":
    main()
