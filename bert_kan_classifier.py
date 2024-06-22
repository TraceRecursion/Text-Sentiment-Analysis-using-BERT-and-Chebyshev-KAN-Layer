import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, Trainer, TrainingArguments, BertModel, \
    BertPreTrainedModel, DataCollatorWithPadding

# 设置 matplotlib 后端为 'agg'，使其可以在非GUI环境下运行
plt.switch_backend('agg')


class ChebyshevKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyshevKANLayer, self).__init__()
        self.coeffs = nn.Parameter(torch.randn(output_dim, input_dim, degree + 1))
        self.degree = degree

    def forward(self, x):
        x_norm = torch.tanh(x)  # [-1, 1]范围内的正则化
        # 初始的多项式
        Tx = [torch.ones_like(x_norm), x_norm]
        for n in range(2, self.degree + 1):
            Tx.append(2 * x_norm * Tx[-1] - Tx[-2])
        T_stack = torch.stack(Tx, dim=-1)  # [batch_size, input_dim, degree+1]
        T_stack = T_stack.permute(0, 2, 1)  # 调整为 [batch_size, degree+1, input_dim]

        # 调整 self.coeffs 的形状以匹配 T_stack
        coeffs_adjusted = self.coeffs.permute(0, 2, 1)  # [output_dim, degree+1, input_dim]
        logits = torch.einsum('bdi,odi->bo', T_stack, coeffs_adjusted)
        return logits


class BertWithChebyshevKAN(BertPreTrainedModel):
    """创建一个新的BERT模型类，将标准的输出层替换为KAN层"""

    def __init__(self, config, degree=3):
        super(BertWithChebyshevKAN, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.kan_layer = ChebyshevKANLayer(config.hidden_size, self.num_labels, degree)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[1]  # 使用CLS标记的输出
        logits = self.kan_layer(sequence_output)

        # 如果没有提供labels，直接返回logits
        if labels is None:
            return {'logits': logits}

        # 如果提供了labels，计算并返回损失和logits
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return {'loss': loss, 'logits': logits}


def load_and_prepare_data(filename):
    """加载和预处理数据集"""
    df = pd.read_csv(filename, encoding='gbk')
    target_map = {'positive': 1, 'negative': 0, 'neutral': 2}
    df['target'] = df['sentiment'].map(target_map)
    prepared_df = df[['text', 'target']]
    prepared_df.columns = ['sentence', 'label']
    prepared_df.to_csv('data.csv', index=False)
    return load_dataset('csv', data_files='data.csv')


def custom_collate_fn(batch):
    tokenizer = AutoTokenizer.from_pretrained('model/bert-base-chinese', local_files_only=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 此处我们假设 'batch' 是一个列表，其中包含多个字典，每个字典都有 'input_ids'、'attention_mask' 和 'label'
    collated_data = data_collator(
        [{"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]} for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    collated_data['labels'] = labels
    return collated_data


def tokenize_data(dataset, tokenizer):
    """使用tokenizer对数据集进行处理"""

    def tokenize_fn(batch):
        # 启用截断和填充确保所有序列长度一致
        return tokenizer(batch['sentence'], truncation=True, padding=True, max_length=512, return_tensors='pt')

    return dataset.map(tokenize_fn, batched=True)


def compute_metrics(eval_pred):
    """计算模型性能指标"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = np.mean(predictions == labels)
    f1 = f1_score(labels, predictions, average='macro')
    return {'accuracy': acc, 'f1': f1}


def plot_metrics(training_history):
    """绘制训练过程中的指标变化"""
    eval_epochs = [x['epoch'] for x in training_history if 'eval_loss' in x]
    eval_accuracy = [x['eval_accuracy'] for x in training_history if 'eval_accuracy' in x]
    eval_f1 = [x['eval_f1'] for x in training_history if 'eval_f1' in x]
    eval_loss = [x['eval_loss'] for x in training_history if 'eval_loss' in x]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(eval_epochs, eval_accuracy, label='Accuracy')
    plt.plot(eval_epochs, eval_f1, label='F1 Score')
    plt.title('Evaluation Metrics Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(eval_epochs, eval_loss, label='Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_bert-kan/metrics_plot.png')
    plt.close()


def main():
    raw_datasets = load_and_prepare_data('test.csv')
    train_test_split = raw_datasets['train'].train_test_split(test_size=0.3, seed=42)

    tokenizer = AutoTokenizer.from_pretrained('model/bert-base-chinese', local_files_only=True)
    tokenized_datasets = tokenize_data(train_test_split, tokenizer)

    model = BertWithChebyshevKAN.from_pretrained(
        'model/bert-base-chinese',
        num_labels=3,
        local_files_only=True,
        degree=3
    )

    training_args = TrainingArguments(
        output_dir='training_bert-kan',
        eval_strategy='epoch',
        save_strategy='epoch',
        num_train_epochs=30,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        logging_dir='logs',
        logging_strategy='epoch',
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=custom_collate_fn  # 使用自定义 collate_fn
    )

    trainer.train()
    plot_metrics(trainer.state.log_history)


if __name__ == "__main__":
    main()
