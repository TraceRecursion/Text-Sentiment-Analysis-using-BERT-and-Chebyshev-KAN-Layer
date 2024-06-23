import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, Trainer, TrainingArguments, BertModel, \
    BertPreTrainedModel, DataCollatorWithPadding

plt.switch_backend('agg')  # 设置matplotlib后端为agg，适用于非GUI环境


class Config:
    # 文件和模型配置
    data_file = 'test.csv'
    encoding = 'gbk'
    output_file = 'data.csv'
    output_dir = 'training_bert-kan'
    model_path = 'model/bert-base-chinese'
    local_files_only = True
    num_labels = 3

    # 数据标签映射
    target_map = {'positive': 1, 'negative': 0, 'neutral': 2}

    # 训练参数配置
    eval_strategy = 'epoch'
    save_strategy = 'epoch'
    num_train_epochs = 30
    per_device_train_batch_size = 16
    per_device_eval_batch_size = 64
    logging_dir = 'logs'
    logging_strategy = 'epoch'
    test_size = 0.3
    random_seed = 42
    degree = 3  # Chebyshev多项式的度数


class ChebyshevKANLayer(nn.Module):
    """Chebyshev多项式卷积层，用于在BERT模型上添加自定义层"""
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyshevKANLayer, self).__init__()
        self.coeffs = nn.Parameter(torch.randn(output_dim, input_dim, degree + 1))
        self.degree = degree

    def forward(self, x):
        x_norm = torch.tanh(x)  # 归一化处理
        Tx = [torch.ones_like(x_norm), x_norm]  # 初始化多项式
        for n in range(2, self.degree + 1):
            Tx.append(2 * x_norm * Tx[-1] - Tx[-2])
        T_stack = torch.stack(Tx, dim=-1).permute(0, 2, 1)
        coeffs_adjusted = self.coeffs.permute(0, 2, 1)
        logits = torch.einsum('bdi,odi->bo', T_stack, coeffs_adjusted)
        return logits


class BertWithChebyshevKAN(BertPreTrainedModel):
    """BERT模型与Chebyshev KAN层结合"""
    def __init__(self, config, degree=3):
        super(BertWithChebyshevKAN, self).__init__(config)
        self.bert = BertModel(config)
        self.kan_layer = ChebyshevKANLayer(config.hidden_size, config.num_labels, degree)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = self.kan_layer(outputs[1])
        if labels is None:
            return {'logits': logits}
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return {'loss': loss, 'logits': logits}


def load_and_prepare_data(filename, config):
    """加载并预处理数据集"""
    df = pd.read_csv(filename, encoding=config.encoding)
    df['target'] = df['sentiment'].map(config.target_map)
    prepared_df = df[['text', 'target']]
    prepared_df.columns = ['sentence', 'label']
    prepared_df.to_csv(config.output_file, index=False)
    return load_dataset('csv', data_files=config.output_file)


def custom_collate_fn(batch):
    """自定义collate函数以适配数据批处理"""
    tokenizer = AutoTokenizer.from_pretrained('model/bert-base-chinese', local_files_only=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    collated_data = data_collator([{"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]} for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    collated_data['labels'] = labels
    return collated_data


def tokenize_data(dataset, tokenizer, config):
    """使用tokenizer处理数据集"""
    def tokenize_fn(batch):
        return tokenizer(batch['sentence'], truncation=True, padding=True, max_length=512, return_tensors='pt')
    return dataset.map(tokenize_fn, batched=True)


def compute_metrics(eval_pred):
    """计算模型性能指标"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = np.mean(predictions == labels)
    f1 = f1_score(labels, predictions, average='macro')
    return {'accuracy': acc, 'f1': f1}


def plot_metrics(training_history, output_dir):
    """绘制训练过程中的性能指标"""
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
    plt.savefig(f'{output_dir}/metrics_plot.png')
    plt.close()


def main():
    """主执行函数"""
    config = Config()
    raw_datasets = load_and_prepare_data(config.data_file, config)
    train_test_split = raw_datasets['train'].train_test_split(test_size=config.test_size, seed=config.random_seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, local_files_only=config.local_files_only)
    tokenized_datasets = tokenize_data(train_test_split, tokenizer, config)

    model = BertWithChebyshevKAN.from_pretrained(
        config.model_path,
        num_labels=config.num_labels,
        local_files_only=config.local_files_only,
        degree=config.degree
    )

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        eval_strategy=config.eval_strategy,
        save_strategy=config.save_strategy,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        logging_dir=config.logging_dir,
        logging_strategy=config.logging_strategy,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=custom_collate_fn
    )

    trainer.train()
    plot_metrics(trainer.state.log_history, config.output_dir)


if __name__ == "__main__":
    main()
