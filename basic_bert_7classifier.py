import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from datasets import load_dataset
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

plt.switch_backend('agg')  # 设置matplotlib后端为agg，适用于非GUI环境


class Config:
    data_file = '7-data.csv'
    encoding = 'utf-8'
    output_file = 'processed_7-data.csv'
    output_dir = 'training_ocemotion-bert'
    model_path = 'model/bert-base-chinese'
    local_files_only = True
    num_labels = 7

    # 训练参数配置
    eval_strategy = 'epoch'
    save_strategy = 'epoch'
    num_train_epochs = 30
    per_device_train_batch_size = 16
    per_device_eval_batch_size = 64
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs/{current_time}"
    logging_strategy = 'epoch'
    test_size = 0.3
    random_seed = 42


def load_and_prepare_data(filename):
    """加载并预处理数据"""
    df = pd.read_csv(filename, encoding=Config.encoding, delimiter='\t', header=None, names=['id', 'sentence', 'label_str'])
    target_map = {
        'sadness': 0, 'happiness': 1, 'disgust': 2, 'anger': 3,
        'like': 4, 'surprise': 5, 'fear': 6
    }
    df['label'] = df['label_str'].map(target_map)
    prepared_df = df[['sentence', 'label']]
    prepared_df.to_csv(Config.output_file, index=False)
    return load_dataset('csv', data_files=Config.output_file)


def tokenize_data(dataset, tokenizer):
    """使用tokenizer对数据集进行处理"""
    def tokenize_fn(batch):
        return tokenizer(batch['sentence'], truncation=True, max_length=512)
    return dataset.map(tokenize_fn, batched=True)


def compute_metrics(eval_pred):
    """计算模型性能指标"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = np.mean(predictions == labels)
    f1 = f1_score(labels, predictions, average='macro')
    return {'accuracy': acc, 'f1': f1}


def plot_metrics(training_history):
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
    plt.savefig(f'{Config.output_dir}/metrics_plot.png')
    plt.close()


def main():
    """主执行函数"""
    config = Config()
    raw_datasets = load_and_prepare_data(config.data_file)
    train_test_split = raw_datasets['train'].train_test_split(test_size=config.test_size, seed=config.random_seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, local_files_only=config.local_files_only)
    tokenized_datasets = tokenize_data(train_test_split, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_path, num_labels=config.num_labels, local_files_only=config.local_files_only)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        eval_strategy=config.eval_strategy,
        save_strategy=config.save_strategy,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        logging_dir=config.log_dir,
        logging_strategy=config.logging_strategy,
        load_best_model_at_end=True
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
    plot_metrics(trainer.state.log_history)


if __name__ == "__main__":
    main()
