import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

plt.switch_backend('agg')


class Config:
    # 数据和文件配置
    data_file = 'test.csv'
    encoding = 'gbk'
    output_file = 'data.csv'
    output_dir = 'training_basic-bert'
    model_path = 'model/bert-base-chinese'
    local_files_only = True
    num_labels = 3

    # 训练配置
    eval_strategy = 'epoch'
    save_strategy = 'epoch'
    num_train_epochs = 30
    per_device_train_batch_size = 16
    per_device_eval_batch_size = 64
    logging_dir = 'logs'
    logging_strategy = 'epoch'
    test_size = 0.3
    seed = 42


def load_and_prepare_data(filename):
    df = pd.read_csv(filename, encoding=Config.encoding)
    target_map = {'positive': 1, 'negative': 0, 'neutral': 2}
    df['target'] = df['sentiment'].map(target_map)
    prepared_df = df[['text', 'target']]
    prepared_df.columns = ['sentence', 'label']
    prepared_df.to_csv(Config.output_file, index=False)
    return load_dataset('csv', data_files=Config.output_file)


def tokenize_data(dataset, tokenizer):
    def tokenize_fn(batch):
        return tokenizer(batch['sentence'], truncation=True, max_length=512)
    return dataset.map(tokenize_fn, batched=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = np.mean(predictions == labels)
    f1 = f1_score(labels, predictions, average='macro')
    return {'accuracy': acc, 'f1': f1}


def plot_metrics(training_history):
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
    config = Config()
    raw_datasets = load_and_prepare_data(config.data_file)
    train_test_split = raw_datasets['train'].train_test_split(test_size=config.test_size, seed=config.seed)

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
        compute_metrics=compute_metrics
    )

    trainer.train()
    plot_metrics(trainer.state.log_history)


if __name__ == "__main__":
    main()
