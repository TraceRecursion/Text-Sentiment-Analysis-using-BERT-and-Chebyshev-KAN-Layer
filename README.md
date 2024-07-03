# Text Sentiment Analysis using BERT and Chebyshev KAN Layer

# 使用 BERT 和 Chebyshev KAN 层的文本情感分析

This project is a Natural Language Processing (NLP) course design that explores the application of BERT and Chebyshev KAN layers in text sentiment analysis. The goal is to enhance sentiment analysis performance by integrating Chebyshev KAN layers into the BERT model.
本项目是一个自然语言处理（NLP）课程设计，探讨了 BERT 和 Chebyshev KAN 层在文本情感分析中的应用。目标是通过将 Chebyshev KAN 层集成到 BERT 模型中来提升情感分析性能。

Before running this project, you need to download Google's open source BERT model and put it in the model/bert-base-chinese directory.
在运行本项目前你需要下载谷歌开源的BERT模型置于model/bert-base-chinese目录下。

[下载链接/Download link](https://huggingface.co/google-bert/bert-base-chinese)

## Table of Contents

## 目录

1. [Introduction](#introduction)
2. [介绍](#introduction)
3. [Related Work](#related-work)
4. [相关工作](#related-work)
5. [Methodology](#methodology)
   - [Model Selection](#model-selection)
   - [模型选择](#model-selection)
   - [Data Processing](#data-processing)
   - [数据处理](#data-processing)
   - [Model Architecture](#model-architecture)
   - [模型架构](#model-architecture)
   - [Training Settings](#training-settings)
   - [训练设置](#training-settings)
6. [Experiments and Results](#experiments-and-results)
   - [Experimental Setup](#experimental-setup)
   - [实验设置](#experimental-setup)
   - [Training Process](#training-process)
   - [训练过程](#training-process)
   - [Results](#results)
   - [结果](#results)
   - [Result Analysis](#result-analysis)
   - [结果分析](#result-analysis)
7. [Seven-Class Sentiment Analysis](#seven-class-sentiment-analysis)
   - [Dataset](#dataset)
   - [Model and Code](#model-and-code)
   - [Challenges and Limitations](#challenges-and-limitations)
8. [Conclusion and Future Work](#conclusion-and-future-work)
9. [结论与未来工作](#conclusion-and-future-work)
10. [References](#references)
11. [参考文献](#references)
12. [Appendix](#appendix)
    - [Training Code](#training-code)
    - [训练代码](#training-code)
    - [Testing Code](#testing-code)
    - [测试代码](#testing-code)

## Introduction

## 介绍

### Research Background

### 研究背景

Text sentiment analysis is a crucial task in NLP that involves identifying and classifying emotional tendencies within text data. With the rise of social media and online forums, vast amounts of textual data offer unique opportunities to extract user sentiment and preferences, providing significant value to businesses and government agencies.
文本情感分析是 NLP 中的一项重要任务，涉及识别和分类文本数据中的情感倾向。随着社交媒体和在线论坛的兴起，大量的文本数据为提取用户情感和偏好提供了独特的机会，为企业和政府机构带来了重要价值。

### Research Purpose

### 研究目的

This project aims to utilize the BERT model for text sentiment analysis. BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained transformer model known for its excellent performance across various NLP tasks. In this study, we introduce the Chebyshev KAN layer to further improve model performance. The Chebyshev KAN layer is a convolutional neural network layer based on Chebyshev polynomials, designed to capture complex emotional features more effectively.
本项目旨在利用 BERT 模型进行文本情感分析。BERT（双向编码器表示模型）是一种预训练的变压器模型，以其在各种 NLP 任务中的出色表现而闻名。在本研究中，我们引入了 Chebyshev KAN 层，以进一步提高模型性能。Chebyshev KAN 层是一种基于 Chebyshev 多项式的卷积神经网络层，旨在更有效地捕捉复杂的情感特征。

## Related Work

## 相关工作

### Existing Methods

### 现有方法

Traditional sentiment analysis methods relied on manual feature extraction and classical machine learning algorithms like SVM and Naive Bayes. While effective in some scenarios, these methods often struggled with complex emotional expressions due to limited contextual understanding.
传统的情感分析方法依赖于手动特征提取和经典的机器学习算法，如 SVM 和朴素贝叶斯。虽然在某些场景中有效，但由于上下文理解能力有限，这些方法往往难以处理复杂的情感表达。

With advancements in deep learning, neural network-based approaches, particularly pre-trained language models like BERT, have become dominant. These models leverage deep bidirectional processing capabilities to capture nuanced emotional changes within text.
随着深度学习的进步，基于神经网络的方法，特别是像 BERT 这样的预训练语言模型，已经成为主流。这些模型利用深度双向处理能力来捕捉文本中的细微情感变化。

### Improvement Directions

### 改进方向

Despite BERT's success, it faces challenges in handling long texts and capturing complex emotional features. To address these limitations, we introduce the Chebyshev KAN layer, enhancing feature representation through polynomial convolution operations. This integration aims to improve performance in sentiment analysis tasks by capturing intricate emotional dependencies and features.
尽管 BERT 取得了成功，但它在处理长文本和捕捉复杂情感特征方面仍面临挑战。为了解决这些局限性，我们引入了 Chebyshev KAN 层，通过多项式卷积操作增强特征表示。这一集成旨在通过捕捉复杂的情感依赖性和特征来提高情感分析任务的性能。

## Methodology

## 方法

### Model Selection

### 模型选择

We selected BERT as the baseline model due to its outstanding performance in NLP tasks and strong pre-training capabilities. To enhance the model's ability to capture complex emotional features, we incorporated the Chebyshev KAN layer.
我们选择 BERT 作为基线模型，因为它在 NLP 任务中的出色表现和强大的预训练能力。为了增强模型捕捉复杂情感特征的能力，我们加入了 Chebyshev KAN 层。

### Data Processing

### 数据处理

Our dataset, `3-data.csv`, includes extensive text data and corresponding sentiment labels. The data processing steps are:
我们的数据集 `3-data.csv` 包含大量文本数据和相应的情感标签。数据处理步骤如下：

1. **Data Cleaning**: Removing invalid characters and noise such as HTML tags and special symbols.
2. **数据清理**：去除无效字符和噪音，如 HTML 标签和特殊符号。
3. **Tokenization**: Using BERT's tokenizer to preprocess the text.
4. **标记化**：使用 BERT 的标记器预处理文本。
5. **Label Mapping**: Converting sentiment labels (positive, negative, neutral) into numerical form for model training and prediction.
6. **标签映射**：将情感标签（积极、消极、中性）转换为数字形式以进行模型训练和预测。
7. **Data Splitting**: Dividing the dataset into training (70%) and testing (30%) sets.
8. **数据划分**：将数据集分为训练集（70%）和测试集（30%）。

### Model Architecture

### 模型架构

The model architecture consists of:
模型架构包括：

- **BERT Model**: We use the `bert-base-chinese` model for its bidirectional Transformer architecture.
- **BERT 模型**：我们使用 `bert-base-chinese` 模型，因为它具有双向变压器架构。
- **Chebyshev KAN Layer**: Integrated into the BERT model to enhance feature extraction and representation.
- **Chebyshev KAN 层**：集成到 BERT 模型中，以增强特征提取和表示。

### Training Settings

### 训练设置

The training setup involves configuring hyperparameters such as learning rate, batch size, and number of epochs. We use a fine-tuning approach to optimize the pre-trained BERT model on our specific sentiment analysis task.
训练设置包括配置学习率、批处理大小和训练轮数等超参数。我们使用微调方法来优化预训练的 BERT 模型，以完成我们特定的情感分析任务。

## Experiments and Results

## 实验与结果

### Experimental Setup

### 实验设置

Details on the experimental setup, including hardware, software, and libraries used, along with the configuration of hyperparameters for training.
实验设置的详细信息，包括使用的硬件、软件和库，以及训练的超参数配置。

### Training Process

### 训练过程

Step-by-step description of the training process, including data loading, model initialization, and training iterations.
训练过程的逐步描述，包括数据加载、模型初始化和训练迭代。

### Results

### 结果

Presentation of results obtained from the trained model, including performance metrics such as accuracy, precision, recall, and F1-score.
展示训练模型获得的结果，包括准确率、精确率、召回率和 F1 值等性能指标。

### Result Analysis

### 结果分析

Detailed analysis of the results, discussing the effectiveness of the Chebyshev KAN layer integration and comparing it with the baseline BERT model.
对结果的详细分析，讨论 Chebyshev KAN 层集成的有效性，并与基线 BERT 模型进行比较。

## Seven-Class Sentiment Analysis

## 七分类情感分析

### Dataset

### 数据集

In addition to the three-class sentiment analysis, we explored a seven-class sentiment analysis using a separate dataset, `7-data.csv`, which contains texts labeled with seven distinct sentiment categories.
除了三分类情感分析之外，我们还探索了使用一个单独的数据集 `7-data.csv` 的七分类情感分析，该数据集包含标有七种不同情感类别的文本。

### Model and Code

### 模型与代码

The model architecture and training code for the seven-class sentiment analysis can be found in the `seven_class` directory of this repository. The approach is similar to the three-class analysis but adapted for multi-class classification.
七分类情感分析的模型架构和训练代码可以在本仓库的 `seven_class` 目录中找到。方法类似于三分类分析，但针对多分类进行了调整。

### Challenges and Limitations

### 挑战与局限性

Due to time constraints, the performance of the seven-class sentiment analysis was not as satisfactory as the three-class analysis. Further optimization and experimentation are required to improve the model's accuracy and robustness for this more complex task.
由于时间限制，七分类情感分析的性能不如三分类分析。需要进一步优化和实验以提高模型在这一更复杂任务中的准确性和鲁棒性。

## Conclusion and Future Work

## 结论与未来工作

### Summary

### 总结

Summarization of the project's findings and the impact of integrating Chebyshev KAN layers into the BERT model for sentiment analysis.
总结项目的发现，以及将 Chebyshev KAN 层集成到 BERT 模型中对情感分析的影响。

### Future Work

### 未来工作

Potential directions for future research, such as exploring different model architectures, datasets, or further optimizing the Chebyshev KAN layer.
未来研究的潜在方向，例如探索不同的模型架构、数据集或进一步优化 Chebyshev KAN 层。

## References

## 参考文献

List of all the references cited in the project.
项目中引用的所有参考文献列表。

## Appendix

## 附录

### Training Code

### 训练代码

- Baseline BERT model training code
- 基线 BERT 模型训练代码
- Chebyshev KAN layer model training code
- Chebyshev KAN 层模型训练代码
- Seven-class sentiment analysis training code
- 七分类情感分析训练代码

### Testing Code

### 测试代码

- Baseline BERT model testing code
- 基线 BERT 模型测试代码
- Chebyshev KAN layer model testing code
- Chebyshev KAN 层模型测试代码
- Seven-class sentiment analysis testing code
- 七分类情感分析测试代码
