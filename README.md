# abstract_tianchi
面向篇章级文本的突发事件摘要生成任务评测

比赛链接：https://github.com/qiyuanlanniao/abstract_tianchi/invitations

任务类型，给定source_text输出target_text，摘要任务，初步考虑transformer结构

# 文件说明

```
|---abstract_tianchi
    |--- figs                                           # 项目中用到的图片
    |--- data_analysis.ipynb                            # 数据集分析
    |--- data_process.py                                # 数据集处理
    |--- evaluate.py                                    # 模型评估
    |--- predict.py                                     # 输出预测文件，提交成绩
```


# 一、数据分析

## 0. 数据量

训练集合和预测集合

![训练集和预测集](figs/datanum.png)


## 1. summarization文本分析

summarization即为模型的输出

![长度分布](figs/summarizaion_length.png)

文本最短长度 5 文本最大长度 218

平均长度为34.33，在transformer生成预测文本的时候长度可以设置为200

输出文本的高频词

![top words](figs/summarization_top.png)

词云图

![词云图](figs/summarization_wordcloud.png)


# 二、finetune阶段

![只考虑对events文本转写](figs/events.png)

## 1. 只对events考虑做summarize
两个模型：BART、Nezha

1. BART中文预选训练下载源：https://huggingface.co/fnlp/bart-base-chinese/tree/main

2. Nezha中文与训练下载源：https://github.com/lonePatient/NeZha_Chinese_PyTorch?tab=readme-ov-file
   
## 2. 考虑对doc长文本做特征提取然后一起喂入Transformer-encoder


# 模型原理

# 成绩记录

# 分工记录

A ： 
|时间|记录|
|-----|----------|
|0531|数据集Summarization文本分析|
|     |          |
|     |          |
|     |          |
|     |          |

B ： 

|时间|记录|
|-----|----------|
|     |          |
|     |          |
|     |          |
|     |          |
|     |          |





# 参考文献
- [1] BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension
- [2] NEZHA: Neural Contextualized Representation for Chinese Language Understanding

