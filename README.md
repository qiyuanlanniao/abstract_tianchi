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
```


# 一、数据分析

# 二、finetune阶段

## 1. 只对events考虑做summarize
两个模型：BART、Nezha

1. BART中文预选训练下载源：https://huggingface.co/fnlp/bart-base-chinese/tree/main

2. Nezha中文与训练下载源：https://github.com/lonePatient/NeZha_Chinese_PyTorch?tab=readme-ov-file
   
## 2. 考虑对doc长文本做特征提取然后一起喂入Transformer-encoder


# 模型原理

# 成绩记录


# 参考文献
- [1] BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension
- [2] NEZHA: Neural Contextualized Representation for Chinese Language Understanding

