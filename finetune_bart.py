# 训练翻译模型

import os
import random
import torch
import torch.nn as nn
from transformers import AutoTokenizer,BartForConditionalGeneration,AutoConfig
from torch.utils.data import Dataset,DataLoader

from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')
from utils import build_optimizer
import pandas as pd

TOKENIZER =None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置参数
MAX_LENGTH=512
BATCH_SIZE=16
EPOCH = 10
LEARNING_RATE=4e-5
WEIGHT_DECAY=1e-2

def collate_fn(batchs):
    source = [batch[0] for batch in batchs]
    target = [batch[1] for batch in batchs]
    input_dict = TOKENIZER(source,padding='longest'
        ,return_tensors='pt',max_length=MAX_LENGTH,truncation=True)
    input_ids = input_dict['input_ids']
    attention_masks = input_dict['attention_mask']
    decoder_input_dict = TOKENIZER(target , padding='longest',
        return_tensors='pt',max_length=MAX_LENGTH,truncation=True)
    decoder_input_ids = decoder_input_dict['input_ids']
    decoder_attention_masks = decoder_input_dict['attention_mask']
    labels = decoder_input_ids.clone() # pad的部分为-100
    labels[decoder_attention_masks==0] = -100
    return dict(
        input_ids = input_ids,
        attention_masks = attention_masks,
        decoder_input_ids = decoder_input_ids[:,:-1],
        decoder_attention_masks = decoder_attention_masks[:,:-1],
        labels = labels[:,1:])

class MyDataset(Dataset):
    def __init__(self,datas):
        self.datas = datas
    def __len__(self):
        return len(self.datas)
    def __getitem__(self,index):
        item = self.datas.iloc[index]
        source_text = item['source']
        target_text = item['target']
        return  source_text , target_text

if __name__ == '__main__':
    EXP_NAME = 'bart_datav1'
    OUT_PATH = os.path.join('model_weights/',EXP_NAME)
    print(f'保存结果至 {OUT_PATH}')
    os.makedirs(OUT_PATH,exist_ok=True)
    # 加载数据集
    ## 版本一数据集 process_v1
    train = pd.read_csv('data/process_v1_train.csv')
    test = pd.read_csv('data/process_v1_val.csv')
    ## 版本一数据集 process_v1
    train , test = MyDataset(train) , MyDataset(test)
    trainloader = DataLoader(
        train,batch_size=BATCH_SIZE,
        shuffle=True,drop_last=True,
        pin_memory=True,
        num_workers=4,
        collate_fn=collate_fn,)
    testloader = DataLoader(
        test,batch_size=BATCH_SIZE,
        shuffle=False,drop_last=False,
        pin_memory=True,
        num_workers=4,
        collate_fn=collate_fn,)
    print(f'训练集 {len(train)} , 测试集 {len(test)}')
    # 加载模型
    model = BartForConditionalGeneration.from_pretrained('model_weights/bart_base_chinese/')
    TOKENIZER = AutoTokenizer.from_pretrained('model_weights/bart_base_chinese/')
    model.to(DEVICE)
    print('加载模型成功')
    total_steps = int(len(trainloader) * EPOCH)
    optimizer, lr_scheduler = build_optimizer(model,
        lr=LEARNING_RATE,
        warmup_ratio=0.1,
        weight_decay=WEIGHT_DECAY,
        train_steps=total_steps)
    train_loss_list , test_loss_list = [] , []
    best_loss = float('inf')
    for epoch in range(EPOCH):
        train_loss = 0
        model.train()
        progress_bar = tqdm(trainloader)
        for batch in progress_bar:
            progress_bar.set_description(f'EP {epoch} Training...')
            batch = {k:v.to(DEVICE) for k,v in batch.items()}
            output = model(
                input_ids=batch['input_ids'],
                attention_mask = batch['attention_masks'],
                decoder_input_ids = batch['decoder_input_ids'],
                decoder_attention_mask = batch['decoder_attention_masks'],
                labels = batch['labels'],)
            loss = output['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            train_loss += loss.item()
        train_loss /= len(trainloader)
        # eval
        model.eval()
        with torch.no_grad():
            test_loss = 0
            progress_bar = tqdm(testloader)
            for batch in progress_bar:
                progress_bar.set_description(f'EP {epoch} Testing...')
                batch = {k:v.to(DEVICE) for k,v in batch.items()}
                output = model(
                    input_ids=batch['input_ids'],
                    attention_mask = batch['attention_masks'],
                    decoder_input_ids = batch['decoder_input_ids'],
                    decoder_attention_mask = batch['decoder_attention_masks'],
                    labels = batch['labels'],)
                loss = output['loss']
                test_loss+=loss.item()
            test_loss /= len(testloader)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        print(f'EP {epoch} train_loss {train_loss:.6f} test_loss {test_loss:.6f}')
        torch.save(model.state_dict() , os.path.join(OUT_PATH,f'model_ep{epoch}.pth'))
        if best_loss > test_loss:
            best_loss = test_loss
            torch.save(model.state_dict() , os.path.join(OUT_PATH,'model.pth'))
    # 记录训练结果
    with open(os.path.join(OUT_PATH,'record.json'),'w') as file:
        json.dump({
            'train_loss':train_loss_list,
            'test_loss':test_loss_list,
        },file,indent=4,ensure_ascii=False)
    # 这个记录同步到github上
    with open(os.path.join('json_results/',f'{EXP_NAME}_record.json'),'w') as file:
        json.dump({
            'train_loss':train_loss_list,
            'test_loss':test_loss_list,
        },file,indent=4,ensure_ascii=False)
