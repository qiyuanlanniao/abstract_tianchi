# 计算模型BLEU、CIDER-D值
import os
import random
import torch
import torch.nn as nn
from transformers import AutoTokenizer,BartForConditionalGeneration
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
import torch
from copy import deepcopy

TOKENIZER =None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 设置参数
MAX_LENGTH=384
BATCH_SIZE=32

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
        labels = labels[:,1:]
    )

class MyDataset(Dataset):
    def __init__(self,datas):
        self.datas = datas
        random.shuffle(self.datas)
    def __len__(self):
        return len(self.datas)
    def __getitem__(self,index):
        item = self.datas[index]
        source_text = item['input']
        target_text = f'{TOKENIZER.bos_token}{item["output"]}'
        return  source_text , target_text

@torch.no_grad()
def evaluate(model,testloader,epoch):
    model_weight = torch.load(f'results/model_ep{epoch}.pth',map_location='cpu')
    model.load_state_dict(model_weight)
    model.eval()

    model.to(DEVICE)
    with torch.no_grad():
        progress_bar = tqdm(testloader)
        # source_text , target_text
        refs = []
        preds = []
        loss_list = []
        for batch in progress_bar:

            progress_bar.set_description(f'Ep {epoch} Predicting...')
            batch = {k:v.to(DEVICE) for k,v in batch.items()}
            result = model.generate(
                batch['input_ids'],
                num_beams = 3,
                max_length = 128,
                early_stopping=True,
                decoder_start_token_id = TOKENIZER.bos_token_id,
            )
            bot_msg = TOKENIZER.batch_decode(result, skip_special_tokens=True)
            labels = TOKENIZER.batch_decode(batch['decoder_input_ids'], skip_special_tokens=True)
            preds.extend(bot_msg)
            refs.extend(labels)

            output = model(input_ids=batch['input_ids'],
                    attention_mask = batch['attention_masks'],
                    decoder_input_ids = batch['decoder_input_ids'],
                    decoder_attention_mask = batch['decoder_attention_masks'],
                    labels = batch['labels'])
            loss = output['loss']
            loss_list.append(loss.cpu().detach())

    # compute cider and bleu
    input_preds = {}
    input_refs = {}
    for idx , (ref , pred) in enumerate(zip(refs , preds)):
        input_preds[idx] = [pred]
        input_refs[idx] = [ref]
    bleu = Bleu()
    score, _ = bleu.compute_score(input_refs,input_preds,verbose=-1)
    score = sum(score) / len(score)
    cider = Cider()
    cider_score, _ = cider.compute_score(input_refs, input_preds)
    # compute ppl loss
    ppl = torch.exp(torch.stack(loss_list).mean())

    return score , cider_score.item() , ppl.item()

if __name__ == '__main__':

    test_json = json.load(open('data/test.json'))
    test = MyDataset(test_json)
    testloader = DataLoader(
        test,batch_size=BATCH_SIZE,
        shuffle=False,drop_last=False,
        pin_memory=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    print(f'测试集 {len(test)}')

    # 加载模型

    model = BartForConditionalGeneration.from_pretrained('bart-cn/')
    TOKENIZER = AutoTokenizer.from_pretrained('bart-cn/')

    bleus = []
    ciders = []
    ppls = []
    for i in range(12):
        bleu,cider , ppl =evaluate(deepcopy(model),testloader,i)
        bleus.append(bleu)
        ciders.append(cider)
        ppls.append(ppl)
    with open(f'results/eval.json','w') as file:
        json.dump({
            'bleu':bleus,'cider':ciders,'ppls':ppls,
        },file,indent=4,ensure_ascii=False)




    
    
