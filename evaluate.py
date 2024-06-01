# 计算模型BLEU、CIDER-D值
import os
import random
import torch
import torch.nn as nn
from transformers import AutoTokenizer,BartForConditionalGeneration
from transformers import BertTokenizer,PegasusForConditionalGeneration
from torch.utils.data import Dataset,DataLoader
import pandas as pd
from rouge_score import rouge_scorer


from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
import torch
from copy import deepcopy
import argparse

TOKENIZER =None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 设置参数

BATCH_SIZE=4

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
    def __len__(self):
        return len(self.datas)
    def __getitem__(self,index):
        item = self.datas.iloc[index]
        source_text = item['source']
        target_text = item['target']
        return  source_text , target_text
    

@torch.no_grad()
def evaluate(model,testloader,epoch,checkpoint_dir ):
    model_weight = torch.load(os.path.join(checkpoint_dir,f'model_ep{epoch}.pth'),map_location='cpu')
    model.load_state_dict(model_weight)
    model.to(DEVICE)
    with torch.no_grad():
        progress_bar = tqdm(testloader)
        refs = []
        preds = []
        for batch in progress_bar:
            progress_bar.set_description(f'Ep {epoch}  Predicting...')
            batch = {k:v.to(DEVICE) for k,v in batch.items()}
            result = model.generate(
                batch['input_ids'],
                num_beams = 3,
                max_length = MAX_LENGTH,)
            bot_msg = TOKENIZER.batch_decode(result, skip_special_tokens=True)
            labels = TOKENIZER.batch_decode(batch['decoder_input_ids'], skip_special_tokens=True)
            preds.extend(bot_msg)
            refs.extend(labels)
            
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
    # Compute ROUGE
    rouge = Rouge()
    rouge_score, _ = rouge.compute_score(input_refs, input_preds)     

    return score , cider_score.item() ,rouge_score.item()

if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir",'-cd',required=True,type=str,help='chkpt路径')
    parser.add_argument("--epoch",default=8,type=int)
    args = parser.parse_args()
    # 检查内容
    assert os.path.exists(args.checkpoint_dir) , f'not exists for {args.checkpoint_dir}'
    assert len(os.listdir(args.checkpoint_dir)) >= args.epoch+2 , f'number of data file less than {args.epoch}+2'
     # 加载模型
    if  'bart' in args.checkpoint_dir:
        print(f'评估bart模型: {args.checkpoint_dir} epoch {args.epoch} ')
        TOKENIZER = AutoTokenizer.from_pretrained('model_weights/bart_base_chinese/')
        model = BartForConditionalGeneration.from_pretrained('model_weights/bart_base_chinese/')
        MAX_LENGTH = 512
    else:
        print(f'评估Pegasus模型 : {args.checkpoint_dir} epoch {args.epoch} ')
        TOKENIZER = BertTokenizer.from_pretrained("model_weights/pegasus_base_chinese/")
        model = PegasusForConditionalGeneration.from_pretrained('model_weights/pegasus_base_chinese/')
        MAX_LENGTH = 512


    
    model.to(DEVICE)

    test = pd.read_csv('data/process_v1_val.csv')
    test =  MyDataset(test)
    
    testloader = DataLoader(
        test,batch_size=BATCH_SIZE,
        shuffle=False,drop_last=False,
        pin_memory=True,
        num_workers=4,
        collate_fn=collate_fn,)

    print(f'测试集 {len(test)}')

    bleus = []
    ciders = []
    rouges = []
    for i in range(args.epoch):
        bleu,cider,rouge =evaluate(deepcopy(model),testloader,i,args.checkpoint_dir)
        print(f'bleu {bleu} cider {cider} rouge {rouge}')
        bleus.append(bleu)
        ciders.append(cider)

        rouges.append(rouge)
    with open(os.path.join(args.checkpoint_dir,'eval.json'),'w') as file:
        json.dump({
            'bleu':bleus,'cider':ciders,'rouges':rouges,
        },file,indent=4,ensure_ascii=False)




    
    
