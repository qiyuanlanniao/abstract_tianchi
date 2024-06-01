# 只计算Events文本
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer,BartForConditionalGeneration
from transformers import BertTokenizer,PegasusForConditionalGeneration
import json
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')
import torch
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
        return  source_text 
    
@torch.no_grad()
def predict(model,testloader):
    with torch.no_grad():
        progress_bar = tqdm(testloader)
        preds = []
        for batch in progress_bar:
            progress_bar.set_description(f'Predicting...')
            batch = {k:v.to(DEVICE) for k,v in batch.items()}
            result = model.generate(
                batch['input_ids'],
                num_beams = 3,
                max_length = MAX_LENGTH,)
            bot_msg = TOKENIZER.batch_decode(result, skip_special_tokens=True)
            preds.extend(bot_msg)
    return preds

if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir",'-cd',required=True,type=str,help='chkpt路径')
    parser.add_argument("--epoch",default=8,type=int)
    args = parser.parse_args()
    # 检查内容
    assert os.path.exists(args.checkpoint_dir) , f'not exists for {args.checkpoint_dir}'
    check_point_path = os.path.join(args.checkpoint_dir ,f'model_ep{args.epoch}.pth')
    assert os.path.exists(check_point_path) , f'not exists for {check_point_path}'
     # 加载模型
    if  'bart' in args.checkpoint_dir:
        print(f'评估bart模型: {args.checkpoint_dir} epoch {args.epoch}')
        TOKENIZER = AutoTokenizer.from_pretrained('model_weights/bart_base_chinese/')
        model = BartForConditionalGeneration.from_pretrained('model_weights/bart_base_chinese/')
        MAX_LENGTH = 512
    else:
        print(f'评估Pegasus模型 : {args.checkpoint_dir} epoch {args.epoch}' )
        TOKENIZER = BertTokenizer.from_pretrained("model_weights/pegasus_base_chinese/")
        model = PegasusForConditionalGeneration.from_pretrained('model_weights/pegasus_base_chinese/')
        MAX_LENGTH = 512
    model.to(DEVICE)
    print(f'使用权重 {check_point_path}')
    model_weight = torch.load(check_point_path,map_location='cpu')
    model.load_state_dict(model_weight)
    model.to(DEVICE)

    out_file = open(os.path.join(args.checkpoint_dir,'submit.txt'),'w',encoding='utf-8')
    

    with open('data/testa.json','r') as file:
        content = file.readlines()
        content = [line.strip() for line in content if line.strip()]
        test = [json.loads(line) for line in content]
    for item in tqdm(test):
        events = item['events']
        item_json = []
        for event in events:
            input_dict = TOKENIZER(event['content'],padding='longest',return_tensors='pt',max_length=MAX_LENGTH,truncation=True)
            input_ids = input_dict['input_ids']
            input_ids = input_ids.to(DEVICE).unsqueeze(0)
            result = model.generate(
                input_ids,num_beams = 3,max_length = MAX_LENGTH,)
            pred_sent = TOKENIZER.batch_decode(result, skip_special_tokens=True)
            pred_sent = pred_sent[0]
            item_json.append({
                'id':event['id'],
                'event-summarization':pred_sent,})
        # 将JSON数组转换为字符串
        item_json = {'summarizations':item_json}
        json_str = json.dumps(item_json, ensure_ascii=False)
        out_file.write(json_str + '\n')

    out_file.close()
        

            







