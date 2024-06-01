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
from torch.utils.data import Dataset

TOKENIZER =None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 设置参数


def collate_fn(source):
    input_dict = TOKENIZER(source,padding='longest'
        ,return_tensors='pt',max_length=MAX_LENGTH,truncation=True)
    input_ids = input_dict['input_ids']
    attention_masks = input_dict['attention_mask']
    return dict(
        input_ids = input_ids,
        attention_masks = attention_masks,)


if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir",'-cd',required=True,type=str,help='chkpt路径')
    parser.add_argument("--epoch",default='none',type=str)
    args = parser.parse_args()
    # 检查内容
    assert os.path.exists(args.checkpoint_dir) , f'not exists for {args.checkpoint_dir}'
    check_point_path = os.path.join(args.checkpoint_dir ,f'model_ep{args.epoch}.pth')
    if not os.path.exists(check_point_path):
        check_point_path = os.path.join(args.checkpoint_dir ,f'model.pth')

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

    print(f'使用权重 {check_point_path}')
    model_weight = torch.load(check_point_path,map_location='cpu')
    model.load_state_dict(model_weight)
    model.to(DEVICE)
    model.eval()

    # 输出文件路径
    out_file = open(os.path.join(args.checkpoint_dir,'submit.txt'),'w',encoding='utf-8')
    
    with open('data/testa.json','r') as file:
        content = file.readlines()
        content = [line.strip() for line in content if line.strip()]
        test = [json.loads(line) for line in content]
    with torch.no_grad():
        for item in tqdm(test):
            events = item['events']
            
            event_ids = [event['id'] for event in events]
            event_content = [event['content'] for event in events]
            input_dict = collate_fn(event_content)
            input_dict = {k:item.to(DEVICE) for k,item in input_dict.items()}

            item_json = []
            sents = model.generate(
                input_dict['input_ids'],
                attention_mask=input_dict['attention_masks'],
                num_beams = 1,max_length = MAX_LENGTH)
            sents = TOKENIZER.batch_decode(sents, skip_special_tokens=True)
            for event_id , sent in zip(event_ids,sents):
                item_json.append({
                    'id':event_id,
                    'event-summarization':''.join(sent.split(' '))
                })

            # for event in events:
            #     input_dict = TOKENIZER(event['content'],padding='longest',return_tensors='pt',max_length=MAX_LENGTH,truncation=True)
            #     input_ids = input_dict['input_ids']
            #     input_ids = input_ids.to(DEVICE).unsqueeze(0)
            #     result = model.generate(
            #         input_ids,num_beams = 1,max_length = MAX_LENGTH,)
            #     pred_sent = TOKENIZER.batch_decode(result, skip_special_tokens=True)
            #     pred_sent = pred_sent[0]
            #     pred_sent = ''.join(pred_sent.split(' '))
                
            #     item_json.append({
            #         'id':event['id'],
            #         'event-summarization':pred_sent,})

            # 将JSON数组转换为字符串
            item_json = {'summarizations':item_json}
            json_str = json.dumps(item_json, ensure_ascii=False)
            out_file.write(json_str + '\n')

    out_file.close()
        

