'''
模型预训练文本
'''
import json
import pandas as pd


if __name__ == '__main__':

    # train
    with open('data/train_data.json','r') as file:
        content = file.readlines()
        content = [line.strip() for line in content if line.strip()]
        data = [json.loads(line) for line in content]
    train_df = []
    for item in data:
        events = item['events']
        for idx , (event,summarization) in enumerate(zip(events)):
            train_df.append({
                'source':event['content'],
                'target':event['content'],})
    print(f'1 . train_df {len(train_df)}')    
    for item in data:
        doc = item['doc']
        docs = doc.split('\n')
        for doc in docs:
            if len(doc) >= 10:
                # 选取长度大于10的文本
                train_df.append({
                    'source':doc,
                    'target':doc})
    print(f'2 . train_df {len(train_df)}')    
            
    train_df = pd.DataFrame(train_df)
