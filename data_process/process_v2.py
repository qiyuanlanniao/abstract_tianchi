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
        for idx , event in enumerate(events):
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
    train_df = train_df[train_df['source'].apply(lambda x:len(x)!=0 )]
    train_df = train_df[train_df['target'].apply(lambda x:len(x)!=0 )]
    train_df.to_csv('data/pretrain_train.csv',index=False)

    # val


    with open('data/val_data.json','r') as file:
        content = file.readlines()
        content = [line.strip() for line in content if line.strip()]
        data = [json.loads(line) for line in content]
    val_df = []
    for item in data:
        events = item['events']
        summarizations = item['summarizations']
        for idx , (event,summarization) in enumerate(zip(events,summarizations)):
            val_df.append({
                'source':summarization['event-summarization'],
                'target':summarization['event-summarization'],
            })
    val_df = pd.DataFrame(val_df)
    val_df = val_df[val_df['source'].apply(lambda x:len(x)!=0 )]
    val_df = val_df[val_df['target'].apply(lambda x:len(x)!=0 )]
    val_df.to_csv('data/pretrain_val.csv',index=False)
    
