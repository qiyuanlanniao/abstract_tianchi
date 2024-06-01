'''
模型输入为event文本，输出为summarzation文本
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
        summarizations = item['summarizations']
        assert len(events) == len(summarizations)
        for idx , (event,summarization) in enumerate(zip(events,summarizations)):
            train_df.append({
                'source':event['content'],
                'target':summarization['event-summarization'],
            })
    train_df = pd.DataFrame(train_df)

    # val
    with open('data/val_data.json','r') as file:
        content = file.readlines()
        content = [line.strip() for line in content if line.strip()]
        data = [json.loads(line) for line in content]
    val_df = []
    for item in data:
        events = item['events']
        summarizations = item['summarizations']
        assert len(events) == len(summarizations)

        for idx , (event,summarization) in enumerate(zip(events,summarizations)):
            val_df.append({
                'source':event['content'],
                'target':summarization['event-summarization'],
            })
    val_df = pd.DataFrame(val_df)

    print(f'训练集合 {train_df.shape}')
    print(f'验证集合 {val_df.shape}')
    # 清洗空文本
    train_df = train_df[train_df['source'].apply(lambda x:len(x)!=0 )]
    train_df = train_df[train_df['target'].apply(lambda x:len(x)!=0 )]
    val_df = val_df[val_df['source'].apply(lambda x:len(x)!=0 )]
    val_df = val_df[val_df['target'].apply(lambda x:len(x)!=0 )]
    print(f'清洗后 训练集合 {train_df.shape}')
    print(f'清洗后 验证集合 {val_df.shape}')

    train_df.to_csv('data/process_v1_train.csv',index=False)
    val_df.to_csv('data/process_v1_val.csv',index=False)
