'''
模型输入为event文本，输出为summarzation文本
'''

import json
import pandas as pd
from sklearn.model_selection import train_test_split




if __name__ == '__main__':
    with open('data/train.json','r') as file:
        content = file.readlines()
        content = [line.strip() for line in content if line.strip()]
        data = [json.loads(line) for line in content]
    out_df = []
    for item in data:
        
        events = item['events']
        summarizations = item['summarizations']
        assert len(events) == len(summarizations)

        for idx , (event,summarization) in enumerate(zip(events,summarizations)):
            out_df.append({
                'source':event['content'],
                'target':summarization['event-summarization'],
            })
    out_df = pd.DataFrame(out_df)

    train,val = train_test_split(out_df,test_size=0.2,random_state=0)

    print(f'训练集合 {train.shape}')
    print(f'验证集合 {val.shape}')
    # 清洗空文本
    train = train[train['source'].apply(lambda x:len(x)!=0 )]
    train = train[train['target'].apply(lambda x:len(x)!=0 )]
    val = val[val['source'].apply(lambda x:len(x)!=0 )]
    val = val[val['target'].apply(lambda x:len(x)!=0 )]
    print(f'清洗后 训练集合 {train.shape}')
    print(f'清洗后 验证集合 {val.shape}')

    train.to_csv('data/process_v1_train.csv',index=False)
    val.to_csv('data/process_v1_val.csv',index=False)

