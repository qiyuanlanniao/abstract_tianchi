# 对数据集进行划分，避免交叉
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    with open('data/train.json','r') as file:
        lines = [line.strip() for line in file.readlines()]
    train,val = train_test_split(lines,test_size=0.2,random_state=0)
    with open('data/train_data.json','w') as file:
        file.write('\n'.join(train))
    with open('data/val_data.json','w') as file:
        file.write('\n'.join(val))



