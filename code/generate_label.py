# coding=utf-8
import pandas as pd

label_info=[]
for i in range(5):
    train_label=pd.read_csv(f'data/{i+1}/label_{i+1}.csv')
    label_info.append(train_label)
all_train_lable=pd.concat(label_info)
all_train_lable.to_csv('model/tmp_data/all_train_label_name.csv',index=False)

print('\n','******** generate label successfully ********','\n')