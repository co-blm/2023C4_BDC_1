# coding=utf-8
import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
import joblib

from global_v import *

n_splits = 5
num_classes=24
lb_encoder = LabelEncoder()
label = pd.read_csv("model/tmp_data/all_train_label_name.csv")
label['label'] = lb_encoder.fit_transform(label['source'])
print('labels:',label['label'].unique())

tr_df=pd.read_csv('model/tmp_data/ex_f_train_feature.csv')
new_tr_df=tr_df[tr_df['m_l']!=-10]
print('the shape of train data:',len(new_tr_df),len(new_tr_df.columns))

all_data = new_tr_df.merge(label[['id', 'label']].groupby(['id'], as_index=False)['label'].agg(list), how='left', on=['id']).set_index("id")  #左连接特征与标签
not_use = ['id', 'label']
feature_name = [i for i in all_data.columns if i not in not_use]
X = all_data[feature_name]
print(f"feature Length of train data :{len(feature_name)}")

kf = MultilabelStratifiedKFold(n_splits=n_splits, random_state=3407, shuffle=True)
scaler_X=X
y = gen_label(all_data)
train_scaler_X = scaler_X.values
train_scaler_X = train_scaler_X.astype('float')
# ovr_oof = np.zeros((len(train_scaler_X), num_classes))
# print(ovr_oof.shape,train_scaler_X.shape)
model_out=True
i=0
for train_index, valid_index in kf.split(train_scaler_X, y):
    X_train, X_valid = train_scaler_X[train_index], train_scaler_X[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]
    clf = OneVsRestClassifier(LGBMClassifier(random_state=0,
                                             n_estimators=2000,n_jobs=-1),
                                             n_jobs=-1)
    clf.fit(X_train, y_train)
    # ovr_oof[valid_index] = clf.predict_proba(X_valid)
    if model_out==True:
        joblib.dump(clf,f'model/model_data/train_model_kFold{i}.m') 
    i+=1


print('\n','******** train successfully ********','\n')