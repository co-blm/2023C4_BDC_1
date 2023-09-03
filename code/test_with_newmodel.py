# coding=utf-8
import pandas as pd
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

from global_v import *

lb_encoder = LabelEncoder()
label = pd.read_csv("model/tmp_data/all_train_label_name.csv")
label['label'] = lb_encoder.fit_transform(label['source'])

n_splits=5

new_te_df=pd.read_csv('model/tmp_data/ex_f_test_feature.csv')
print('test feature shape(with id):',len(new_te_df),len(new_te_df.columns))

def test_single_label(test_feature,model_path):
    test_feature.set_index("id",drop=False,inplace=True)
    not_use=['id']
    feature_name = [i for i in test_feature.columns if i not in not_use]
    print(f"test feature length = {len(feature_name)}")
    test_scaler_X = test_feature[feature_name].values
    ovr_preds = np.zeros((len(test_scaler_X), num_classes))
    # print(ovr_preds.shape,test_scaler_X.shape)
    for i in range(n_splits):
        clf=joblib.load(f'{model_path}/train_model_kFold{i}.m')
        ovr_preds += clf.predict_proba(test_scaler_X)/n_splits

    submit = pd.DataFrame(ovr_preds, columns=lb_encoder.classes_)
    submit.index = test_feature.index
    submit.reset_index(inplace=True)
    submit = submit.melt(id_vars="id", value_vars=lb_encoder.classes_, value_name="score", var_name="source")
    submit.to_csv("result/new_result.csv", index=False)
    return None

test_single_label(new_te_df,'model/model_data')

print('\n','******** test with newmodel successfully ********','\n')
