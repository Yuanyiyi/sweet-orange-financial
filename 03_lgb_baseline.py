# !/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: 02_lgb_cv.py 
@Time: 2018/10/22 16:01
@Software: PyCharm 
@Description:
"""
import numpy as np
from utils import create_feature
from sklearn.model_selection import train_test_split,KFold
from lightgbm import LGBMClassifier
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import roc_auc_score
scaler=MinMaxScaler()

df_train,df_test=create_feature()
def get_train_test(test_size=0.2):
    X = df_train.drop(['UID', 'Tag'], axis=1, inplace=False)
    X=scaler.fit_transform(X)

    y = df_train['Tag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test=get_train_test()



clf=LGBMClassifier()

clf.fit(X_train, y_train,
        eval_metric=['auc'],
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=30, verbose=2)
test_pred = clf.predict_proba(X_test)
score = roc_auc_score(y_test, test_pred[:,1])
print("test ruc auc soce：",score)

# 提交结果
x_sub = df_test.drop(['UID', 'Tag'], axis=1, inplace=False)
x_sub=scaler.transform(x_sub)
sub_pred = clf.predict_proba(x_sub)
df_test['Tag'] = sub_pred[:,1]
df_test[['UID', 'Tag']].to_csv('result/03_lgb_baseline.csv', index=False)
print("predictin done")