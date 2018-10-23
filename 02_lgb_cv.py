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
# scaler=MinMaxScaler()
# 设置随机种子
SEED=222
np.random.seed(SEED)
df_train,df_test=create_feature()

# 加载数据
X = df_train.drop(['UID', 'Tag'], axis=1, inplace=False)
# X=scaler.fit_transform(X)
y = df_train['Tag']


x_sub = df_test.drop(['UID', 'Tag'], axis=1, inplace=False)
# x_sub=scaler.transform(x_sub)


kf=KFold(n_splits=5, shuffle=False,random_state=42)
model=LGBMClassifier()
mean_auc=[]
for index, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_test, y_test)],
              early_stopping_rounds=30, verbose=2)

    test_pre = model.predict_proba(X_test)[:,1]
    score = roc_auc_score(y_test, test_pre)
    mean_auc.append(score)

    pred_result = model.predict_proba(x_sub)[:,1]
    if index == 0:
        df_test['Tag'] = pred_result
    else:
        df_test['Tag'] += pred_result

print(mean_auc)
df_test['Tag']=df_test['Tag']/5
df_test[['UID', 'Tag']].to_csv('result/02_lgb_cv.csv', index=False)
print("predictin done")