# -*- coding: utf-8 -*-
# @Time    : 2018/10/21 22:19
# @Author  : quincyqiang
# @File    : 01_sklearn_baseline.py
# @Software: PyCharm

import pandas as pd
import numpy as np
from utils import load_feature_first
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# 设置随机种子
SEED=222
np.random.seed(SEED)

df_train,df_test=load_feature_first()


def get_train_test(test_size=0.2):
    X = df_train.drop(['UID', 'Tag'], axis=1, inplace=False)
    y = df_train['Tag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test=get_train_test()


def get_models():
    """Generate a library of base learners."""

    # lr = LogisticRegression(C=1.0,max_iter=100,random_state=10)
    rf = RandomForestClassifier(random_state=SEED)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=SEED)
    ab = AdaBoostClassifier(random_state=SEED)
    xgb = XGBClassifier()
    lgb = LGBMClassifier()

    models = {
        # 'logistic': lr,
              'random forest': rf,
              'gbm': gb,
              'ab': ab,
              'xgb': xgb,
              'lgb': lgb
              }

    return models


def train_predict(model_list):
    """Fit models in list on training set and return preds"""
    P = np.zeros((y_test.shape[0], len(model_list)))
    x_sub = df_test.drop(['UID', 'Tag'], axis=1, inplace=False)
    P_sub = np.zeros((x_sub.shape[0], len(model_list)))
    P = pd.DataFrame(P)
    P_sub = pd.DataFrame(P_sub)
    print("训练各个模型")
    cols = list()
    for i, (name, m) in enumerate(models.items()):
        print("%s..." % name, end=" ", flush=False)
        m.fit(X_train, y_train)
        P.iloc[:, i] = m.predict_proba(X_test)[:, 1]
        P_sub.iloc[:, i] = m.predict_proba(x_sub)[:, 1]
        cols.append(name)
        print("done")
    P.columns = cols
    P_sub.columns = cols
    print("Done.\n")
    return P,P_sub


def score_models(P, y):
    """Score model in prediction DF"""
    print("评价每个模型.")
    for m in P.columns:
        score = roc_auc_score(y, P.loc[:, m])
        print("%-26s: %.3f" % (m, score))
    print("Done.\n")


def predict(P_sub):
    df_test['Tag'] = P_sub.mean(axis=1)
    df_test[['UID', 'Tag']].to_csv('result/01_sklearn_baseline.csv', index=False)
    print("predictin done")


models = get_models()
P,P_sub = train_predict(models)
score_models(P, y_test)
print("Ensemble ROC-AUC score: %.3f" % roc_auc_score(y_test, P.mean(axis=1)))
predict(P_sub)
