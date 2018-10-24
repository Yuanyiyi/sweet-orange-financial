# !/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: demo.py 
@Time: 2018/10/24 13:26
@Software: PyCharm 
@Description:
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

df_train_trans=pd.read_csv('input/transaction_TRAIN.csv') # (161965, 27)
df_train_op=pd.read_csv('input/operation_TRAIN.csv') # (424481, 20)
df_train_tag=pd.read_csv('input/tag_TRAIN.csv') # (13422, 2)

df_test_trans=pd.read_csv('input/transaction_round1.csv') # (161965, 27)
df_test_op=pd.read_csv('input/operation_round1.csv') # (424481, 20)
df_test_tag=pd.read_csv('input/tag_TEST_round1.csv') # (13422, 2)


train_trans_uids=df_train_trans['UID']
train_op_uids=df_train_op['UID']
train_tag_uids=df_train_tag['UID'] # 标签中的UID都是唯一的
print(train_trans_uids.drop_duplicates().shape,train_op_uids.drop_duplicates().shape,train_tag_uids.shape)


test_trans_uids=df_test_trans['UID']
test_op_uids=df_test_op['UID']
test_tag_uids=df_test_tag['UID'] # 标签中的UID都是唯一的
print(test_trans_uids.drop_duplicates().shape,
      test_op_uids.drop_duplicates().shape,
      test_tag_uids.shape)
