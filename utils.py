# -*- coding: utf-8 -*-
# @Time    : 2018/10/21 0:37
# @Author  : quincyqiang
# @File    : utils.py
# @Software: PyCharm

import pandas as pd
import matplotlib.pyplot as plt
df_train_trans=pd.read_csv('input/transaction_TRAIN.csv') # (161965, 27)
df_train_op=pd.read_csv('input/operation_TRAIN.csv') # (424481, 20)
df_train_tag=pd.read_csv('input/tag_TRAIN.csv') # (13422, 2)


df_test_trans=pd.read_csv('input/transaction_round1.csv') # (161965, 27)
df_test_op=pd.read_csv('input/operation_round1.csv') # (424481, 20)
df_test_tag=pd.read_csv('input/tag_TEST_round1.csv') # (13422, 2)

def load_feature_first():
    ## 训练集
    df_trans_uids=df_train_trans['UID']
    df_op_uids=df_train_op['UID']
    df_tag_uids=df_train_tag['UID'] # 标签中的UID都是唯一的

    trans_cnt_dict=dict(df_trans_uids.value_counts())
    trans_cnt=[trans_cnt_dict[uid] if uid in trans_cnt_dict else 0 for uid in df_tag_uids ]
    df_train_tag['trans_cnt']=trans_cnt


    op_cnt_dict=dict(df_op_uids.value_counts())
    op_cnt=[op_cnt_dict[uid] if uid in op_cnt_dict else 0 for uid in df_tag_uids ]
    df_train_tag['op_cnt']=op_cnt

    # 测试集
    df_trans_uids = df_test_trans['UID']
    df_op_uids = df_test_op['UID']
    df_tag_uids = df_test_tag['UID']  # 标签中的UID都是唯一的

    trans_cnt_dict = dict(df_trans_uids.value_counts())
    trans_cnt = [trans_cnt_dict[uid] if uid in trans_cnt_dict else 0 for uid in df_tag_uids]
    df_test_tag['trans_cnt'] = trans_cnt

    op_cnt_dict = dict(df_op_uids.value_counts())
    op_cnt = [op_cnt_dict[uid] if uid in op_cnt_dict else 0 for uid in df_tag_uids]
    df_test_tag['op_cnt'] = op_cnt
    return df_train_tag,df_test_tag

load_feature_first()