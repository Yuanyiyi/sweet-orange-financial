# -*- coding: utf-8 -*-
# @Time    : 2018/10/21 0:37
# @Author  : quincyqiang
# @File    : utils.py
# @Software: PyCharm

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_train_trans=pd.read_csv('input/transaction_TRAIN.csv') # (161965, 27)
df_train_op=pd.read_csv('input/operation_TRAIN.csv') # (424481, 20)
df_train_tag=pd.read_csv('input/tag_TRAIN.csv') # (13422, 2)


df_test_trans=pd.read_csv('input/transaction_round1.csv') # (161965, 27)
df_test_op=pd.read_csv('input/operation_round1.csv') # (424481, 20)
df_test_tag=pd.read_csv('input/tag_TEST_round1.csv') # (13422, 2)


def create_fea_first(df_trans,df_op,df_tag):
    """
    用户操作和交易数量
    :param df_trans:
    :param df_op:
    :param df_tag:
    :return:
    """
    print("create fea1...")
    ## 训练集
    df_trans_uids=df_trans['UID']
    df_op_uids=df_op['UID']
    df_tag_uids=df_tag['UID'] # 标签中的UID都是唯一的

    trans_cnt_dict=dict(df_trans_uids.value_counts())
    trans_cnt=[trans_cnt_dict[uid] if uid in trans_cnt_dict else 0 for uid in df_tag_uids ]
    df_tag['trans_cnt']=trans_cnt

    op_cnt_dict=dict(df_op_uids.value_counts())
    op_cnt=[op_cnt_dict[uid] if uid in op_cnt_dict else 0 for uid in df_tag_uids ]
    df_tag['op_cnt']=op_cnt

    return df_tag


def create_fea_second(df_trans,df_tag):
    """
    用户交易金额
    :param df_trans:
    :param df_op:
    :param df_tag:
    :return:
    """
    print("create fea2...")

    df_tag_uids = df_tag['UID']  # 标签中的UID都是唯一的

    ## 训练集
    trans_amt_sum= df_trans.groupby('UID').sum()['trans_amt']
    trans_amt_mean= df_trans.groupby('UID').mean()['trans_amt']

    trans_amt_sum = dict(trans_amt_sum)
    trans_amt_sum = [trans_amt_sum[uid] if uid in trans_amt_sum else 0 for uid in df_tag_uids]

    trans_amt_mean = dict(trans_amt_mean)
    trans_amt_mean = [trans_amt_mean[uid] if uid in trans_amt_mean else 0 for uid in df_tag_uids]

    df_tag['trans_amt_sum']=trans_amt_sum
    df_tag['trans_amt_mean']=trans_amt_mean

    # plt.plot(df_tag['trans_amt_sum'].values)
    # plt.plot(df_tag['trans_amt_mean'].values)
    # plt.show()

    return df_tag


def create_fea_third(df_op,df_tag):
    print("create fea3...")

    # 训练集中所有的版本号，对应成索引
    op_versions = dict(df_op['version'].value_counts()).keys()
    op_versions = list(op_versions)
    op_versions.sort()
    index = range(len(op_versions))
    ver_index = dict(zip(op_versions, index))
    df_tag_uids = df_tag['UID']  # 标签中的UID都是唯一的
    uid_op_ver = df_op[df_op['UID'].isin(df_tag_uids)]
    all_vers = []
    for uid in df_tag_uids:
        # ver_arr = np.zeros(30)
        ver_arr = []
        ver = dict(uid_op_ver[uid_op_ver['UID'] == uid]['version'].value_counts())
        for key, value in ver.items():
            index = ver_index[key]
            ver_arr.append(index)
        if not ver_arr:
            ver_arr=[30]
        all_vers.append(max(ver_arr))
    df_tag['op_version'] = all_vers

    return df_tag

def create_feature():
    # 创建特征1
    df_train=create_fea_first(df_train_trans,df_train_op,df_train_tag)
    df_test=create_fea_first(df_test_trans,df_test_op,df_test_tag)

    # 创建特征2
    df_train = create_fea_second(df_train_trans, df_train)
    df_test = create_fea_second(df_test_trans, df_test)

    # 创建特征3
    df_train = create_fea_third(df_train_op, df_train)
    df_test = create_fea_third(df_train_op, df_test)
    print(df_train['op_version'])
    return df_train,df_test


create_feature()
