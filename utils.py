# -*- coding: utf-8 -*-
# @Time    : 2018/10/21 0:37
# @Author  : quincyqiang
# @File    : utils.py
# @Software: PyCharm

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


def create_fea_1(df_trans,df_op,df_tag):
    """
    用户操作和交易数量
    :param df_trans:
    :param df_op:
    :param df_tag:
    :return:
    """
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


def create_fea_2(df_trans,df_tag):
    """
    用户交易金额
    :param df_trans:
    :param df_op:
    :param df_tag:
    :return:
    """
    df_tag_uids = df_tag['UID']  # 标签中的UID都是唯一的
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


def create_fea_3(df_op,df_tag):
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


def create_fea_4(df_trans,df_tag):
    """
    脱敏后账户交易余额
    :param df_trans:
    :param df_tag:
    :return:
    """
    df_tag_uids = df_tag['UID']  # 标签中的UID都是唯一的
    trans_bal_sum = df_trans.groupby('UID').sum()['bal']
    trans_bal_mean = df_trans.groupby('UID').mean()['bal']

    trans_bal_sum = dict(trans_bal_sum)
    trans_bal_sum = [trans_bal_sum[uid] if uid in trans_bal_sum else 0 for uid in df_tag_uids]

    trans_bal_mean = dict(trans_bal_mean)
    trans_bal_mean = [trans_bal_mean[uid] if uid in trans_bal_mean else 0 for uid in df_tag_uids]

    df_tag['trans_bal_sum'] = trans_bal_sum
    df_tag['trans_bal_mean'] = trans_bal_mean

    return df_tag


def create_fea_5(df_trans,df_tag):
    """
    交易 ip 特征
    :param df_trans:
    :param df_tag:
    :return:
    """
    black_uids = df_train_tag[df_train_tag['Tag'] == 1]['UID']
    black_trans = df_train_trans[df_train_trans['UID'].isin(black_uids)]
    black_ips = black_trans['ip1'].value_counts().index.tolist()

    trans_ips = df_trans.groupby('UID')['ip1'].value_counts().index.tolist()

    def get_ip_label(uid):
        temp = []
        for uid_ip in trans_ips:
            if uid_ip[0] == uid:
                temp.append(uid_ip[1])
        if set(black_ips).intersection(set(temp)):
            return 1  # 黑用户
        else:  # 存在于白用户ip或者不存在ip记录
            return 0  # 白用户

    ip_labels = [get_ip_label(uid) for uid in df_tag['UID']]
    label_encoder=LabelEncoder()
    df_tag['ip1_label']=label_encoder.fit_transform(ip_labels)
    return df_tag


def create_feature(df_train_tag=df_train_tag,df_test_tag=df_test_tag):
    # 创建特征1
    # print("create fea1...")
    # df_train_tag=create_fea_1(df_train_trans,df_train_op,df_train_tag)
    # df_test_tag=create_fea_1(df_test_trans,df_test_op,df_test_tag)

    # 创建特征2
    print("create fea2 trans_amt...")
    df_train_tag = create_fea_2(df_train_trans, df_train_tag)
    df_test_tag = create_fea_2(df_test_trans, df_test_tag)

    # 创建特征3
    # print("create fea3...")
    # df_train = create_fea_third(df_train_op, df_train)
    # df_test = create_fea_third(df_train_op, df_test)

    # 创建特征4
    print("create fea4 bal...")
    df_train_tag = create_fea_4(df_train_trans, df_train_tag)
    df_test_tag = create_fea_4(df_test_trans, df_test_tag)

    # 创建特征5
    print("create fea5 ip...")
    df_train_tag = create_fea_5(df_train_trans, df_train_tag)
    df_test_tag = create_fea_5(df_test_trans, df_test_tag)

    return df_train_tag,df_test_tag



