# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:42:48 2019

@author: yangy

LeaderRank of weighed directed networks
IDC information diffusion network
"""
import numpy as np
import pandas as pd

def gen_matrix_a(table, nodes_num, topic):
    matrix = np.zeros((nodes_num) * (nodes_num)).reshape(nodes_num, nodes_num)
    for i in table.index:
        source = int(table.loc[i].Source)
        target = int(table.loc[i].Target)
        weight = table.loc[i][topic]
        ''' target --> source'''
        #        matrix[target,source] = 1
        matrix[target, source] = weight
    w_gi = []
    for i in range(nodes_num):
        #        i = 641
        #        i = 0
        wi_in = np.sum(table[table.Source == i][topic])
        w_gi.append(wi_in)
    matrix = np.insert(matrix, 0, values=w_gi, axis=0)
    matrix = np.insert(matrix, 0, values=np.ones(nodes_num + 1), axis=1)
    matrix[0, 0] = 0

    # matrix is normalized for iteration
    a = np.zeros((nodes_num + 1) * (nodes_num + 1)).reshape(nodes_num + 1, nodes_num + 1)
    a = matrix/matrix.sum(axis=1)
    return matrix, a


def leaderrank(a, nodes_num, cycle_num):
    lr = np.zeros((nodes_num + 1) * cycle_num).reshape((nodes_num + 1), cycle_num)
    lr[:, 0] = 1
    lr[0, 0] = 0  # ground node is at the top
    for i in range(1, cycle_num):
        lri = np.dot(a.T, lr[:, i - 1])
        lr[:, i] = lri
    lr_end = lr[:, cycle_num - 1] + lr[0, cycle_num - 1] / nodes_num
    lr_end[0] = 0
    lr = np.insert(lr, cycle_num, values=lr_end, axis=1)
    return lr


def recal_weight_target_divided(recal_edge,topics):  # Target's influence is divide to Source[s]
    new_edge = pd.DataFrame(columns=[])
    targets = list(set(recal_edge.Target.tolist()))
    for target in targets:
        #        source = 0
        sources = recal_edge[recal_edge.Target == target]
        #        targets.describe()
        tp_sum = []
        for topic in topics:
            tp_sum.append(sum(sources[topic]))
        for i in sources.index:
            d = {'Source': sources.loc[i].Source,
                'Target': target,
                 'fre':sources.loc[i].fre}
            for topic_index in range(len(topics)):
                d[topics[topic_index]] = sources.loc[i][topics[topic_index]]/ tp_sum[topic_index]
            d = pd.DataFrame(d, index=[i])
            frame = [new_edge, d]
            new_edge = pd.concat(frame)
    max_fre = max(new_edge.fre)
    min_fre = min(new_edge.fre)
    new_edge['fre3'] = new_edge.fre.apply(lambda x: (x-min_fre)/(max_fre-min_fre))
    return new_edge


def recal_weight_source_divided(recal_edge,topics):  # Source's influence is divided to Target[s]
    new_edge = pd.DataFrame(columns=[])
    sources = list(set(recal_edge.Source.tolist()))
    for source in sources:
        #        source = 0
        targets = recal_edge[recal_edge.Source == source]
        #        targets.describe()
        tp_sum = []
        for topic in topics:
            tp_sum.append(sum(targets[topic]))
        for i in targets.index:
            d = {'Source': source,'Target': targets.loc[i].Target}
            for topic_index in range(len(topics)):
                d[topic] = targets.loc[i][topics[topic_index]]/ tp_sum[topic_index]
            d = pd.DataFrame(d, index=[i])
            frame = [new_edge, d]
            new_edge = pd.concat(frame)
    return new_edge


def cal_loss_e(table_recaled, topic, lr_tp, c):  # Source's LR*exp(trs*fre)
    loss_tp = []
    for i in table_recaled.index:
        loss = lr_tp[int(table_recaled.loc[i].Source)] * np.exp(
            c * table_recaled.loc[i][topic] + (1 - c) * table_recaled.loc[i].fre3)
        loss_tp.append(loss)
    return loss_tp


def write_rank(table0):
    topics = [column for column in table0.columns if 'loss_topic' in column]
    for topic in topics:
        table0 = table0.sort_values(by=topic, ascending=False)
        table0[topic + '_rank'] = 0
        l = []
        for i in table0.index:
            if table0.loc[i][topic] not in l:
                l.append(table0.loc[i][topic])
                table0.at[i, topic + '_rank'] = len(l)
            else:
                table0.at[i, topic + '_rank'] = len(l)
    return table0


def gen_new_user_id(table):
    users = list(set(table.parent_user_id) | set(table.user_id))
    nodes_num = len(users)
    users.sort()
    table['Source'] = table['parent_user_id'].apply(lambda x: users.index(x))
    table['Target'] = table['user_id'].apply(lambda x: users.index(x))
    return table,nodes_num


def IDC_main(table0,c = 0.7,cycle_num = 300):
    '''
    table0: trs table
    c:exp [ c*s + (1-c)* A ]
    cycle_num: LR iteration times
    '''
    table0,nodes_num = gen_new_user_id(table0)
    topics = []
    for column in table0.columns:
        if 'topic' in column:
            topics.append(column)

    table = recal_weight_target_divided(table0,topics) #must run, because leaderrank add the ground node, the weight should be divided before adding the ground node
    table0['fre3'] = table['fre3']
    for topic in topics[:]:
        # print("topic is {}, c is {}.".format(topic, c))
        matrix, a = gen_matrix_a(table, nodes_num, topic)
        lr = leaderrank(a, nodes_num, cycle_num)
        lr_tp = lr[1:, cycle_num]
        l = cal_loss_e(table0, topic, lr_tp, c) # table0:the weighted are divided by Source
        table0['loss_'+topic] = l
        table0['Source_lr_' + topic] = table0.Source.apply(lambda x: lr_tp[x])
        table0['Target_lr_' + topic] = table0.Target.apply(lambda x: lr_tp[x])
    table0 = write_rank(table0)
    print('[IDC calculating finished.]')
    return table0

