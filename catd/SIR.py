# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 12:51:53 2019

@author: yangy

SIR1
"""
import os.path

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import catd.util

def gen_matrix(weighted, edges, nodes_num, topic='1'):
    matrix = np.zeros(nodes_num * nodes_num).reshape(nodes_num, nodes_num)
    if weighted == 0:
        for i in edges.index:
            matrix[int(edges.loc[i].Source), int(edges.loc[i].Target)] = 1
        matrix_degree = matrix
        return matrix_degree,matrix
    else:
        matrix_degree = np.zeros(nodes_num * nodes_num).reshape(nodes_num, nodes_num)
        for i in edges.index:
            matrix[int(edges.loc[i].Source), int(edges.loc[i].Target)] = edges.loc[i][topic] # trs
            matrix_degree[int(edges.loc[i].Source), int(edges.loc[i].Target)] = 1
        return matrix_degree, matrix


def find_source(matrix, nodes_num, source_num):
    indegree = np.zeros(nodes_num)
    for i in range(nodes_num):
        indegree[i] = sum(matrix[i, :])
    indegree = list(indegree)
    unique_indegree = list(set(indegree))
    unique_indegree.sort()
    sources = []
    for i in range(source_num):
        max_degree = unique_indegree.pop()
        max_sources = np.where(indegree==max_degree)
        sources.append(random.choice(max_sources[0]))
    return sources


# sources  = find_source(matrix,nodes_num,source_num)

def spreading(matrix, sources,m,b):  # 一次有m个节点感染

    T = 50
    iter_num = 200
    nodes_num = len(matrix)
    # I_p = np.zeros((iter_num,T))
    S_p = np.zeros((iter_num, T))
    R_p = np.zeros((iter_num, T))
    S = np.zeros(T)
    # I = np.zeros(T)
    R = np.zeros(T)
    global I
    I = []
    for t0 in range(iter_num):
        #        t0 = 2
        state = np.zeros((T, nodes_num))
        for source in sources:
            state[0, source] = 1  # source 在第一步状态为 I ；传播源
        for t in range(1, T):
            #            t = 1
            #            print('t:',t,',')
            for i in range(nodes_num):
                #                i = 3
                if state[t - 1, i] == 1:
                    #                    print('i:',i)
                    #                    i = 3
                    if sum(matrix[i, :]) > 0:
                        j_list0 = list(np.nonzero(matrix[i, :])[0])
                        #                        print(j_list0)
                        j_list = []
                        j_weight = []
                        for j in j_list0:
                            if state[t - 1, j] == 0:
                                #                            print(j)
                                j_list.append(j)
                                j_weight.append(matrix[i, j])
                        if len(j_list) > 0:
                            #                            传染概率 m ,感染节点数len(j_list)* m 四舍五入
                            #                            int(np.round(1.5,0))
                            infected_j = random.choices(j_list, weights=j_weight, k=int(np.round(len(j_list) * m, 0)))
                        #                            print(i,j_list,j_weight,infected_j)
                        for j in j_list:
                            if j in infected_j:
                                state[t, j] = 1
                            else:
                                state[t, j] = 0
                    rand = random.uniform(0, 1)
                    if rand <= b:
                        state[t, i] = -1
                    else:
                        state[t, i] = 1
                #                    state[t,i] = 1
                elif state[t - 1, i] == -1:
                    state[t, i] = -1
                #                    rand = random.uniform(0,1)
        #                    if rand <= 0.8:
        #                        state[t,i] = 0
        #                    else:
        #                        state[t,i] = -1

        for t1 in range(T):
            S_p[t0, t1] = list(state[t1, :]).count(0) / nodes_num
            # I_p[t0,t1] = list(state[t1,:]).count(1)/nodes_num
            r = list(state[t1, :]).count(-1) + list(state[t1, :]).count(1)
            R_p[t0, t1] = r / nodes_num
        I.extend(list(np.nonzero(state[49, :])[0]))
        I = list(set(I))
    for t in range(T):
        S[t] = np.sum(S_p[:, t]) / iter_num
        # I[t] = np.sum(I_p[:,t])/iter_num
        R[t] = np.sum(R_p[:, t]) / iter_num
    #    print(j_list,j_weight,infected_j)
    return R


def plot_tp1(topic, all_egdes, partial_edges, labels,records_source,savepath = os.path.join(os.getcwd(),'output','sir')):
    """
    all_edges: all edges' diffusion results
    partial_edges: partial edges' diffusion resuls
    label: labels of results
    """
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    fig, ax = plt.subplots(figsize=(6.5, 6.5), nrows=1, ncols=1)
    colors = ['black','blue','grey']
    for i in range(len(all_egdes)):
        ax.plot(np.arange(50), all_egdes[i], color = colors[i], marker ='*', linewidth=1, label=labels[i], alpha=0.8)

    for i in range(len(partial_edges)):
        ax.plot(np.arange(50), partial_edges[i], color = colors[i], marker ='.', linewidth=1, label=labels[i] +' Part', alpha=0.8)

    plt.legend(loc='center right', fontsize=14)
    plt.xlabel('t', size=18)
    plt.ylabel('Coverage', size=18)
    plt.tick_params(labelsize=16)
    plt.title(records_source)
    fig.tight_layout()
    plt.savefig(os.path.join(savepath,topic+',critical edges.png'),dpi=1200)
    plt.savefig(os.path.join(savepath,topic+',critical edges.svg'),dpi=1200)

    plt.show()


def plot_R1(topic,R1, R11):
    fig, ax = plt.subplots(figsize=(6, 6), nrows=1, ncols=1)
    ax.plot(np.arange(50), R1, 'k*-', linewidth=1, label='diffusion', alpha=0.8)
    ax.plot(np.arange(50), R11, 'k.-', linewidth=1, label='diffusion part', alpha=0.8)
    plt.legend(loc='center right')
    plt.xlabel('t', size=14)
    plt.ylabel('R', size=14)
    fig.tight_layout()
    plt.tick_params(labelsize=12)
    plt.title(topic)


def SIR_top_n(topic,edges_idc,edges_connectivity,edges_centrality,records_source='Weibo',
              c = 0.7,beta = 0.3,gamma = 0.2,source_num = 2,
              savepath = os.path.join(os.getcwd(),'output','sir')):
    """
    c:parameter in IDC calculating
    beta：infected probability
    gamma：recovered probability
    source_num: number of spreading source
    """
    nodes_num = len(set(edges_idc.Source)|set(edges_idc.Target))
    n = len(edges_connectivity[edges_connectivity.消失节点数>0]) #number of top n edges
    I = []
    i = 0

    weighted = 1
    matrix1_degree, matrix = gen_matrix(weighted, edges_idc, nodes_num, 'loss_'+topic)
    sources1 = find_source(matrix1_degree, nodes_num, source_num) # entire network
    R1 = spreading(matrix,sources1,beta,gamma)

    edges_idc_1 = edges_idc.sort_values(by='loss_' + topic, ascending=False)[:n]  #top n edges
    matrix1_degree1, matrix1 = gen_matrix(weighted, edges_idc_1, nodes_num, 'loss_'+topic)
    sources1_1 = find_source(matrix1_degree1, nodes_num, source_num)
    R11 = spreading(matrix1, sources1_1, beta, gamma)
    catd.util.write_execute_log("method IDC, whole net diffusion covers {} nodes, top  {} edges covers {} of ratio {}.\n".format(R1[-1], n, R11[-1],R11[-1] / R1[-1]))


    weighted = 0
    matrix_degree,matrix = gen_matrix(weighted, edges_connectivity, nodes_num)
    # sources0 = find_source(matrix_degree, nodes_num, source_num)
    sources0 = sources1 # the same entire network
    R0 = spreading(matrix, sources0,beta,gamma)

    edges_connectivity_1 = edges_connectivity.sort_values(by='消失节点数', ascending=False)[:n]
    matrix_degree,matrix = gen_matrix(weighted, edges_connectivity_1, nodes_num)
    sources0_1 = find_source(matrix, nodes_num, source_num)
    R01 = spreading(matrix, sources0_1, beta,gamma)
    catd.util.write_execute_log("method connectivity, whole net diffusion covers {} nodes, top n {} edges covers {} of ratio {}.\n".format(R0[-1], n, R01[-1], R01[-1] / R0[-1]) )


    weighted = 1
    matrix_degree,matrix = gen_matrix(weighted, edges_centrality, nodes_num, 'edge_centrality_'+topic)
    sources2 = sources1
    R2 = spreading(matrix, sources2,beta,gamma)

    edges_centrality_1 = edges_centrality.sort_values(by='edge_centrality_'+topic, ascending=False)[:n]
    matrix_degree,matrix = gen_matrix(weighted, edges_centrality_1, nodes_num,'edge_centrality_'+topic)
    sources2_1 = find_source(matrix_degree, nodes_num, source_num)
    R21 = spreading(matrix, sources2_1, beta,gamma)
    catd.util.write_execute_log("method edge centrality, whole net diffusion covers {} nodes, top n {} edges covers {} of ratio {}.\n".format(R2[-1], n, R21[-1], R21[-1] / R2[-1]) )

    all_edges = [R1,R0,R2] #['IDC','Connectivity','Edge Centrality']
    partial_edges = [R11,R01,R21]
    labels = ['IDC','Connectivity','Edge Centrality']
    plot_tp1(topic,all_edges, partial_edges,labels,records_source,savepath = savepath)


def SIR_different_number_of_edges(topic,
                                  edges,method='IDC',
                                  eighty_percent_number=1270,
                                beta = 0.3,gamma = 0.2,source_num = 2 ,
                                savepath = os.path.join(os.getcwd(),'output','sir')):
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    if 'topic' in topic:
        weighted = 1
        if method == 'IDC':
            by = 'loss_'+topic
        if method == 'edge centrality':
            by = 'edge_centrality_' + topic
    elif topic.lower() == 'connectivity':
        by = '消失节点数'
        weighted = 0

    nodes_num = len(set(edges.Source) | set(edges.Target))

    matrix_degree,matrix = gen_matrix(weighted, edges, nodes_num, by)
    sources1 = find_source(matrix_degree, nodes_num, source_num)
    R1 = spreading(matrix, sources1, beta, gamma)
    edges_number = [0]
    coverage = [0]
    catd.util.write_execute_log("method {} diffusion process\n".format(topic if weighted==0 else method))

    edges_1 = edges.sort_values(by=by,ascending=False)
    for i in np.linspace(0,len(edges),10,endpoint=True)[1:]:
        edges_1_1 = edges_1[:int(i)]
        matrix1_degree1,matrix1 = gen_matrix(weighted,edges_1_1,nodes_num,by)
        sources1_1 = find_source(matrix1_degree1,nodes_num,source_num)
        R11 = spreading(matrix1,sources1_1,beta,gamma)
        catd.util.write_execute_log("gamma={}, top {} edges cover {} nodes with proportion of {} in the whole net\n".format(gamma,int(i),R11[-1],R11[-1]/R1[-1]))

        coverage.append(R11[-1])#/R1[-1]
        edges_number.append(int(i))
    # coverage.append(R1[-1])#1
    # edges_number.append(2224)

    edges_1_1 = edges_1[:eighty_percent_number]
    matrix1_degree1, matrix1 = gen_matrix(weighted, edges_1_1, nodes_num,by)
    sources1_1 = find_source(matrix1_degree1, nodes_num, source_num)
    R11 = spreading(matrix1, sources1_1, beta, gamma)
    catd.util.write_execute_log(
        "gamma={}, top {} edges cover {} nodes with proportion of {} in the whole net\n".format(gamma, eighty_percent_number, R11[-1],
                                                                                                R11[-1] / R1[-1]))

    return [edges_number,coverage],eighty_percent_number,R11[-1] #/ R1[-1]


def plot_different_number_edges(topic, sir_steps_idc,sir_steps_connectivity,sir_steps_edge_centrality,
                                eighty_percent_edges_number,eighty_percent,
                                records_source='Weibo',
                                eighty_paint=True,
                                savepath = os.path.join(os.getcwd(),'output','sir')):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    fig, ax = plt.subplots(figsize=(6.5, 6.5), nrows=1, ncols=1)
    if eighty_paint == True:
        sir_steps_idc_edge_num = [edge_number for edge_number in sir_steps_idc[0] if
                                  edge_number < eighty_percent_edges_number] + [eighty_percent_edges_number] + [
                                     edge_number for edge_number in sir_steps_idc[0] if
                                     edge_number > eighty_percent_edges_number]
        sir_steps_idc_coverage = [coverage for coverage in sir_steps_idc[1] if coverage < eighty_percent] + [
            eighty_percent] + [coverage for coverage in sir_steps_idc[1] if coverage > eighty_percent]

        plt.plot([eighty_percent_edges_number], [eighty_percent], 'r*')
        plt.annotate('(' + str(eighty_percent_edges_number) + ', ' + str(np.round(eighty_percent, 2)) + ')',
                     xy=[eighty_percent_edges_number, eighty_percent], xytext=(-80, 10),
                     textcoords='offset points', fontsize=16)  # 将x主刻度标签设置为3的倍数
    else:
        sir_steps_idc_edge_num = sir_steps_idc[0]
        sir_steps_idc_coverage = sir_steps_idc[1]
    plt.plot(sir_steps_idc_edge_num , sir_steps_idc_coverage, 'k.-', label='IDC')
    plt.plot(sir_steps_connectivity[0][0],sir_steps_connectivity[0][1],'b.-',label = 'Connectivity')
    plt.plot(sir_steps_edge_centrality[0][0],sir_steps_edge_centrality[0][1],color='grey',marker='.',label = 'Edge Centrality')
    plt.xticks([int(i) for i in np.linspace(0,sir_steps_idc[0][-1],10,endpoint=True)],fontsize=14)#,rotation=15) #将x主刻度标签设置为3的倍数
    plt.yticks(fontsize=14)
    plt.xlabel('Top-N edges',fontsize=16)
    plt.ylabel('Coverage',fontsize=16)
    plt.title(records_source)
    fig.tight_layout()
    plt.legend(loc='lower right',fontsize=16)

    plt.savefig(os.path.join(savepath,topic+'-e-different_number.svg'),dpi=1200)
    plt.savefig(os.path.join(savepath,topic+'-e-different_number.png'),dpi=1200)
    plt.show()


