import os

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def write_execute_log(text,path=os.path.join(os.getcwd(),'output','execute_log.txt')):
    try:
        with open(path,'a') as f:
            f.write(text)
            print(text)
    except:
        with open(path,'w') as f:
            f.write(text)
            print(text)

def IDC_distribution(data,topics,topic_labels,record_source='Weibo',savepath = os.path.join(os.getcwd(),'output','distribution')):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    n_bins = 20
    x = np.zeros(len(data)*len(topics)).reshape(len(data),len(topics))
    for i in range(len(topics)):
        x[:,i] = data['loss_'+topics[i]].tolist()
    fig, ax0 = plt.subplots(figsize=(6,6),nrows=1, ncols=1)
    # ax0 = axes.flatten()

    colors = ['blue', 'black', 'gray' ][:len(topics)]#['#0072bd','#d95319','#edb120','#7e2f8e','#77ac30','#4dbeee','#a2142f']
    labels = topic_labels
    ax0.hist(x, n_bins, density=False, histtype='bar', color=colors, label=labels,alpha =0.9)

    ax0.legend(prop={'size': 16})
    # ax0.set_title('bars with legend')
    # ax0.set_xticks([i for i in range(11)])
    # ax0.set_yscale("log")
    ax0.set_xlim(0,np.max(x)+0.1)
    ax0.set_xlabel('IDC', fontsize=16)
    ax0.set_ylabel('Number', fontsize=16)
    # ax0.set_xticks(np.linspace())

    plt.tick_params(labelsize=14)
    plt.title(record_source)
    fig.tight_layout()
    plt.savefig(os.path.join(savepath,'IDC_distribution_{}.svg'.format(topics[0])),dpi=1200)
    plt.savefig(os.path.join(savepath,'IDC_distribution_{}.png'.format(topics[0])),dpi=1200)
    plt.show()


def connectivity_distribution(connectivity,record_source='Weibo',savepath = os.path.join(os.getcwd(),'output','distribution')):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    data = connectivity['消失节点数']
    max_size = max(data)
    labels = [i for i in range(max_size+1)][::-1]

    ls = sorted(data)
    ls.reverse()
    sdict = {}
    for s in ls:
        sdict[s] = sdict.get(s, 0) + 1
    sizes = []
    for size in labels:
        sizes.append(sdict.get(size, 0))

    width = 0.65  # the width of the bars: can also be len(x) sequence
    fig, ax = plt.subplots(figsize=(6, 6),nrows=1, ncols=1)
    ax.bar(labels, sizes, width, color='grey', label='Connectivity', alpha=0.9)
    ax.set_ylabel('Number', fontsize=16)
    ax.set_xlabel('Size', fontsize=16)
    plt.tick_params(labelsize=14)
    plt.yscale('log')
    plt.title(record_source)
    fig.tight_layout()
    plt.savefig(os.path.join(savepath,'connectivity_distribution.svg'), dpi=1200)
    plt.savefig(os.path.join(savepath,'connectivity_distribution.png'), dpi=1200)
    plt.show()

def path_evaluation_edge_centrality(edges):
    topics = []
    for column in edges.columns:
        if ('topic' in column) & ('_' not in column):
            topics.append(column)
    edges['edge'] = edges.apply(lambda row: (row['Source'],row['Target']),axis=1)

    for topic in topics:
        Gi = nx.DiGraph()
        Gi.add_weighted_edges_from(np.array(edges[['Source','Target',topic]]))
        centrality = nx.edge_betweenness_centrality(Gi,weight='weight')
        edges_centrality = pd.DataFrame(columns=['edge'])
        edges_centrality['edge'] = centrality.keys()
        edges_centrality['edge_centrality_'+topic] = centrality.values()
        edges = pd.concat([edges,edges_centrality],axis=1,join='inner')
    columns = ['Source','Target','parent_user_id','user_id','parent_user','user'] + topics + ['edge_centrality_'+topic for topic in topics]
    edges = edges[columns]
    return edges

def edge_centrality_distribution(edge_centrality,topics,topic_labels,record_source='Weibo', n_bins=20, savepath = os.path.join(os.getcwd(), 'output', 'distribution')):

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    x = np.zeros(len(edge_centrality)*len(topics)).reshape(len(edge_centrality),len(topics))
    for i in range(len(topics)):
        x[:,i] = edge_centrality['edge_centrality_'+topics[i]].tolist()
    fig, ax0 = plt.subplots(figsize=(6,6),nrows=1, ncols=1)
    # ax0 = axes.flatten()

    colors = ['blue', 'black', 'gray' ][:len(topics)]#['#0072bd','#d95319','#edb120','#7e2f8e','#77ac30','#4dbeee','#a2142f']
    labels = topic_labels
    ax0.hist(x, n_bins, density=False, histtype='bar', color=colors, label=labels,alpha =0.9)

    ax0.legend(prop={'size': 16})
    # ax0.set_title('bars with legend')
    # ax0.set_xticks([i for i in range(11)])
    # ax0.set_yscale("log")
    ax0.set_xlim(0,np.max(x)+0.1)
    # ax0.legend(prop={'size': 16})
    # ax0.set_title('bars with legend')
    # ax0.set_xticks([i for i in range(11)])
    ax0.set_yscale("log")
    # ax0.set_xscale("log")
    ax0.set_xlim(0, np.max(x))
    ax0.set_xlabel('Edge centrality', fontsize=16)
    ax0.set_ylabel('Number', fontsize=16)
    # ax0.set_xticks(np.linspace())
    plt.title(record_source)
    plt.tick_params(labelsize=14)
    fig.tight_layout()
    plt.savefig(os.path.join(savepath, 'edge_centrality_distribution_{}.svg'.format(topics[0])), dpi=1200)
    plt.savefig(os.path.join(savepath, 'edge_centrality_distribution_{}.png'.format(topics[0])), dpi=1200)
    plt.show()

def descending_sort(ls):
    ls = sorted(ls)
    ls.reverse()
    return ls

def trs_distribution(data,topic_labels,record_source='Weibo',savepath = os.path.join(os.getcwd(),'output')):
    # n_bins = np.linspace(0, 1, 41)
    # topics = pd.read_csv(r'D:\E\trs_path_evaluation\output\topics_label.csv')

    topics_num = len(topic_labels)
    x = np.zeros(len(data)* topics_num).reshape(len(data), topics_num)

    for i in range(topics_num):
        tp = 'topic' + str(i)
        x[:, i] = descending_sort(list(data[tp]))

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(18, 4))
    colors = ['k','gray','blue','red']
    markers = ['o','+','^','*']
    for i in range(int(topics_num/4)+1):
        if i * 4 < topics_num:
            ax0.plot(list(range(1, 1+len(data))), x[:, i * 4], marker=markers[i],color=colors[i], markerfacecolor='none', linewidth=0,markersize=2, alpha=0.7,
                     label=topic_labels.loc[i * 4, 'topic_label'])
        if i * 4+1 < topics_num:
            ax1.plot(list(range(1, 1+len(data))), x[:, i * 4 + 1], marker=markers[i], color=colors[i], markerfacecolor='none',linewidth=0, markersize=2,
                     alpha=0.7, label=topic_labels.loc[i * 4 + 1, 'topic_label'])
        if i * 4 +2 < topics_num:
            ax2.plot(list(range(1, 1+len(data))), x[:, i * 4 + 2], marker=markers[i],color=colors[i], markerfacecolor='none',linewidth=0, markersize=2, alpha=0.7,
                     label=topic_labels.loc[i * 4 + 2, 'topic_label'])
        if i * 4 + 3 < topics_num:
            ax3.plot(list(range(1, 1+len(data))), x[:, i * 4 + 3], marker=markers[i],color=colors[i], markerfacecolor='none',linewidth=0, markersize=2, alpha=0.7,
                     label=topic_labels.loc[i * 4 + 3, 'topic_label'])
        # ax0.plot([0, 1000], [0.5, 0.], '--r')
        # ax1.plot([0, 1000], [0.5, 0.], '--r')
        # ax2.plot([0, 1000], [0.5, 0.], '--r')
        # ax3.plot([0, 1000], [0.5, 0.], '--r')
        ax0.set_xlabel('Sequence Number', fontsize=12)
        ax1.set_xlabel('Sequence Number', fontsize=12)
        ax2.set_xlabel('Sequence Number', fontsize=12)
        ax3.set_xlabel('Sequence Number', fontsize=12)
        ax0.set_ylabel('Relationship Strength', fontsize=14)
        ax0.legend()
        ax1.legend()
        ax2.legend()
        ax3.legend()
    plt.title(record_source)
    plt.savefig(os.path.join(savepath,'trs_distribution.svg'), dpi=600)
    plt.show()
