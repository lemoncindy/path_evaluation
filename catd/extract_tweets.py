import random
import sqlite3
import numpy as np
import os

import pandas as pd
import networkx as nx

def get_data_by_sql(sql_sentence,
                    database=os.path.join(os.getcwd(),'..','data','tweet.db')):
    con = sqlite3.connect(database)
    cursor = con.cursor()
    cursor.execute(sql_sentence)
    r = cursor.fetchall()
    r = np.array(r)
    return r
def correct_parent_user_id(x):#some user has two id, we correct them into the same one.
    if x == '58462':#tifanyazzac
        return 83268
    elif x == '44941':#ilovelohan
        return 80791
    else:
        return int(x)

def match_id_words(words_id,words=os.path.join(os.getcwd(),'..','data','WordTable.txt')):
    with open (words,'r') as f:
        words_table = f.readlines()
    words_id = words_id.split(' ')
    words_ = []
    for id in words_id:
        words_.append()

def internet_repost(following_network,follower,followee):
    # print(follower,followee)
    inter_net = np.where((following_network[:,0]==follower) & (following_network[:,1]==followee))
    # print(follower,followee,len(inter_net[0]))
    return len(inter_net[0])

def gen_new_user_id(table):
    users = list(set(table.parent_user_id) | set(table.user_id))
    sorted(users)
    nodes_num = len(users)
    table['Source'] = table['parent_user_id'].apply(lambda x: users.index(x))
    table['Target'] = table['user_id'].apply(lambda x: users.index(x))
    nodes = pd.DataFrame(columns=['user_id','user_temporal_id'])
    nodes['user_id'] = users
    nodes['user_temporal_id'] = [i for i in range(len(users))]
    # nodes.reset_index('user_temporal_id')
    return table, nodes,nodes_num

def filter_twitter_network(twitter_network_file=os.path.join(os.getcwd(),'..','data','graph_cb.txt'),delimiter=' ',timestamp=0):
    following_network = np.loadtxt(twitter_network_file,delimiter=delimiter,dtype=int)
    if timestamp == 0:
        twitter_network = following_network
    else:
        twitter_network = following_network[np.where(following_network[:,2]==timestamp)]
    table0 = pd.DataFrame(columns=['user_id','parent_user_id','timestamp'],data=twitter_network)
    table0, nodes, nodes_num = gen_new_user_id(table0)
    G = nx.DiGraph()
    G.add_edges_from(np.array(table0[['Target','Source']]))
    nodes_set = []
    for c in nx.weakly_connected_components(G):
        nodes_set_i = G.subgraph(c).nodes()
        if len(nodes_set_i) > len(nodes_set):
            nodes_set = nodes_set_i
    c = nodes_set
    table0['Source_inter'] = table0['Source'].apply(lambda x:1 if x in c else 0)
    table0['Target_inter'] = table0['Target'].apply(lambda x:1 if x in c else 0)
    connected_component = table0[(table0['Source_inter']==1) & (table0['Target_inter']==1)]
    return connected_component

def pruning(twitter_network_file=os.path.join(os.getcwd(),'..','data','graph_cb.txt'),delimiter=' ',timestamp=0):

    following_network = np.loadtxt(twitter_network_file,delimiter=delimiter,dtype=int)
    if timestamp == 0:
        twitter_network = following_network
    else:
        twitter_network = following_network[np.where(following_network[:,2]==timestamp)]
    table0 = pd.DataFrame(columns=['user_id','parent_user_id','timestamp'],data=twitter_network)
    table0, nodes, nodes_num = gen_new_user_id(table0)
    nodes = nodes.reset_index(drop=True)
    G = nx.DiGraph()
    G.add_edges_from(np.array(table0[['Target', 'Source']]))#following network,target is the follower
    nodes_in_degree = np.array(G.in_degree())
    nodes_out_degree = np.array(G.out_degree())
    zero_in_degree_nodes = nodes_in_degree[np.where(nodes_in_degree[:, 1] == 0)[0]]
    zero_out_degree_nodes = nodes_out_degree[np.where(nodes_out_degree[:, 1] == 0)[0]]
    zero_nodes = np.row_stack((zero_out_degree_nodes,zero_in_degree_nodes))[:,0]
    zero_nodes = np.unique(zero_nodes)
    count = 0
    while len(zero_nodes):
        #delete nodes with zero indegree or zero outdegree
        # for i in range(len(zero_nodes)):
        nodes = nodes.drop(labels=list(zero_nodes),axis=0)
        print('{}-th pruning, leaving {} nodes.'.format(count,len(nodes)))
        count += 1
        table0['Source_leaf'] = table0.Source.apply(lambda x: 1 if x in zero_nodes else 0)
        table0['Target_leaf'] = table0.Target.apply(lambda x: 1 if x in zero_nodes else 0)
        table0 = table0[(table0.Source_leaf==0)&(table0.Target_leaf==0)]
        table0, nodes, nodes_num = gen_new_user_id(table0)
        nodes = nodes.reset_index(drop=True)
        G = nx.DiGraph()
        G.add_edges_from(np.array(table0[['Target', 'Source']]))
        nodes_in_degree = np.array(G.in_degree())
        nodes_out_degree = np.array(G.out_degree())
        zero_in_degree_nodes = nodes_in_degree[np.where(nodes_in_degree[:, 1] == 0)[0]]
        zero_out_degree_nodes = nodes_in_degree[np.where(nodes_out_degree[:, 1] == 0)[0]]
        zero_nodes = np.row_stack((zero_out_degree_nodes, zero_in_degree_nodes))[:, 0]
        zero_nodes = np.unique(zero_nodes)
    return table0[['user_id','parent_user_id','timestamp']],nodes[['user_id']]


def graph(twitter_network_file=os.path.join(os.getcwd(),'..','data','graph_cb.txt'),delimiter=' ',timestamp=0):
    """
    following:
    """
    following_network = np.loadtxt(twitter_network_file,delimiter=delimiter,dtype=int)
    if timestamp == 0:
        following_network = following_network
    else:
        following_network = following_network[np.where(following_network[:,2]==timestamp)]
    table = pd.DataFrame(columns=['user_id', 'parent_user_id', 'timestamp'], data=following_network)
    table, nodes, nodes_num = gen_new_user_id(table)
    graph = {}
    users = nodes.user_temporal_id.tolist()
    for i in users:
        graph[i] = []
    for i in table.index:
        followers = graph.get(table.loc[i]['Source'])
        followers.append(table.loc[i]['Target'])
        graph[table.loc[i]['Source']] = followers
    return graph,table,nodes


def snowball_sampling(table,nodes,snowball_method = 'or'):
    table['Source_inter'] = table.parent_user_id.apply(lambda x:1 if x in nodes else 0)
    table['Target_inter'] = table.user_id.apply(lambda x:1 if x in nodes else 0)
    if snowball_method == 'or':
        table0 = table[(table.Source_inter==1) | (table.Target_inter==1)]
    else:
        table0 = table[(table.Source_inter == 1) & (table.Target_inter == 1)]
    return table0[['user_id','parent_user_id','timestamp']]

def gen_matrix(twitter_network_file=os.path.join(os.getcwd(),'..','data','graph_cb.txt'),delimiter=' ',timestamp=0):
    following_network = np.loadtxt(twitter_network_file,delimiter=delimiter,dtype=int)
    if timestamp == 0:
        following_network = following_network
    else:
        following_network = following_network[np.where(following_network[:,2]==timestamp)]
    table = pd.DataFrame(columns=['user_id','parent_user_id','timestamp'],data=following_network)
    table, nodes, nodes_num = gen_new_user_id(table)
    matrix = np.zeros(nodes_num * nodes_num,dtype=int).reshape(nodes_num, nodes_num)
    for i in table.index:
        matrix[table.loc[i]['Source'], table.loc[i]['Target']] = 1
    return matrix,table,nodes

def find_source(graph, source_num):
    nodes_num = len(graph)
    indegree = {}
    for i in range(nodes_num):
        indegree[i] = len(graph[i])
    indegrees_descending = sorted(indegree.values())[::-1][:source_num]
    sources = []
    for k,v in indegree.items():
        if v in indegrees_descending:
            sources.append(k)
    return sources

def spreading(graph, sources,nodes,m=0.2):  # 一次有m个节点感染
    T = 50
    iter_num = 200
    nodes_num = len(graph)
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
            for i in range(nodes_num):
                if state[t - 1, i] == 1:
                    if len(graph[i]) > 0:
                        j_list0 = graph[i]
                        j_list = []
                        j_weight = []
                        for j in j_list0:
                            if state[t - 1, j] == 0:
                                j_list.append(j)
                                j_weight.append(1)#连边权重均为1
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
                    state[t, i] = 1
                    # rand = random.uniform(0, 1)
                #     if rand <= b:
                #         state[t, i] = -1
                #     else:
                #         state[t, i] = 1
                # #                    state[t,i] = 1
                # elif state[t - 1, i] == -1:
                #     state[t, i] = -1
                # #                    rand = random.uniform(0,1)
        #                    if rand <= 0.8:
        #                        state[t,i] = 0
        #                    else:
        #                        state[t,i] = -1

        for t1 in range(T):
            S_p[t0, t1] = list(state[t1, :]).count(0) / nodes_num
            # I_p[t0,t1] = list(state[t1,:]).count(1)/nodes_num
            r = list(state[t1, :]).count(1)
            R_p[t0, t1] = r / nodes_num
        I.extend(list(np.nonzero(state[49, :])[0]))
        I = list(set(I))
    for t in range(T):
        S[t] = np.sum(S_p[:, t]) / iter_num
        # I[t] = np.sum(I_p[:,t])/iter_num
        R[t] = np.sum(R_p[:, t]) / iter_num
    #    print(j_list,j_weight,infected_j)
    nodes['snowballing'] = 0
    for node in I:
        nodes.loc[node]['snowballing'] = 1
    return R,nodes[nodes.snowballing==1].user_id.tolist()

if __name__ == "__main__":
    #graph后缀：s-snowball sampling; c-connected component; p-pruning ; 多个字符 是按字符顺序进行的处理

    #snowball_method default 'or'
    snowball_method = 'or'
    method = ''

    #######################################filtering following netwotrk####################################
    #
    # # graph, table, nodes = graph(twitter_network_file=os.path.join(os.getcwd(),'..','data','little.csv'),delimiter=',')
    # # matrix,table,nodes = gen_matrix(twitter_network_file=os.path.join(os.getcwd(),'..','data','interactive.txt'))
    # # graph,table,nodes = graph(twitter_network_file=os.path.join(os.getcwd(),'..','data','interactive.txt'))
    # graph,table,nodes = graph(twitter_network_file=os.path.join(os.getcwd(),'..','data','graph_cb.txt'),timestamp=1)
    # sources = find_source(graph,20)
    # R,I = spreading(graph,sources,nodes,m=0.3)
    # snowballing = snowball_sampling(table,I,snowball_method=snowball_method)
    # np.savetxt(os.path.join(os.getcwd(),'..','data','graph_s{}.txt'.format(snowball_method)),np.array(snowballing),fmt='%d')
    # np.savetxt(os.path.join(r'C:\Users\yxy\Desktop\framework\path_evaluation\twitter','gephi_graph_s{}.csv'.format(snowball_method)),np.array(snowballing[['user_id','parent_user_id']]),delimiter=',',fmt='%d')
    #
    #
    #
    # table0,nodes = pruning(twitter_network_file=os.path.join(os.getcwd(),'..','data','graph_s{}.txt'.format(snowball_method)))#剪枝
    # np.savetxt(os.path.join(os.getcwd(),'..','data','graph_sp{}.txt'.format(snowball_method)),np.array(table0[['user_id','parent_user_id','timestamp']]),fmt='%d',)
    #
    #
    # connected_component = filter_twitter_network(twitter_network_file=os.path.join(os.getcwd(),'..','data','graph_sp{}.txt'.format(snowball_method)))#最大连通体
    # np.savetxt(os.path.join(os.getcwd(),'..','data','graph_spc{}.txt'.format(snowball_method)),np.array(connected_component[['user_id','parent_user_id','timestamp']]),fmt='%d',)
    # np.savetxt(os.path.join(r'C:\Users\yxy\Desktop\framework\path_evaluation\twitter','gephi_graph_spc{}.csv'.format(snowball_method)),
    #            np.array(connected_component[['user_id','parent_user_id','timestamp']]),delimiter=',',fmt='%d',)

    #######################################filtering following netwotrk####################################


    # table0,nodes = pruning()# connected_component = filter_twitter_network(twitter_network_file=os.path.join(os.getcwd(),'..','data','graph.txt'),delimiter=' ')
    # np.savetxt(os.path.join(os.getcwd(),'..','data','graph_p.txt'),np.array(table0[['user_id','parent_user_id','timestamp']]),fmt='%d',)
    # connected_component = filter_twitter_network(twitter_network_file=os.path.join(os.getcwd(),'..','data','graph_p.txt'))
    # np.savetxt(os.path.join(os.getcwd(),'..','data','graph_pc.txt'),np.array(connected_component[['user_id','parent_user_id','timestamp']]),fmt='%d',)
    # table0,nodes = pruning(twitter_network_file=os.path.join(os.getcwd(),'..','data','graph_pc.txt'))
    # np.savetxt(os.path.join(os.getcwd(),'..','data','graph_pcp.txt'),np.array(table0[['user_id','parent_user_id','timestamp']]),fmt='%d',)


    sql = 'select * from (select t.*, u.user_name as parent_user_1, u.original_user_id as original_parent_user_id,u.user_new_id as parent_user_id from (select id, tweet_id,time,user_name as user, user_new_id as user_id,original_user_id,retweet_from as parent_user, content  from tweet3 where retweet_from != -1 and content != "")t left join usermap u on t.parent_user=u.user_name ) a where a.user_id != " " and a.parent_user_1 != ""'
    a = get_data_by_sql(sql)
    df = pd.DataFrame(columns=['id','order','repost_time','user','user_id','original_user_id','parent_user','content','parent_user_1','original_parent_user_id','parent_user_id'],data=a)
    df['parent_user_id'] = df.parent_user_id.apply(lambda x:correct_parent_user_id(x))
    df['user_id'] = df.user_id.astype(int)
    df['csd'] = 1
    df['fre'] = 1
    df['分词'] = df.content
    df.to_excel(os.path.join(os.getcwd(),'..','data','tweet_records_all.xlsx'),index=False)


    #现在运行的部分会在这里结束，因为之前graph_sp.txt 按照graph_sp.csv读取
    #还有一个问题是，snowball sampling 按照 或 “|” 取 连边，数量大大增加。是不是应按照 和 “&” 来取 ？


    ########################用筛选好的following network过滤交互记录#########################################################


    # # df = pd.read_excel(os.path.join(os.getcwd(),'..','data','tweet_records.xlsx'))
    # following_network = np.loadtxt(os.path.join(os.getcwd(),'..','data','graph_spc{}.txt'.format(snowball_method)),dtype=int)
    # # following_network = np.loadtxt(os.path.join(os.getcwd(),'..','data','graph.txt'),dtype=int)
    # # following_network = np.loadtxt(os.path.join(os.getcwd(),'..','data','graph_{}.txt'.format(method)),dtype=int)
    # print('internet deciding')
    # for i in df.index:
    #     df.loc[i,'inter_net'] = internet_repost(following_network,df.loc[i,'user_id'],df.loc[i,'parent_user_id'])
    # df_inter_net = df[df.inter_net>0]
    #
    #
    # df_inter_net = df_inter_net[df_inter_net.repost_time <'2011-01-01 00:00:00+00:00']
    #
    # df_inter_net = df_inter_net.drop_duplicates(['order','user','parent_user','content','repost_time'])
    # df_inter_net.to_excel(os.path.join(os.getcwd(),'..','data','tweet_records_graph.xlsx'),index=False)
    #
    #
    #
    # # df_inter_net = pd.read_excel(os.path.join(os.getcwd(),'..','data','tweet_records_{}.xlsx'.format(method)))
    # interactive = df_inter_net[['user_id','parent_user_id']]
    # interactive['timestamp'] = 1
    # interactive = interactive.drop_duplicates(['user_id','parent_user_id'])
    # np.savetxt(os.path.join(os.getcwd(),'..','data','interactive_{}.txt'.format(method)),np.array(interactive[['user_id','parent_user_id','timestamp']]),fmt='%d')
    # np.savetxt(os.path.join(r'C:\Users\yxy\Desktop\framework\path_evaluation\twitter','gephi_interactive_{}.csv'.format(method)),np.array(interactive[['user_id','parent_user_id']]),delimiter=',',fmt='%d',)
    #######################用筛选好的following network过滤交互记录#########################################################








    ##########################################筛选交互网络###################################################################
    df_inter_net = pd.read_excel(os.path.join(os.getcwd(), '..', 'data', 'tweet_records_graph.xlsx'))

    connected_component = filter_twitter_network(twitter_network_file=os.path.join(os.getcwd(),'..','data','interactive_{}.txt'.format(method)))
    np.savetxt(os.path.join(os.getcwd(),'..','data','interactive_c{}.txt'.format(method)),np.array(connected_component[['user_id','parent_user_id','timestamp']]),fmt='%d',)


    table,nodes = pruning(twitter_network_file=os.path.join(os.getcwd(),'..','data','interactive_c{}.txt'.format(method)),delimiter=' ')
    np.savetxt(os.path.join(os.getcwd(),'..','data','interactive_cp{}.txt'.format(method)),np.array(table[['user_id','parent_user_id','timestamp']]),fmt='%d',)
    
    # table0,nodes = pruning(twitter_network_file=os.path.join(os.getcwd(),'..','data','interactive.txt'))
    # np.savetxt(os.path.join(os.getcwd(),'..','data','interactive_p.txt'),np.array(table0[['user_id','parent_user_id','timestamp']]),fmt='%d',)
    


    graph,table,nodes = graph(twitter_network_file=os.path.join(os.getcwd(),'..','data','interactive_c{}.txt'.format(method)))
    sources = find_source(graph,20)
    R,I = spreading(graph,sources,nodes,m=0.3)
    snowballing = snowball_sampling(table,I,snowball_method=snowball_method)
    np.savetxt(os.path.join(os.getcwd(),'..','data','interactive_cs{}.txt'.format(method)),np.array(snowballing),fmt='%d')
    # np.savetxt(os.path.join(os.getcwd(),'..','data','gephi_interactive_s{}.csv'.format(snowball_method)),np.array(snowballing[['user_id','parent_user_id']]),delimiter=',',fmt='%d')
    # # table0,nodes = pruning(twitter_network_file=os.path.join(os.getcwd(),'..','data','interactive_s{}.csv'.format(snowball_method)),delimiter=',')
    # # np.savetxt(os.path.join(os.getcwd(),'..','data','interactive_sp.txt'),np.array(table0[['user_id','parent_user_id','timestamp']]),fmt='%d')
    #
    #
    #
    following_network = np.loadtxt(os.path.join(os.getcwd(),'..','data','interactive_cp{}.txt'.format(method)),dtype=int)
    # df['inter_net'] = df.apply(lambda row:internet_repost(row['user_id'],row['parent_user_id'],following_network),axis=1)
    for i in df_inter_net.index:
        df_inter_net.loc[i,'inter_net'] = internet_repost(following_network,df_inter_net.loc[i,'user_id'],df_inter_net.loc[i,'parent_user_id'])
    df_inter_net = df_inter_net[df_inter_net.inter_net>0]


    df_inter_net = df_inter_net[df_inter_net.repost_time < '2011-01-01 00:00:00+00:00']

    df_inter_net_output_file = os.path.join(os.getcwd(),'..','data','tweet_records_cp.xlsx')

    df_inter_net = df_inter_net.drop_duplicates(['order','user','parent_user','content','repost_time'])
    df_inter_net.to_excel(df_inter_net_output_file,index=False)
    print('df output file is {}'.format(df_inter_net_output_file))


    # df_inter_net = pd.read_excel(os.path.join(os.getcwd(),'..','data','tweet_records_s.xlsx'))
    interactive_file = os.path.join(r'C:\Users\yxy\Desktop\framework\path_evaluation\twitter','gephi_interactive_cp.csv')
    interactive = df_inter_net[['user_id','parent_user_id']]
    interactive['timestamp'] = 1
    interactive = interactive.drop_duplicates(['user_id','parent_user_id'])
    np.savetxt(os.path.join(os.getcwd(),'..','data','interactive_cs{}.txt'.format(method)),np.array(interactive[['user_id','parent_user_id']]),fmt='%d')
    np.savetxt(interactive_file,np.array(interactive[['user_id','parent_user_id']]),delimiter=',',fmt='%d')
    interactive_users = list(set(interactive.parent_user_id)|set(interactive.user_id))

    print('interactive network file is {}'.format(interactive_file))






