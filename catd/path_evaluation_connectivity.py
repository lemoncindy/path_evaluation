# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 11:28:23 2019

@author: yangy
"""
import queue
import numpy as np

def generate_dict_graph(edges):  # the edge to be searched
    d = {}
    for i in edges.index:
        d_node = d.get(edges.loc[i].Source,[])  # Target-follower, Source-followee
        d_node.append(int(edges.loc[i].Target))
        d[edges.loc[i].Source] = d_node
    return d

def bfs(adj, source, target):
    """
    adj: graph for bfs
    source: source of the edge
    target: target of the edge
    """
    #    adj = graph_whole
    #    start = 0
    visited = set()
    q = queue.Queue()
    start = int(target) # bfs from the target
    q.put(start) #q is the
    visited.add(start) # visited: the visited nodes
    l = []
    while not q.empty():
        u = q.get()  # get the next node to be searched
        if u != source: #avoid self-loop
            l.append(int(u))
            #        print(l)
        for v in adj.get(u, []):
            if u != source: #avoid self-loop
                if v not in visited:
                    visited.add(int(v))
                    q.put(int(v))  # the next node to be searched
        #                l.append(v)
    return l

def disconnected_nodes_set_conditional(edge_index,edges,graph_whole):
    """
    conditional judgement: bfs nodes are divided into target and others.
    if target has other sources besides source, other bfs nodes are connected.
    if target only has source, following bfs nodes need to determine whether they are connected
    First target is not connected, next, if following bfs nodes has other sources besides target, then the node is still connected
    !!! bfs without source to avoid  self-loop
    edge_index: int, to find the edge in edges
    edges: edges, dataframe
    graph_whole:graph with all nodes and edges
    """
    i = edge_index
    nodes = bfs(graph_whole, edges.loc[i].Source,edges.loc[i].Target)
    other_sources_of_target = edges[edges.Target == edges.loc[i].Target].Source.tolist()
    if len(other_sources_of_target) > 1:
        f = []
        return f
    else:
        f = []
        f.append(edges.loc[i].Target) # if target only have one source, ie source, and the edge is blocked, target is disconnected.  #f与遍历的先后顺序相关，源节点到目标节点的多条路径导致 漏记。【一个节点看起来有多个源，其实多个源都来自于同一最初源，如果最初源不能获得信息，那么后面的也不能】
        for node in nodes[1:]:#for each node in Target's bfs_set except target
            other_sources = edges[edges.Target == node].Source.tolist() # the sources of the node in bfs_set

            pl = [] # the other sources in f,f is the disconnected nodes set
            for source in other_sources: # for each source in other sources
                if source in f: # if the source is in bfs_set,
                    pl.append(source) # add it into the pl
            if len(pl) == len(other_sources): # if the other sources are totally same as pl(the nodes in the disconnected set)
                f.append(node)
        f = list(set(f))
        if edges.loc[i].Source in f:
            f.remove(edges.loc[i].Source)
        return f

def write_rank1(table1, column):
    table1[column + '_1rank'] = 0
    table1 = table1.sort_values(by=column, ascending=False)
    l = []
    for i in table1.index:
        if table1.loc[i][column] not in l:
            l.append(table1.loc[i][column])
            table1.at[i, column + '_1rank'] = len(l)
        else:
            table1.at[i, column + '_1rank'] = len(l)
    return table1

def connectivity_main(edges):
    edges = edges[['Source','Target','parent_user_id','user_id','parent_user','user']]
    graph_whole  = generate_dict_graph(edges)
    edges['消失节点' ] = np.empty((len(edges), 0)).tolist()
    edges['消失节点数'] = 0

    edges['id'] = list(np.random.randint(0,10000,len(edges)))

    for i in edges.index:
        disconnected = disconnected_nodes_set_conditional(i, edges, graph_whole)
        edges.at[i, '消失节点'] = disconnected
        edges.at[i, '消失节点数'] = len(disconnected)

    edges = edges.sort_values(by=['消失节点数','id'],ascending=False)
    edges.fillna('[]')
    print('[Connectivity calculating finished.]')
    return edges
    # [['Source','Target','parent_user_id','user_id','parent_user','user','消失节点数','消失节点']]


