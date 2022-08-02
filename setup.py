import time

import catd.construct_interactive_network
import catd.util
import catd.path_evaluation_IDC
import catd.SIR
import catd.path_evaluation_connectivity
from catd import *
import pandas as pd
import os
import winsound
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    record_file_name = 'tweet_records'
    records_source = 'Twitter'
    records_file_path = os.path.join(os.getcwd(),'data',record_file_name+'.xlsx')
    records_output_path = os.path.join(os.getcwd(),'output')
    catd.util.write_execute_log("executed time is {}, file is {}.\n".format(time.strftime('%Y-%m-%d %H:%M:%S'),record_file_name))
    df = pd.read_excel(records_file_path)


    #  text processing
    # cutting word
    if '分词' not in df.columns:
        t = pd.DataFrame(df['content'].astype(str))  # content to be cut
        start=time.perf_counter()
        t["分词"] = t.content.apply(catd.construct_interactive_network.CUT_CHINESE_WORD)
        catd.util.write_execute_log('Wordcut costs time of {} s.\n'.format(time.perf_counter()-start))
        #
        df['分词'] = t["分词"]
        df.to_excel(records_file_path, index=False)

    if 'topic0' not in df.columns:
        #generating document-word matrix, tf
        df = pd.read_excel(records_file_path)#[['Id', '分词']]
        n_top_words = 30
        n_features = 500
        p_ls = []
        corre_ls = []
        start = time.perf_counter()
        tf, weight, tf_feature_names = catd.construct_interactive_network.tf_idf(n_features, df,count='tfidf')
        catd.util.write_execute_log('tf features number is {}.\n'.format(n_features))

        catd.util.write_execute_log('tf costs time of {} s.\n'.format(time.perf_counter() - start))

        # topic model to assign texts' topic vector
        topics_numbers = np.linspace(6,20,8)
        for i in topics_numbers:#iterating to find the best topics number
            n_topics = int(i)
            # print('topics number is {}'.format(n_topics))
            start = time.perf_counter()
            c, perplexity, lda = catd.construct_interactive_network.lda_model(tf, n_topics)
            # print('perplexity is {}.'.format(perplexity))
            p_ls.append(perplexity)
            corre_ls.append(c)
            catd.util.write_execute_log('LDA: topics number is {} and perplexity is {}.\n'.format(n_topics,perplexity))

        a = list(topics_numbers)
        # a.insert(0,5)
        plt.plot(a, p_ls, '*-')
        # plt.plot(a, corre_ls, '.-')
        # plt.yscale("symlog")
        plt.savefig(os.path.join(records_output_path,'lda_process_'+'.png'),dpi=600)
        plt.show()


        i =  a[p_ls.index(min(p_ls))]
        n_topics = int(i)
        c, perplexity, lda = catd.construct_interactive_network.lda_model(tf, n_topics)
        catd.util.write_execute_log('LDA: topics number is {} and perplexity is {}.\n'.format(n_topics, perplexity))
        catd.util.write_execute_log('LDA costs time of {} s.\n'.format(time.perf_counter() - start))

        catd.construct_interactive_network.write_top_words(lda, tf_feature_names, n_top_words, n_topics)
        catd.construct_interactive_network.ave_corre(c)
        a = lda.components_
        docres = lda.fit_transform(weight)

        for i in range(n_topics):
            topic_name = 'topic'+str(i)
            df[topic_name] = docres[:,i]
        df.to_excel(records_file_path,index=False)

    #calculating users' preferences
    output_files = os.listdir(records_output_path)
    if 'tp_ur.xlsx' not in output_files:
        df = pd.read_excel(records_file_path)  # the interactive records with topic vector
        start = time.perf_counter()
        user_posts = catd.construct_interactive_network.gen_users_posts(df)
        user_posts.to_excel(os.path.join(records_output_path,'user_posts.xlsx'),index=False)  # output: Denominator of the relationship strength calculation
        tp_ur = catd.construct_interactive_network.cal_user_preference(user_posts)
        tp_ur.to_excel(os.path.join(records_output_path,'tp_ur.xlsx'),index=False)  # output: Denominator of the relationship strength calculation
        catd.util.write_execute_log('users\' preference calculating costs time of {} s.\n'.format(time.perf_counter() - start))

    tp_ur = pd.read_excel(os.path.join(records_output_path,'tp_ur.xlsx'))

    #calculating multi-topic relationship strength [trs]
    cal_trs = 0
    if cal_trs == 1:
        start = time.perf_counter()
        df = pd.read_excel(records_file_path)
        trs = catd.construct_interactive_network.cal_trs(df, tp_ur)
        trs.to_excel(os.path.join(records_output_path,'trs.xlsx'),index=False)
        catd.util.write_execute_log('trs calculating costs time of {} s.\n'.format(time.perf_counter() - start))

    #load trs
    trs = pd.read_excel(os.path.join(records_output_path, 'trs.xlsx'))
    cal_idc = 0
    c = 0.7
    # calculating IDC
    # trs = catd.path_evaluation_IDC.IDC_main(trs)
    # trs = catd.path_evaluation_IDC.write_rank(trs)
    # trs.to_excel(os.path.join(records_output_path, 'trs.xlsx'), index=False)
    if cal_idc == 1:
    # if 'loss_topic0' not in trs.columns:
        start = time.perf_counter()
        # trs.rename(columns = {"parent_user_id":"Source","user_id":"Target"},inplace=True)
        trs = catd.path_evaluation_IDC.IDC_main(trs,c = c)
        trs = catd.path_evaluation_IDC.write_rank(trs)
        trs.to_excel(os.path.join(records_output_path, 'trs.xlsx'),index=False)
        catd.util.write_execute_log('IDC calculating costs time of {} s.\n'.format(time.perf_counter() - start))

    # calculating connectivity[disconnected nodes set]
    if 'connectivity.xlsx' not in output_files:
        start = time.perf_counter()
        connectivity = catd.path_evaluation_connectivity.connectivity_main(trs)
        connectivity.to_excel(os.path.join(records_output_path, 'connectivity.xlsx'), index=False)
        catd.util.write_execute_log('connectivity calculating costs time of {} s.\n'.format(time.perf_counter() - start))


    # calculating edge centrality(betweenness centrality)
    if 'edge_centrality.xlsx' not in output_files:
        start = time.perf_counter()
        edge_centrality = catd.util.path_evaluation_edge_centrality(trs)
        # edge_centrality = catd.path_evaluation_connectivity.write_rank1(edge_centrality,'edge_centrality')
        edge_centrality.to_excel(os.path.join(records_output_path, 'edge_centrality.xlsx'), index=False)
        catd.util.write_execute_log('edge centrality calculating costs time of {} s.\n'.format(time.perf_counter() - start))

    # compare the results of the three methods
    idc = pd.read_excel(os.path.join(records_output_path,'trs.xlsx'))
    connectivity = pd.read_excel(os.path.join(records_output_path,'connectivity.xlsx'))
    edge_centrality = pd.read_excel(os.path.join(records_output_path, 'edge_centrality.xlsx'))

    #distribution of trs
    topic_labels = pd.read_csv(os.path.join(records_output_path,'topic_labels.csv'))[:n_topics]
    catd.util.trs_distribution(idc,topic_labels,record_source=records_source)

    # #granularity comparing
    topics_list = [[],[],[],[]]
    for i in range(int(n_topics/4)+1):
        for j in range(4):
            if i * 4 + j < n_topics:
                topics_list[j].append('topic'+str(i * 4 + j))

    topic_labels0 = pd.read_csv(os.path.join(records_output_path, 'topic_labels.csv'), encoding='utf8')
    for topics in topics_list:
        topic_labels = [topic_labels0.loc[int(topic[5:])].topic_label for topic in topics]
        # [topicl[5:] for topic in topics]
        catd.util.IDC_distribution(idc, topics,topic_labels,record_source=records_source)
        catd.util.edge_centrality_distribution(edge_centrality, topics,topic_labels,record_source=records_source)
    catd.util.connectivity_distribution(connectivity,record_source=records_source)

    #spreading ability comparing
    source_num = 2
    start = time.perf_counter()
    topics = []
    for column in df.columns:
        if 'topic' in column:
            topics.append(column)
            
    sir_steps_connectivity = catd.SIR.SIR_different_number_of_edges('connectivity', connectivity,eighty_percent_number=2000,source_num=source_num)
    for topic in topics[：]:
        catd.util.write_execute_log('{} SIR process.\n'.format(topic))
        catd.SIR.SIR_top_n(topic,idc,connectivity,edge_centrality,records_source,source_num = source_num)
        sir_steps_idc,eighty_percent_number,eighty_percent = catd.SIR.SIR_different_number_of_edges(topic,idc,source_num = source_num)
        sir_steps_edge_centrality = catd.SIR.SIR_different_number_of_edges(topic, edge_centrality,method = 'edge centrality',source_num = source_num)
        catd.SIR.plot_different_number_edges(topic, sir_steps_idc, sir_steps_connectivity, sir_steps_edge_centrality,eighty_percent_number, eighty_percent,records_source=records_source,eighty_paint=False)
    catd.util.write_execute_log('SIR costs time of {} s.\n'.format(time.perf_counter() - start))

    #dynamic_evaluating
    # dynamic_output_path = os.path.join(os.getcwd(),'output','dynamic_process')
    # if not os.path.exists(dynamic_output_path):
    #     os.makedirs(dynamic_output_path)
    # start = time.perf_counter()
    # df1,df2,df3 = df[df.month <= 9],df[df.month <= 10],df[df.month <= 11]
    # user_posts = pd.read_excel(os.path.join(records_output_path,'user_posts.xlsx'))
    # user_posts1, user_posts2, user_posts3  = user_posts[user_posts.repost_time<='2017/9/30'],user_posts[user_posts.repost_time<='2017/10/31'],user_posts[user_posts.repost_time<='2017/11/30']
    # tp_ur1 =catd.construct_interactive_network.cal_user_preference(user_posts1)
    # tp_ur2 =catd.construct_interactive_network.cal_user_preference(user_posts2)
    # tp_ur3 =catd.construct_interactive_network.cal_user_preference(user_posts3)
    # trs1 = catd.construct_interactive_network.cal_trs(df1, tp_ur1)
    # trs2 = catd.construct_interactive_network.cal_trs(df2, tp_ur2)
    # trs3 = catd.construct_interactive_network.cal_trs(df3, tp_ur3)
    # idc1 = catd.path_evaluation_IDC.IDC_main(trs1)
    # idc2 = catd.path_evaluation_IDC.IDC_main(trs2)
    # idc3 = catd.path_evaluation_IDC.IDC_main(trs3)
    # dynamic_idc_writer = pd.ExcelWriter(os.path.join(dynamic_output_path,'dynamic_IDC.xlsx'))
    # idc1.to_excel(dynamic_idc_writer,sheet_name='9',index=False)
    # idc2.to_excel(dynamic_idc_writer,sheet_name='9+10',index=False)
    # idc3.to_excel(dynamic_idc_writer,sheet_name='9+10+11',index=False)
    # dynamic_idc_writer.save()
    #
    # connectivity1 = catd.path_evaluation_connectivity.connectivity_main(trs1)
    # connectivity2 = catd.path_evaluation_connectivity.connectivity_main(trs2)
    # connectivity3 = catd.path_evaluation_connectivity.connectivity_main(trs3)
    #
    # dynamic_connectivity_writer = pd.ExcelWriter(os.path.join(dynamic_output_path, 'dynamic_connectivity.xlsx'))
    # connectivity1.to_excel(dynamic_connectivity_writer, sheet_name='9', index=False)
    # connectivity2.to_excel(dynamic_connectivity_writer, sheet_name='9+10', index=False)
    # connectivity3.to_excel(dynamic_connectivity_writer, sheet_name='9+10+11', index=False)
    # dynamic_connectivity_writer.save()
    # catd.util.write_execute_log('dynamic evaluating costs time of {} s.\n'.format(time.perf_counter() - start))
