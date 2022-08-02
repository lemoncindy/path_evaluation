import os
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation #LDAmodel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

# 创建停用词列表
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

# 分词函数，精确模式
def CUT_CHINESE_WORD(text,
                     userdict_filepath=os.path.join(os.getcwd(),'data','dots1.txt'),
                     stopwords_filepath=os.path.join(os.getcwd(),'data','stopwords3.txt')):
    jieba.load_userdict(userdict_filepath)
    stopwords = stopwordslist(stopwords_filepath)  # loading the path of the stopwords list
    seg_list = jieba.cut(text, cut_all=False)  # cut_all=True [cut mode]
    seg_list_without_stopwords = ' '
    for word in seg_list:
        if word not in stopwords:
            if word != '\t':
                seg_list_without_stopwords += ' ' + word
    return seg_list_without_stopwords
#提取特征词
def tf_idf(n_features,df,count='count'):
    if count == 'count':
        tf_vectorizer= CountVectorizer(strip_accents='unicode',
                                 max_features=n_features,
                                 stop_words='english',
                                 max_df=0.5,
                                 min_df=1)
    else:
        tf_vectorizer = TfidfVectorizer(strip_accents='unicode',
                                 max_features=n_features,
                                 stop_words='english',
                                 max_df=0.5,
                                 min_df=1)
    tf=tf_vectorizer.fit_transform(df['分词'].astype(str))
    tf_feature_names=tf_vectorizer.get_feature_names()
    weight = tf_vectorizer.fit_transform(df['分词'].astype(str)).toarray()
    return tf,weight,tf_feature_names

#LDA
def lda_model(tf,n_topics):
    lda=LatentDirichletAllocation(n_components=n_topics,max_iter=50,
                             learning_method='batch',# if the dataset is not that big, batch is enough. [less parameters]
                             learning_offset=50,
                             random_state=0)
    lda.fit(tf)
    perplexity=lda.perplexity(tf)
    c=lda.components_#topic is the distribution of words
    c=np.mat(c)
#    docres = lda.fit_transform()
#    perplexity = 0
    return c,perplexity,lda

#calculting the similarity
def ave_corre(c):
    b=[]
    sum_m=0
    for i in range(len(c)):
        b.append(np.linalg.norm(c[i]))
    for i in range(len(c)-1):
        for j in range(i+1,len(c)):
#            print(i,j,c[i]*c[j].T/(b[i]*b[j]))
            sum_m+=c[i]*c[j].T/(b[i]*b[j])
    ave_corre=sum_m*2/(len(c)*(len(c)-1))
    return ave_corre[0,0]


#output the feature words of each topic
def write_top_words(model, feature_names, n_top_words,n_topics):
    topic_words_path = os.path.join(os.getcwd(),'output',"topic_words_"+str(n_topics)+".txt")
    words_file = open(topic_words_path,'w')
#    model = lda
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
#        print(message)
        words_file.writelines(message)
        words_file.write('\n')
    words_file.close()

#make a piece of user's post record
def make_user_record(d, csd, user, user_id):
    d = d.reset_index(drop=True)
    d.loc[0, 'user'] = user
    d.loc[0, 'user_id'] = user_id
    d.loc[0, 'csd'] = csd
    return d

# generating users' post record from the interactive records
def gen_users_posts(df):

    df['fre'] = 1
    xuhao = df[['order', 'fre']].groupby(['order'], as_index=False).sum()
    d_order = ['order', 'csd', 'repost_time', 'user', 'user_id']
    for column in df.columns:
        if 'topic' in column:
            d_order.append(column)

    df_all_users = pd.DataFrame(columns=['order'])
    for i in xuhao.index:
        #        print(i,xuhao.loc[i].order,xuhao.loc[i].fre)
        all_csds = df[df.order == xuhao.loc[i].order]
        if xuhao.loc[i].fre > 1:
            all_csds = all_csds.sort_values(by='csd', ascending=False)  # ascending by the cascade
            d = pd.DataFrame(all_csds.iloc[0:1])
            d = d[d_order]  #d: topic vector and the user who reposts the post

            d = make_user_record(d, all_csds.iloc[0].csd + 1, all_csds.iloc[0].parent_user,
                                 all_csds.iloc[0].parent_user_id)  # recording the post of the parent_user of the first cascade
            frame = [df_all_users, d]
            df_all_users = pd.concat(frame)

            d = make_user_record(d, all_csds.iloc[0].csd, all_csds.iloc[0].user,
                                 all_csds.iloc[0].user_id)  # recording the post of the user(child) of the first cascade
            frame = [df_all_users, d]
            df_all_users = pd.concat(frame)

            for j in range(1, len(all_csds)):
                #            print(j,all_csds.iloc[j].csd,all_csds.iloc[j].csd == all_csds.iloc[j-1].csd - 1)
                if all_csds.iloc[j].csd == all_csds.iloc[j - 1].csd - 1:
                    d = make_user_record(d, all_csds.iloc[j].csd, all_csds.iloc[j].user,
                                         all_csds.iloc[j].user_id)  # recording the post of the user(child) of the first cascade
                    frame = [df_all_users, d]
                    df_all_users = pd.concat(frame)

                else:
                    d = make_user_record(d, all_csds.iloc[j].csd + 1, all_csds.iloc[j].parent_user,
                                         all_csds.iloc[j].parent_user_id)  # recording the post of the parent_user of the first cascade
                    frame = [df_all_users, d]
                    df_all_users = pd.concat(frame)

                    d = make_user_record(d, all_csds.iloc[j].csd, all_csds.iloc[j].user,
                                         all_csds.iloc[j].user_id)  # recording the post of the user(child) of the first cascade
                    frame = [df_all_users, d]
                    df_all_users = pd.concat(frame)
        else:
            d = pd.DataFrame(all_csds.iloc[0:1])
            d = d[d_order] #d: topic vector and the user who reposts the post

            d = make_user_record(d, all_csds.iloc[0].csd + 1, all_csds.iloc[0].parent_user,
                                 all_csds.iloc[0].parent_user_id)  # recording the post of the parent_user of the first cascade
            frame = [df_all_users, d]
            df_all_users = pd.concat(frame)

            d = make_user_record(d, all_csds.iloc[0].csd, all_csds.iloc[0].user,
                                 all_csds.iloc[0].user_id)  # recording the post of the user(child) of the first cascade
            frame = [df_all_users, d]
            df_all_users = pd.concat(frame)
    return df_all_users

#calculting users' preferences
def cal_user_preference(df_all_users):# df_all_users is the users' post records
    # df_all_users = cal_user_tp(df)
    d_order = df_all_users.columns
    user_order = d_order[3:]
    tp_ur = df_all_users.groupby(['user', 'user_id'], as_index=False).sum()  #sum as Denominator
    tp_ur = tp_ur[user_order]
    print('[users\' topic preference finished.]')
    return tp_ur

#calculating trs
def cal_trs(df, tp_ur):
    df['fre'] = 1
    tp_vx = df.groupby(['parent_user', 'parent_user_id', 'user', 'user_id'], as_index=False).sum()
    order = ['parent_user', 'parent_user_id', 'user', 'user_id','fre']
    topics = []
    for column in df.columns:
        if 'topic' in column:
            topics.append(column)
    order += topics
    tp_vx = tp_vx[order]
    trs = pd.DataFrame(columns=order)
    for i in tp_vx.index:
        parent = tp_ur[tp_ur.user == tp_vx.loc[i].parent_user]
        parent = parent.reset_index()
        d = {'parent_user': tp_vx.loc[i]['parent_user'],
             'user': tp_vx.loc[i]['user'],
             'parent_user_id': tp_vx.loc[i]['parent_user_id'],
             'user_id': tp_vx.loc[i]['user_id'],
             'fre':tp_vx.loc[i]['fre']} # fre: the interactive frequency/activity
        for topic in topics:
            d[topic] = tp_vx.loc[i][topic] / (parent.iloc[0][topic] + 1e-5) # plus 1e-5 to avoid the denominator of 0
        d = pd.DataFrame(d, index=[i])
        frame = [trs, d]
        trs = pd.concat(frame)
        trs = trs[order]
    print('[topic-oriented relationship strength finished.]')
    return trs


