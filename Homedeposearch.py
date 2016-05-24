# -*- coding: utf-8 -*-
import time
start_time = time.time()

import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import gensim
import sys
import string
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, make_scorer
import random
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn import cross_validation

from nltk.corpus import stopwords
from SpellCheck import spell_check
import logging

def prodinfo_to_wordlist( prodinfo_text):
    # Function to convert a document to a sequence of words,
    # 1. Remove non-letters
    prodinfo_text = re.sub("[^a-zA-Z]"," ", prodinfo_text)
    # 2. Convert words to lower case, split them and remove stop words
    stops = set(stopwords.words("english"))
    words = " ".join([word for word in spell_check(prodinfo_text.lower()).split() if word not in stops])
    # 3. Return a list of words
    return(words)

# Define a function to split a prodinfo into parsed sentences
def prodinfo_to_sentences( prodinfo, tokenizer):
    # Function to split a prodinfo into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(prodinfo.strip())
    # raw_sentences = tokenizer.tokenize(prodinfo.decode('utf-8').strip())
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
    # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call prodinfo_to_wordlist to get a list of words
            sentences.append( prodinfo_to_wordlist( raw_sentence ))
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences

def prodinfo_to_wordlist( prodinfo):
    prodinfo_text = re.sub("[^a-zA-Z]"," ", prodinfo)
    words = prodinfo_text.lower().split()
    stops = set(stopwords.words("english"))
    words = [w for w in words if not w in stops]
    return(words)

# Function to average all of the word vectors in a given paragraph
def makeFeatureVec(words, model, num_features):
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    featureVec = np.divide(featureVec,nwords)
    return featureVec

# Given a set of prodinfos (each one a list of words), calculate
# the average feature vector for each one and return a 2D numpy array
def getAvgFeatureVecs(prodinfos, model, num_features):
    # Initialize a counter
    counter = 0.
    # Preallocate a 2D numpy array, for speed
    prodinfoFeatureVecs = np.zeros((len(prodinfos),num_features),dtype="float32")
    # Loop through the prodinfos
    for prodinfo in prodinfos:
       #
       # Print a status message every 1000th prodinfo
       if counter%100. == 0.:
           print "done %d of %d" % (counter, len(prodinfos))
       # Call the function (defined above) that makes average feature vectors
       prodinfoFeatureVecs[counter] = makeFeatureVec(prodinfo, model, num_features)
       #Increment the counter
       counter = counter + 1.
    return prodinfoFeatureVecs

def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

RMSE = make_scorer(fmean_squared_error, greater_is_better=False)

def RMSE(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_


def run_est_func(params):
    n_estimators, learning_rate, max_depth, subsample = params
    n_estimators=int(n_estimators)
    print params
    clf = GradientBoostingRegressor(n_estimators= n_estimators, learning_rate=learning_rate, max_depth=max_depth, subsample=subsample)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    rmse = RMSE( y_test, y_pred )
    return rmse

def optimize(trials):
    space = (
             hp.quniform('n_estimators', 100,200, 100),
             hp.quniform('learning_rate', 0.1, 0.2, 0.1),
             hp.quniform('max_depth', 5, 6, 1),
             hp.quniform('subsample', 0.5, 1, 0.5)
             )
    best = fmin(run_est_func, space, algo=tpe.suggest, trials=trials, max_evals=10)
    print best

random.seed(2016)
stemmer = SnowballStemmer('english')



if __name__ == '__main__':
    df_train = pd.read_csv('../../Data/train.csv', encoding="ISO-8859-1")
    df_test = pd.read_csv('../../Data/test.csv', encoding="ISO-8859-1")
    df_pro_desc = pd.read_csv('../../Data/product_descriptions.csv',encoding="ISO-8859-1")
    df_attribute = pd.read_csv('../../Data/attributes.csv',encoding="ISO-8859-1")
    df_brand = df_attribute[df_attribute.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})
    df_material = df_attribute[df_attribute.name == "Material"][["product_uid", "value"]].rename(columns={"value": "material"})

    df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
    df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')
    df_material1 = df_material.drop_duplicates(['product_uid'])
    df_all = pd.merge(df_all, df_material1, how='left', on='product_uid')
    print df_all.shape

    reload (sys)
    sys.setdefaultencoding('ISO-8859-1')

    print("--- Files Loaded: %s minutes ---" % round(((time.time() - start_time)/60),2))

    stop_w = ['for', 'xbi', 'and', 'in', 'th','on','sku','with','what','from','that','less','er','ing'] #'electr','paint','pipe','light','kitchen','wood','outdoor','door','bathroom'
    strNum = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9}

    print ("String clean ups...")
    df_all['brand']  = df_all['brand'].fillna('unbrand')
    df_all['brand'] = np.where(df_all['brand'].isin (['n a ','na',' na','nan']),'unbrand',df_all['brand'])

    regex = re.compile('[%s]' % re.escape(string.punctuation))
    df_all["search_term"] = df_all["search_term"].map(lambda x: regex.sub("", x))
    df_all["product_title"] = df_all["product_title"].map(lambda x: regex.sub("", x))
    df_all["product_description"] = df_all["product_description"].map(lambda x: regex.sub("", x))

    df_all['combined_info'] = df_all['search_term'] + " " + \
                                    df_all['product_title'] + " " + \
                                    df_all['product_description']

    df_all.to_csv('../../Data/df_all_vect.csv',index=False)
    # Line from 171 to 210 commented to avoid creation of word2vec model again and again
    # print "Read %d labeled train and test prodinfos" % (df_all["combined_info"].size)
    #
    # # Load the punkt tokenizer
    # tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    #
    # # ****** Split the labeled and unlabeled training sets into clean sentences
    # #
    # sentences = []  # Initialize an empty list of sentences
    # from gensim.models import Word2Vec
    # print "Parsing sentences from training set"
    # for prodinfo in df_all["combined_info"]:
    #     sentences += prodinfo_to_sentences(prodinfo, tokenizer)
    #
    # # Import the built-in logging module and configure it so that Word2Vec
    # # creates nice output messages
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', \
    #                     level=logging.INFO)
    #
    # # Set values for various parameters
    # num_features = 300  # Word vector dimensionality
    # min_word_count = 30  # Minimum word count
    # num_workers = -1  # Number of threads to run in parallel
    # context = 20  # Context window size
    # downsampling = 1e-3  # Downsample setting for frequent words
    #
    # # Initialize and train the model (this will take some time)
    # print "Training Word2Vec model..."
    # model = Word2Vec(sentences, workers=num_workers, \
    #                  size=num_features, min_count=min_word_count, \
    #                  window=context, sample=downsampling, seed=1)
    #
    # # If you don't plan to train the model any further, calling
    # # init_sims will make the model much more memory-efficient.
    # model.init_sims(replace=True)
    #
    # # It can be helpful to create a meaningful model name and
    # # save the model for later use. You can load it later using Word2Vec.load()
    # model_name = "300features_30minwords_20context"
    # model.save(model_name)

    #Load the word2vec model
    model = gensim.models.Word2Vec.load("300features_30minwords_20context")
    print 'Model loaded'
    # ****************************************************************
    # Calculate average feature vectors

    num_style_desc = df_all.shape[0]
    for col in ['search_term','product_title','product_description']:
        cleantxt=[]
        for i in xrange( 0, num_style_desc ):
            if( (i+1)%10000== 0 ):
                print "vectors %d of %d\n" % ( i+1, num_style_desc )
            cleantxt.append(prodinfo_to_wordlist(df_all[col][i]) )
        print col
        vec = col+'_Vecs'
        num_features=300
        vec = getAvgFeatureVecs( cleantxt, model, num_features )
        print vec.shape
        df_vec = pd.DataFrame(vec,columns = ['vec_'+col+'_'+str(k) for k in range(num_features)])
        df_vec.to_csv('../../Data/'+col+'_word2vecs.csv',index=False)

    df_all = pd.read_csv('../../Data/df_all_vect.csv', encoding="ISO-8859-1")
    srch_vec = pd.read_csv('../../Data/search_term_word2vecs.csv').fillna(0.0)
    pdt_ttl_vec = pd.read_csv('../../Data/product_title_word2vecs.csv').fillna(0.0)
    pdt_desc_vec = pd.read_csv('../../Data/product_description_word2vecs.csv').fillna(0.0)

    srch_vec = srch_vec.as_matrix(columns=[srch_vec.columns[:300]])
    pdt_ttl_vec = pdt_ttl_vec.as_matrix(columns=[pdt_ttl_vec.columns[:300]])
    pdt_desc_vec = pdt_desc_vec.as_matrix(columns=[pdt_desc_vec.columns[:300]])

    warnings.filterwarnings('ignore')

    dst_srch_ttl1 = np.zeros(srch_vec.shape[0])
    for i in range(srch_vec.shape[0]):
        d1 = srch_vec[i, :]
        d2 = pdt_ttl_vec[i, :]
        dst_srch_ttl1[i] = cosine_similarity(d1, d2)

    dst_srch_desc1 = np.zeros(srch_vec.shape[0])
    for i in range(srch_vec.shape[0]):
        d1 = srch_vec[i, :]
        d2 = pdt_desc_vec[i, :]
        dst_srch_desc1[i] = cosine_similarity(d1, d2)

    dst_ttl_desc1 = np.zeros(srch_vec.shape[0])
    for i in range(srch_vec.shape[0]):
        d1 = pdt_ttl_vec[i, :]
        d2 = pdt_desc_vec[i, :]
        dst_srch_desc1[i] = cosine_similarity(d1, d2)

    svd = TruncatedSVD(n_components=30, random_state=2016)

    srch_vec = svd.fit_transform(srch_vec)
    pdt_ttl_vec = svd.fit_transform(pdt_ttl_vec)
    pdt_desc_vec = svd.fit_transform(pdt_desc_vec)

    srch_vec = pd.DataFrame(srch_vec, columns=['srch_vec_' + str(i) for i in range(srch_vec.shape[1])])
    pdt_ttl_vec = pd.DataFrame(pdt_ttl_vec, columns=['ttl_vec_' + str(i) for i in range(pdt_ttl_vec.shape[1])])
    pdt_desc_vec = pd.DataFrame(pdt_desc_vec, columns=['desc_vec_' + str(i) for i in range(pdt_desc_vec.shape[1])])

    id = list(df_all['id'])
    srch_vec['id'] = id
    pdt_ttl_vec['id'] = id
    pdt_desc_vec['id'] = id

    df_all = pd.merge(df_all, srch_vec, how='left', on='id')
    df_all = pd.merge(df_all, pdt_ttl_vec, how='left', on='id')
    df_all = pd.merge(df_all, pdt_desc_vec, how='left', on='id')

    df_all['dst_srch_ttl1'] = dst_srch_ttl1
    df_all['dst_srch_desc1'] = dst_srch_desc1
    df_all['dst_ttl_desc1'] = dst_ttl_desc1

    cols = list(df_all.select_dtypes(include=['object']).columns)

    df_all1 = df_all.drop(cols, 1)
    df_all1.to_csv('../../Data/df_all_new_feat3.csv', index=False)

    # Training
    df_all = pd.read_csv('../../Data/df_all_new_feat3.csv', encoding="ISO-8859-1")

    df_val = df_all[df_all['relevance'].isnull()]
    df_train = df_all[~df_all['relevance'].isnull()]
    id_val = df_val['id']
    y_train = df_train['relevance'].values

    X_train, X_test, y_train, y_test = train_test_split(df_train, y_train, test_size=0.3, random_state=1234)

    X_train = X_train.drop(['id', 'relevance'], axis=1)
    X_test = X_test.drop(['id', 'relevance'], axis=1)

    trials = Trials()
    optimize(trials)
    trials.best_trial

    df_val = df_all[df_all['relevance'].isnull()]
    df_train = df_all[~df_all['relevance'].isnull()]
    id_val = df_val['id']
    y_train = df_train['relevance'].values

    X_train = df_train.drop(['id', 'relevance'], axis=1)
    X_test = df_val.drop(['id', 'relevance'], axis=1)

    clf = GradientBoostingRegressor(n_estimators=340, learning_rate=0.15, max_depth=4, subsample=0.75)

    t0 = time.time()
    scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=3, scoring="mean_squared_error", n_jobs=-1)
    print scores
    print 'Cross Val time taken:', (time.time() - t0) / 60.0, 'minutes'

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    min_y_pred = min(y_pred)
    max_y_pred = max(y_pred)
    min_y_train = min(y_train)
    max_y_train = max(y_train)
    print(min_y_pred, max_y_pred, min_y_train, max_y_train)
    for i in range(len(y_pred)):
        if y_pred[i] < 1.0:
            y_pred[i] = 1.0
        if y_pred[i] > 3.0:
            y_pred[i] = 3.0

    id_test = df_val['id']
    pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('../Predictions/submission.csv', index=False)
