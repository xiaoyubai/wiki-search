import pandas as pd
import numpy as np
import pyspark as ps
import json
import os
import re
import networkx as nx
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics.pairwise import pairwise_distances


NUMBER_CLUSTERS = 10


def get_title_link(article):
    try:
        links = re.findall(r'\s\[\[([\w+\s]+)[\|\w+\s]*\]\]\s', article)
        title = re.search(r'\'\'\'([\w+\s]+)\'\'\'', article).group(1)
        result = title + "|" + "|".join(links)
        return result
    except:
        return

def get_content(article):
    try:
        title = re.search(r'\'\'\'([\w+\s]+)\'\'\'', article).group(1)
        return [title, article]
    except:
        return

def first_n_articles(rdd, nlines):
    title_articles = np.array(rdd.map(get_content).take(nlines))
    return title_articles

def get_rdd():
    sc = ps.SparkContext()
    try:
        ACCESS_KEY = os.environ['AWS_ACCESS_KEY_ID']
        SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY']
        link = 's3n://%s:%s@wikisample10/sample2' % (ACCESS_KEY, SECRET_ACCESS_KEY)
    except:
        link = 's3n://wikisample10/sample2'
    wiki = sc.textFile(link)
    return wiki


def write_link_to_file(rdd, filename, nlines):
    link_list = rdd.map(get_title_link).take(nlines)
    f = open(filename, 'w+')
    #f = open("../data/test.adjlist", 'w+')
    for link in link_list:
        if link is None: continue
        else:
            f.write(link+'\n')
    f.close()

def get_target_page(rdd, nlines=10):
    pages = rdd.map()

def pre_processing(articles, model=None):
    tokenizer = RegexpTokenizer(r'\w+')
    wordnet = WordNetLemmatizer()
    word_tokens = [tokenizer.tokenize(article.lower()) for article in articles]
    word_stem = [" ".join([wordnet.lemmatize(word) for word in row]) for row in word_tokens]
    if model:
        mat = model.transform(word_stem).toarray()
        return mat
    else:
        model = TfidfVectorizer(stop_words='english', tokenizer=word_tokenize, decode_error='ignore')
        mat = model.fit_transform(word_stem).toarray()
        return mat, model

def kmeans_label(mat, scoring=False):
    kmeans_model = KMeans(n_clusters=NUMBER_CLUSTERS, random_state=1).fit(mat)
    labels = kmeans_model.labels_
    if scoring:
        print "kmeans score: ", silhouette_score(tif_mat, labels, metric='euclidean')
    return labels

def pred_category(model_vect, test, model_pred):
    test_mat = pre_processing([test], model=model_vect)
    print test_mat
    y_pred = model_pred.predict(test_mat)
    y_prob = model_pred.predict_proba(test_mat)
    return y_pred, y_prob



if __name__ == '__main__':
    filename = "../data/test.adjlist"
    first_n_lines = 2000
    redirect_str = '^#REDIRECT'
    rdd = get_rdd()
    rdd = rdd.filter(lambda x: not re.match(redirect_str, x))
    write_link_to_file(rdd, filename, first_n_lines)
    articles = first_n_articles(rdd, first_n_lines)
    titles = np.array([row[0] for row in articles if row])
    contents = np.array([row[1] for row in articles if row])
    tif_mat, model_vect = pre_processing(contents)
    G = nx.read_adjlist(filename, delimiter='|', create_using=nx.DiGraph())
    pr = nx.pagerank(G)
    labels = kmeans_label(tif_mat)

    test_key = 'subject'
    test_category = 'philosophy'
    model_pred = MultinomialNB()
    model_pred.fit(tif_mat, labels)
    #y_pred, y_prob = pred_category(model_vect=model_vect, test=test_category, model_pred=model_pred)
    #possible_articles = titles[labels==y_pred[0]]
    important_articles = [key for key in titles if test_key in key]
    print important_articles
    if test_key in important_articles:
        print "www.wikipedia.org/wiki/%s" % test_key
    else:
        test_mat = pre_processing([test_category], model_vect)
        important_article_index = np.in1d(titles, important_articles)
        pair_cos = pairwise_distances(tif_mat, test_mat, metric='cosine')
        for row in zip(important_articles, \
                       [pr[important_article] for important_article in important_articles], \
                       pair_cos):
            print row
