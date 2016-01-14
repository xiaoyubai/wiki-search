import os
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords, words, wordnet
from pyspark.mllib.linalg import Vectors, SparseVector
import string
from collections import Counter
from pyspark.mllib.clustering import KMeans
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
import json
import string
import pyspark as ps
import numpy as np
import cPickle as pickle

from make_tfidf import TfToken, tokenizing
from als_topic_model import TopicModel, convert_rating, transform


def get_title_link(article):
    try:
        links = re.findall(r'\[\[(.*?)[\]\]|\|]', article)
        title = re.search(r'\'\'\'(.*?)\'\'\'', article).group(1)
        return title, "|".join(links)
    except:
        return "", [""]

def get_title(article):
    try:
        title = re.search(r'\'\'\'(.*?)\'\'\'', article).group(1)
        return title
    except:
        return " "

def get_title_tfidf(title_string):
    titles = title_string.split("|")
    for title in titles:
        yield title

# calcuate cosine similarity between two sparse vectors
def cosine_sim(v1, v2, origin):
    try:
        return v1.dot(v2) / (v1.squared_distance(origin) * v2.squared_distance(origin))
    except:
        return v1.dot(v2) / (v1.squared_distance(origin) * v2.squared_distance(origin) + 1)

def max_cosine_sim(related_tfidf, tf_category):
    num_cols = len(tf_category)
    # initilize a 0 sparse vector to calcuate norm+
    origin = SparseVector(num_cols, {})
    title_cos_sim = np.array([[title, cosine_sim(vector, tf_category, origin)] for title, vector in related_tfidf])
    print title_cos_sim
    return title_cos_sim[np.argmax(title_cos_sim[:,1]),0]

def get_most_similiar_ariticle(idf, keyword, category, multi_links, title_tfidf):
    tf_category = transform(idf, category)
    related_links = multi_links.map(get_title_link).filter(lambda x: x[0]==keyword).map(lambda x: x[1]).first().split("|")
    # related_tfidf = title_tfidf.take(3)
    related_tfidf = title_tfidf.filter(lambda x: x[0] in related_links).collect()
    most_related_title = max_cosine_sim(related_tfidf,  tf_category)
    most_related_tfidf = title_tfidf.filter(lambda x: x[0]==keyword).map(lambda x: x[1]).collect()[0]
    return most_related_title, most_related_tfidf


def same_topic(category, most_related_tfidf, idf, topic_model):
    category_tfidf = transform(idf, category)
    category_topic = topic_model.predict(category_tfidf)
    article_topic = topic_model.predict(most_related_tfidf)
    if category_topic == article_topic:
        return True
    else:
        return False

def train_model(rdd, idf, tfidf):
    multi_links = rdd.filter(lambda line: "may refer to:" in line)
    title_rdd = rdd.map(get_title)
    title_index = title_rdd.zipWithIndex().map(lambda x: (x[1], x[0]))
    tfidf_index = tfidf.zipWithIndex().map(lambda x: (x[1], x[0]))
    title_tfidf = title_index.join(tfidf_index).map(lambda x: x[1])

    topic_model = TopicModel(idf=idf, tfidf=tfidf)
    topic_model.preprocessing()
    topic_model.label()
    topic_model.train()

    return multi_links, title_tfidf, topic_model

if __name__ == '__main__':

    keyword = "recall"
    category = "statistics math math"

    sc = ps.SparkContext()
    aws_link = "jyt109/wiki_articles"

    # aws_link = "wikisample10/sample2"

    # filename="../keypair.json"
    # with open(filename) as f:
    #     data = json.load(f)
    #     access_key = data['ACCESS_KEY']
    #     secret_access_key = data['SECRET_ACCESS_KEY']
    #
    # link = 's3n://%s:%s@%s' % (access_key, secret_access_key, aws_link)
    # rdd = sc.textFile(link)
    # print rdd.count()
    # print "rdd.getNumPartitions(): ", rdd.getNumPartitions()

    tf_token = TfToken(sc=sc, aws_link=aws_link, tokenizer=tokenizing, filename="../keypair.json")
    rdd, idf, tfidf = tf_token.fit()
    print tfidf.take(2)[1]
    f = open('result.csv', 'w')
    f.write("tfidf done")
    multi_links, title_tfidf, topic_model = train_model(rdd, idf, tfidf)
    f.write("tfidf done, train_model done")
    most_related_title, most_related_tfidf = get_most_similiar_ariticle(idf, keyword, category, multi_links, title_tfidf)
    if same_topic(category, most_related_tfidf, idf, topic_model):
        return_title = most_related_title
    fw = "tfidf done, train_model done %s" % most_related_title
    f.write(fw)
