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
    # related_links = multi_links.map(get_title_link).filter(lambda x: x[0]==keyword).map(lambda x: x[1]).first().split("|")
    related_tfidf = title_tfidf.take(3)
    # related_tfidf = title_tfidf.filter(lambda x: x[0] in related_links).collect()
    most_related_title = max_cosine_sim(related_tfidf,  tf_category)
    most_related_tfidf = title_tfidf.filter(lambda x: x[0]==keyword).map(lambda x: x[1]).collect()[0]
    return most_related_title, most_related_tfidf

if __name__ == '__main__':

    sc = ps.SparkContext()
    aws_link = "wikisample10/sample2"
    tf_token = TfToken(sc=sc, aws_link=aws_link, tokenizer=tokenizing, filename="../keypair.json")
    rdd, idf, tfidf = tf_token.fit()

    multi_links = rdd.filter(lambda line: "may refer to:" in line)
    title_rdd = rdd.map(get_title)
    title_index = title_rdd.zipWithIndex().map(lambda x: (x[1], x[0]))
    tfidf_index = tfidf.zipWithIndex().map(lambda x: (x[1], x[0]))
    title_tfidf = title_index.join(tfidf_index).map(lambda x: x[1])

    keyword = "Abatement"
    category = "statistics math math"

    most_related_title, most_related_tfidf = get_most_similiar_ariticle(idf, keyword, category, multi_links, title_tfidf)
    print most_related_title
    print most_related_tfidf


    # tf_category = transform(idf, category)
    # related_links = multi_links.map(get_title_link).filter(lambda x: x[0]==keyword).map(lambda x: x[1]).first().split("|")
    #
    # related_tfidf = title_tfidf.filter(lambda x: x[0] in related_links).collect()
    # # related_tfidf = title_tfidf.filter(lambda x: x[0] in related_links)
    #
    # related_tfidf = title_tfidf.take(3)
    # related_title = max_cosine_sim(related_tfidf,  tf_category)
    # print "related article: ", related_title
    # print "related article topic: "
