import sys
import os
import re
# from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords, words, wordnet
from pyspark.mllib.recommendation import ALS, Rating
from pyspark.mllib.linalg import Vectors, SparseVector
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.regression import LabeledPoint
import string
from collections import Counter
from pyspark.mllib.clustering import KMeans
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
import json
import pyspark as ps
import numpy as np
import cPickle as pickle
from collections import Counter
import boto
from boto.s3.connection import S3Connection
from boto.s3.key import Key

sys.setrecursionlimit(2 ** 31 -1)

PUNCTUATION = set(string.punctuation)
STOPWORDS = set(stopwords.words('english'))


class TfToken(object):
    """
    Preprocessing data in S3 into rdd, tfidf vectors

    INPUT:
    - sc: pyspark.SparkContext
    - aws_link: aws link for your data, eg: jyt109/wiki_articles
    - tokenizer: function object to tokenize each line in spark rdd
    - filename: file with key pair info (optional if keypair info stored in bash_profile)
    """
    def __init__(self, sc, aws_link, tokenizer, access_key, secret_access_key, numPartition=200):
        self.access_key = access_key
        self.secret_access_key = secret_access_key
        self.aws_link = aws_link
        self.link = None
        self.sc = sc
        self.token_rdd = None
        self.filename = filename
        self.tokenizer = tokenizer
        self.rdd = None
        self.num_partition = numPartition

    def fit(self):
        """
        OUTPUT:
        - article rdd and tfidf matrix rdd in sparse vector format for all files
        """
        self.load_keys()
        self.create_link(self.aws_link)
        return self.tfidf(self.tokenizer)

    def create_link(self, aws_link):
        """
        Create aws link with keypairs
        """
        self.link = 's3n://%s:%s@%s' % (self.access_key, self.secret_access_key, self.aws_link)

    def _create_rdd(self, tokenizer):
        """
        Create rdd for the given aws link
        Remove lines with #REDIRECT (for wikipedia dataset)
        Tokenize data
        """
        self.rdd = self.sc.textFile(self.link, self.num_partition)
        # take a subsample of wikipedia pages
        # self.rdd = self.sc.parallelize(self.rdd.take(600), 24)
        self.rdd = self.rdd.filter(lambda line: '#REDIRECT' not in line)
        self.token_rdd = self.rdd.map(tokenizer)

    def tfidf(self, tokenizer):
        """
        Get TFIDF matrix rdd with spark tfidf functions
        """
        self._create_rdd(tokenizer)
        hashingTF = HashingTF()
        tf = hashingTF.transform(self.token_rdd)
        idf = IDF(minDocFreq=2).fit(tf)
        tfidf = idf.transform(tf)
        return self.rdd, idf, tfidf


def tokenizing(text):
    """
    Tokenize a single article to english words with a stemmer
    """
    regex = re.compile('<.+?>|[^a-zA-Z]')
    clean_txt = regex.sub(' ', text)
    tokens = clean_txt.split()
    lowercased = [t.lower() for t in tokens]

    no_punctuation = []
    for word in lowercased:
        punct_removed = ''.join([letter for letter in word if not letter in PUNCTUATION])
        no_punctuation.append(punct_removed)
    no_stopwords = [w for w in no_punctuation if not w in STOPWORDS]

    # STEMMER = PorterStemmer()
    # stemmed = [STEMMER.stem(w) for w in no_stopwords]
    return [w for w in no_stopwords if w]


def get_title_link(article):
    """
    Use regular expression to get title and all links in each wikipage
    """
    try:
        links = re.findall(r'\[\[(.*?)[\]\]|\|]', article)
        title = re.search(r'\'\'\'(.*?)\'\'\'', article).group(1)
        return title, "|".join(links)
    except:
        return "", [""]


def get_title(article):
    """
    Use regular expression to get title in each wikipage
    """
    try:
        title = re.search(r'\'\'\'(.*?)\'\'\'', article).group(1)
        return title
    except:
        return " "


def get_title_tfidf(title_string):
    """
    Flat map related links in each grouped title
    eg: title has multiple meanings
    """
    titles = title_string.split("|")
    for title in titles:
        yield title


def cosine_sim(v1, v2, origin):
    """
    Calcuate cosine similarity for sparse vectors based on functions in mllib

    INPUT: two vectors and origin
    OUTPUT: cosine similarity
    """
    try:
        return v1.dot(v2) / (v1.squared_distance(origin) * v2.squared_distance(origin))
    except:
        return v1.dot(v2) / (v1.squared_distance(origin) * v2.squared_distance(origin) + 1)

def max_cosine_sim(related_tfidf, tf_category):
    """
    Calcuate cosine similarities between user input category and all possible pages for the same keyword
    Save the cosine similarity matrix to file for checking

    INPUT: tfidf for all related pages, tfidf for input category
    OUTPUT: title with highest cosine similarity with the category
    """
    num_cols = len(tf_category)
    # initilize a 0 sparse vector to calcuate norm+
    origin = SparseVector(num_cols, {})
    title_cos_sim = np.array([[title, cosine_sim(vector, tf_category, origin)] for title, vector in related_tfidf])
    np.savetxt("cos_sim.csv", title_cos_sim, delimiter=",")
    return title_cos_sim[np.argmax(title_cos_sim[:,1]),0]


def get_most_similiar_ariticle(idf, keyword, category, multi_links, title_tfidf):
    """
    Find most related article based one keyword and category

    INPUT:
    - idf: to transform category string
    - keyword: to get all relevent links in that page
    - category: to find the highest cosine similarity, since all pages are very relevent to keyword, it's better to only use category
    - multi_links: all pages with multiple links with similar keyword
    - title_tfidf: tuple rdd of (title, tfidf) for all wikipages and used to find the tfidf for any given title

    OUTPUT:
    - most related title
    - tfidf for most related title
    """
    tf_category = transform(idf, category)
    related_links = multi_links.map(get_title_link).filter(lambda x: x[0]==keyword).map(lambda x: x[1]).first().split("|")
    # related_tfidf = title_tfidf.take(3)
    related_tfidf = title_tfidf.filter(lambda x: x[0] in related_links).collect()
    most_related_title = max_cosine_sim(related_tfidf,  tf_category)
    most_related_tfidf = title_tfidf.filter(lambda x: x[0]==most_related_title).map(lambda x: x[1]).collect()[0]
    return most_related_title, most_related_tfidf


def same_topic(category, most_related_tfidf, idf, topic_model):
    """
    Check if most related title and input category are in the same topic group from model prediction

    INPUT:
    - user input category
    - tfidf for most related link
    - als model after training with all wiki pages

    OUTPUT:
    - Boolean: If category and the most relavent article are in the same topic group
    """
    category_tfidf = transform(idf, category)
    category_topic = topic_model.predict(category_tfidf)
    article_topic = topic_model.predict(most_related_tfidf)
    if category_topic == article_topic:
        return True
    else:
        return False


def train_model(rdd, idf, tfidf):
    """
    Train clusters with als model in mllib

    INPUT:
    - rdd: rdd of all raw wiki pages
    - idf: transformer for test strings
    - tfidf: vectorized rdd for all wiki pages

    OUTPUT:
    - multi_links: keyword may have multiple meanings
    - title_tfidf: tuple rdd of title and corresponding tfidf
    - topic_model: als model after training with all wiki pages
    """
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


def convert_rating(sparse_vector_and_index):
    """
    INPUT:
    - sparse_vector and index combined
    OUTPUT:
    - tuple of (user, movie, rating) format
    """
    sparse_vector, index = sparse_vector_and_index
    for key, value in zip(sparse_vector.indices, sparse_vector.values):
        if value != 0:
            yield (index, key, value)


def transform(idf, article):
    """
    transform article to a sparse vector
    """
    token = tokenizing(article)
    hashingTF = HashingTF()
    tf_test = hashingTF.transform(token)
    return idf.transform(tf_test)


class TopicModel(object):
    """
    INPUT: idf transformer, tfidf sparse matrix
    OUTPUT: topic prediction given article
    """
    def __init__(self, idf, tfidf):
        """
        Initialize sparse matrix and article to vector transformer
        """
        self.idf = idf
        self.tfidf = tfidf
        self.model = None
        self.tfidf_rating = None
        self.train_data = None

    def preprocessing(self, method=convert_rating):
        """
        Convert tfidf matrix to tuples of (user, movie, rating) format
        """
        self.tfidf_rating = self.tfidf.zipWithIndex().flatMap(method).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))


    def label(self, rank=50, numIterations=10, alpha=0.01):
        """
        INPUT:
        - rank: number of topics
        - numIterations: number of iterations for matrix factorization
        - alpha: learning rate
        OUTPUT:
        - data for training naive bayes with label, feature tuples
        """
        als_model = ALS.trainImplicit(self.tfidf_rating, rank, numIterations, alpha)
        index_label = als_model.userFeatures().map(lambda x: (x[0], np.argmax(x[1])))
        index_feature = self.tfidf.zipWithIndex().map(lambda x: (x[1], x[0]))
        index_label_feature = index_label.join(index_feature)
        label_feature = index_label_feature.map(lambda x: x[1])
        self.train_data = label_feature.map(lambda x: LabeledPoint(x[0], x[1]))

    def train(self, score=False):
        """
        Train NaiveBayes model
        """
        self.label()
        self.model = NaiveBayes.train(self.train_data, 1.0)
        if score:
            training, test = self.train_data.randomSplit([0.6, 0.4], seed=0)
            predictionAndLabel = test.map(lambda p: (self.model.predict(p.features), p.label))
            accuracy = 1.0 * predictionAndLabel.filter(lambda (x, v): x == v).count() / test.count()
            print "accuracy: ", accuracy

    def predict(self, tfidf_test):
        """
        Predict topic based on any string
        """
        return self.model.predict(tfidf_test)


def keypairs(filename=None):
    """
    load key pairs from bash_profile or \
    manually create a json file with:

    cat aws.json
    {"ACCESS_KEY": "ACCESS_KEY", "SECRET_ACCESS_KEY": "SECRET_ACCESS_KEY"}
    Use json.load to load keys

    filename is optional if key is stored in bash_profile or bashrc
    """
    try:
        access_key = os.environ['AWS_ACCESS_KEY_ID']
        secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']
    except:
        with open(filename) as f:
            data = json.load(f)
            access_key = data['ACCESS_KEY']
            secret_access_key = data['SECRET_ACCESS_KEY']
    return access_key, secret_access_key


def save_rdd_to_file(rdd, key_name, access_key, secret_access_key):
    """
    connect to s3 and save rdd file to s3
    """
    conn = S3Connection(access_key, secret_access_key)
    bucket_name = 'wiki-search'

    if conn.lookup(bucket_name) is None:
        b = conn.create_bucket(bucket_name)
    else:
        b = conn.get_bucket(bucket_name)

    k = Key(b)
    k.key = key_name

    rdd.saveAsTextFile('s3://%s:%s@%s/%s' % (access_key, secret_access_key, bucket_name, key_name))


def main():
    """
    For now, only testing with keyword and category separately, they should be user input from web app
    """
    keyword = "recall"
    category = "statistics math math"

    sc = ps.SparkContext()
    aws_link = "jyt109/wiki_articles"

    access_key, secret_access_key = keypairs("../keypair.json")

    # preprocessing S3 data into rdd, tfidf vectors and save the transformer
    tf_token = TfToken(sc=sc, aws_link=aws_link, tokenizer=tokenizing, access_key, secret_access_key)
    # Trying to find ways to save idf model, pickle doesn't work
    rdd, idf, tfidf = tf_token.fit()

    save_rdd_to_file(rdd, 'rdd123', access_key, secret_access_key)
    save_rdd_to_file(tfidf, 'tfidf', access_key, secret_access_key)


    # train model with matrix factorization
    # will save rdds multi_links, title_tfidf to s3 later, but need help saving model
    multi_links, title_tfidf, topic_model = train_model(rdd, idf, tfidf)

    try:
        # check keyword in title_tfidf for all pages with multiple meanings
        if title_tfidf.filter(lambda x: x[0]==keyword).collect():
            most_related_title, most_related_tfidf = get_most_similiar_ariticle(idf, keyword, category, multi_links, title_tfidf)
            # if category and the most related article are in the same cluster
            if same_topic(category, most_related_tfidf, idf, topic_model):
                # connect strings with "_" to be put in urls later
                return_title = "_".join(most_related_title)
            fw = "tfidf done, train_model done %s" % most_related_title
            f.write(fw)
        # if keyword not in the page with multiple meanings, which means keyword in other title which has unique meaning
        elif rdd.map(get_title).filter(lambda x: x==keyword).collect():
            return_title = "_".join(keyword)
        # no keyword found in all titles, use search function in wikipedia
        else:
            return_title = "Special:Search?search=%s&go=Go" % ("+".join(keyword.split()))
    # if there is an error or nothing applies, use search function in wikipedia
    except:
        return_title = "Special:Search?search=%s&go=Go" % ("+".join(keyword.split()))

    # redirect_link is ready for flask app
    redirect_link = "https://www.wikipedia.org/wiki/" + return_title

if __name__ == '__main__':
    main()
