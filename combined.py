import sys
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

sys.setrecursionlimit(2 ** 31 -1)

PUNCTUATION = set(string.punctuation)
STOPWORDS = set(stopwords.words('english'))


class TfToken(object):
    """
    INPUT:
    - sc: pyspark.SparkContext
    - aws_link: aws link for your data, eg: jyt109/wiki_articles
    - tokenizer: function object to tokenize each line in spark rdd
    - filename: file with key pair info (optional if keypair info stored in bash_profile)
    """
    def __init__(self, sc, aws_link, tokenizer, filename=None):
        self.access_key = None
        self.secret_access_key = None
        self.aws_link = aws_link
        self.link = None
        self.sc = sc
        self.token_rdd = None
        self.filename = filename
        self.tokenizer = tokenizer
        self.rdd = None

    def fit(self):
        """
        OUTPUT:
        - article rdd and tfidf matrix rdd in sparse vector format for all files
        """
        self.load_keys()
        self.create_link(self.aws_link)
        return self.tfidf(self.tokenizer)

    def load_keys(self):
        """
        load key pairs from bash_profile or \
        manually create a json file with:

        cat aws.json
        {"ACCESS_KEY": "ACCESS_KEY", "SECRET_ACCESS_KEY": "SECRET_ACCESS_KEY"}
        Use json.load to load keys

        filename is optional if key is stored in bash_profile or bashrc
        """
        try:
            self.access_key = os.environ['AWS_ACCESS_KEY_ID']
            self.secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']
        except:
            with open(self.filename) as f:
                data = json.load(f)
                self.access_key = data['ACCESS_KEY']
                self.secret_access_key = data['SECRET_ACCESS_KEY']

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
        self.rdd = self.sc.textFile(self.link)
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

    STEMMER = PorterStemmer()
    stemmed = [STEMMER.stem(w) for w in no_stopwords]
    return [w for w in stemmed if w]


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


    def label(self, rank=3, numIterations=10, alpha=0.01):
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


if __name__ == '__main__':

    keyword = "recall"
    category = "statistics math math"

    sc = ps.SparkContext()
    aws_link = "jyt109/wiki_articles"
    tf_token = TfToken(sc=sc, aws_link=aws_link, tokenizer=tokenizing, filename="../keypair.json")
    rdd, idf, tfidf = tf_token.fit()
    multi_links, title_tfidf, topic_model = train_model(rdd, idf, tfidf)
    most_related_title, most_related_tfidf = get_most_similiar_ariticle(idf, keyword, category, multi_links, title_tfidf)
    if same_topic(category, most_related_tfidf, idf, topic_model):
        return_title = most_related_title
    print most_related_title
