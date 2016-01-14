from pyspark.mllib.recommendation import ALS, Rating
from pyspark.mllib.linalg import Vectors, SparseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
import pyspark as ps
import numpy as np

from make_tfidf import TfToken, tokenizing


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
    sc = ps.SparkContext('local[4]')
    aws_link = "wikisample10/sample2"
    tf_token = TfToken(sc=sc, aws_link=aws_link, tokenizer=tokenizing, filename="../keypair.json")
    rdd, idf, tfidf = tf_token.fit()
    rdd.cache()
    tfidf.cache()

    topic_model = TopicModel(idf=idf, tfidf=tfidf)
    topic_model.preprocessing()
    topic_model.label()
    topic_model.train()
    category = 'math math math math statistics'
    tfidf_test = transform(idf, category)
    topic_model.predict(tfidf_test)
