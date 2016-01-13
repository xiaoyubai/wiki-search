#! /usr/bin python27

import os
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords, words, wordnet
import string
from collections import Counter
from pyspark.mllib.clustering import KMeans
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
import json
import string
import pyspark as ps

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
        self.rdd = self.sc.parallelize(self.rdd.take(600), 24)
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

if __name__ == '__main__':
    sc = ps.SparkContext()
    aws_link = "wikisample10/sample2"
    tf_token = TfToken(sc=sc, aws_link=aws_link, tokenizer=tokenizing, filename="../keypair.json")
    rdd, idf, tfidf = tf_token.fit()
    print tfidf.take(2)
