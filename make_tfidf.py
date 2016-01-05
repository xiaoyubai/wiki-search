import os
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords, words, wordnet
import string
from collections import Counter
from pyspark.mllib.clustering import KMeans
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF

class TfToken(object):
    def __init__(self, sc):
        self.access_key = None
        self.secret_access_key = None
        self.link = None
        self.sc = sc
        self.token_rdd = None


    def load_keys(self, filename=None):
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
            with open(filename) as f:
                data = json.load(f)
                self.access_key = data['ACCESS_KEY']
                self.secret_access_key = data['SECRET_ACCESS_KEY']

        def create_link(self, aws_link):
            self.link = 's3n://%s:%s@%s' % (self.access_key, self.secret_access_key, aws_link)

    def _tokenizing(self, text):
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

    def _create_rdd(self):
        wiki_rdd = self.sc.textFile(self.link)
        wiki_no_redirect_rdd = wiki_rdd.filter(lambda line: '#REDIRECT' not in line)
        self.token_rdd = wiki_no_redirect_rdd.map(self._tokenizing)

    def tfidf(self):
        self._create_rdd()
        hashingTF = HashingTF()
        tf = hashingTF.transform(self.token_rdd)
        tf.cache()
        idf = IDF(minDocFreq=2).fit(tf)
        tfidf = idf.transform(tf)
        return tfidf
