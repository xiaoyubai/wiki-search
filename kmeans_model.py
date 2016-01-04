import pandas as pd
import numpy as np
import pyspark as ps
import os
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
from collections import Counter
from pyspark.mllib.clustering import KMeans
PUNCTUATION = set(string.punctuation)
STOPWORDS = set(stopwords.words('english'))

NUMBER_CLUSTERS = 10

def load_keys(filename):
    """
    create a json file with:
    cat aws.json
    {"ACCESS_KEY": "ACCESS_KEY", "SECRET_ACCESS_KEY": "SECRET_ACCESS_KEY"}
    Use json.load to load keys
    """
    with open(filename) as f:
        data = json.load(f)
        ACCESS_KEY = data['ACCESS_KEY']
        SECRET_ACCESS_KEY = data['SECRET_ACCESS_KEY']
    return ACCESS_KEY, SECRET_ACCESS_KEY

def get_content(article):
    try:
        title = re.search(r'\'\'\'([\w+\s]+)\'\'\'', article).group(1)
        return [title, article]
    except:
        return

# def get_rdd():
#     sc = ps.SparkContext('local[4]')
#     link = 's3n://%s:%s@wikisample10/sample2' % (ACCESS_KEY, SECRET_ACCESS_KEY)
#     wiki = sc.textFile(link)
#     return wiki

# def pre_processing(articles, model=None):
#     tokenizer = RegexpTokenizer(r'\w+')
#     wordnet = WordNetLemmatizer()
#     word_tokens = [tokenizer.tokenize(article.lower()) for article in articles]
#     word_stem = [" ".join([wordnet.lemmatize(word) for word in row]) for row in word_tokens]
#     if model:
#         mat = model.transform(word_stem).toarray()
#         return mat
#     else:
#         model = TfidfVectorizer(stop_words='english', tokenizer=word_tokenize, decode_error='ignore')
#         mat = model.fit_transform(word_stem).toarray()
#         return mat, model

def tokenize(text):
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

def get_tf(word_lst):
    count_of_each_word = Counter(word_lst)
    doc_word_count = len(word_lst) * 1.
    term_freq = np.array([count_of_each_word[v] / doc_word_count if v in count_of_each_word else 0 for v in vocab])
    return term_freq


def kmeans_label(mat, scoring=False):
    kmeans_model = KMeans(n_clusters=NUMBER_CLUSTERS, random_state=1).fit(mat)
    labels = kmeans_model.labels_
    if scoring:
        print "kmeans score: ", silhouette_score(tif_mat, labels, metric='euclidean')
    return labels


if __name__ == '__main__':
    first_n_lines = 2000
    redirect_str = '^#REDIRECT'
    sc = ps.SparkContext('local[4]')
    try:
        ACCESS_KEY = os.environ['AWS_ACCESS_KEY_ID']
        SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY']
    except:
        ACCESS_KEY, SECRET_ACCESS_KEY = load_keys('../aws.json')
    link = 's3n://%s:%s@wikisample10/sample2' % (ACCESS_KEY, SECRET_ACCESS_KEY)
    wiki_rdd = sc.textFile(link)
    wiki_rdd_samples = sc.parallelize(wiki_rdd.take(first_n_lines), 5)
    wiki_no_redirect_rdd = wiki_rdd_samples.filter(lambda line: '#REDIRECT' not in line)
    token_rdd = wiki_no_redirect_rdd.map(tokenize)
    vocab = token_rdd.flatMap(lambda x: x).distinct().collect()
    tf_rdd = token_rdd.map(get_tf)
    total_doc_count = tf_rdd.count()
    times_words_in_doc = tf_rdd.map(lambda tf_lst: ((np.array(tf_lst) > 0) + 0)).sum()
    idf = np.log(total_doc_count / times_words_in_doc)
    tfidf_rdd = tf_rdd.map(lambda tf_vec: tf_vec * idf)
    tfidf_rdd.cache()
    tfidf_rdd.setName('tfidf')
