from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import SGDClassifier
import numpy as np
from twenty_newsgroups import load_20newsgroups
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD

# all_dataset = load_20newsgroups(data_home='./test', subset='all')
# training_dataset = load_20newsgroups(data_home='./test', subset='train')
# test_dataset = load_20newsgroups(data_home='./WI2020_Data', subset='test')

#### word cloud
categories = ['happy', 'sad']
for category in categories:
    ## categories must be a list in here
    train_dataset1 = load_20newsgroups(data_home='./test', subset='train', categories=[category])
    string = train_dataset1.data.__str__()
    # print(string)
    wordcloud = WordCloud().generate(string)
    # wordcloud.to_file(category + ".png")
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
# dataset = load_20newsgroups(data_home='./WI2020_Data', subset='all', categories=['sci.med'])