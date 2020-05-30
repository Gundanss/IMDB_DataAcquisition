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

## question A
## load the training data and all the data
all_dataset = load_20newsgroups(data_home='./WI2020_Data', subset='all')
training_dataset = load_20newsgroups(data_home='./WI2020_Data', subset='train')
test_dataset = load_20newsgroups(data_home='./WI2020_Data', subset='test')
# print(training_dataset.target)
# list out all the categories name in the dataset
print('all_dataset.target_names:', all_dataset.target_names)
print('training_dataset.target_names:', training_dataset.target_names)
print('')

# print('Size of dataset.data: %d' % len(all_dataset.data))
# print('Size of training_dataset.data: %d' % len(training_dataset.data))
print('')
# for i in range(3):
#     print('Doc Number %d' % i)
#     print('Target Index: %d' % all_dataset.target[i]) ## ??
#     print('Doc Type: %s' % all_dataset.target_names[all_dataset.target[i]])
#     print(all_dataset.data[i])
#     print('')

## question b
## compute tf-idf for the training data
count_vect = CountVectorizer()
bow_train_counts = count_vect.fit_transform(training_dataset.data)
bow_test_counts = count_vect.transform(test_dataset.data)
# print('Number of documents in twenty_train.data:', len(all_dataset.data))
# print('Number of extracted features:', len(count_vect.get_feature_names()))
# print('Size of bag-of-words:', bow_train_counts.shape)
# print('Bag of words:' + '\n' + '[(doc_id, features_id): Occurrence] ')
# print(bow_train_counts)
# print(count_vect.get_feature_names())
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(bow_train_counts)
print(type(X_train_tfidf))
print("tfidf table of training data:")
print(X_train_tfidf)

# # train NB classifier
# clf = MultinomialNB().fit(X_train_tfidf, training_dataset.target)
#
# # Prepare the testing data set
# test_dataset = load_20newsgroups(data_home='./WI2020_Data', subset='test')
# X_test_counts = count_vect.transform(test_dataset.data)
# X_test_tfidf = tfidf_transformer.transform(X_test_counts)
#
# # use the trained classifier to predict results for testing data set
# predicted = clf.predict(X_test_tfidf)
#
# for doc, category in zip(training_dataset.data, predicted):
#    print('Classified as: %s\n%s\n' % (training_dataset.target_names[category], doc))

## Another method to use NB classifier to compute
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])
text_clf.fit(training_dataset.data, training_dataset.target)
docs_test = test_dataset.data
predicted = text_clf.predict(docs_test)
print("NB classifier accuracy:", np.mean(predicted == test_dataset.target))

## question c
## use SVM to compute
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)),
])
text_clf.fit(training_dataset.data, training_dataset.target)
predicted = text_clf.predict(docs_test)
print("SVM classifier accuracy:", np.mean(predicted == test_dataset.target))


## question d
print('')
print("confusion matrix:")
print(metrics.confusion_matrix(test_dataset.target, predicted))
print('')
print(metrics.classification_report(test_dataset.target, predicted, target_names=test_dataset.target_names))


## question e
########### truncated SVD
for n in (5, 10, 20, 50, 100):
    svd = TruncatedSVD(n_components=n, n_iter=25, random_state=12)
    X_train = svd.fit_transform(bow_train_counts)
    X_test = svd.transform(bow_test_counts)
    # clf = LogisticRegression().fit(X_train, training_dataset.target)
    # predicted = clf.predict(X_test)
    # print("Truncated SVD in", n, "dimension accuracy: ", np.mean(predicted == test_dataset.target))
    print(n, "dimension")
    print(X_train)
    print('')

## question f
### use SVD to train a SVM
list = []
for n in (5, 10, 20, 50, 100):
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('svd', TruncatedSVD(n_components=n, n_iter=25, random_state=12)),
        ('clf', SGDClassifier(loss='hinge', penalty='l2',
                              alpha=1e-3, random_state=42,
                              max_iter=5, tol=None)),
    ])
    text_clf.fit(training_dataset.data, training_dataset.target)
    predicted = text_clf.predict(docs_test)
    list.append(np.mean(predicted == test_dataset.target))
    print("SVM classifier in", n, "dimension accuracy:", np.mean(predicted == test_dataset.target))

## question G
# print(list)
x = ['5', '10', '20', '50', '100']
y_pos = np.arange(len(x))
plt.bar(y_pos, list, align='center', alpha=0.5)
plt.xticks(y_pos, x)
plt.ylabel('Accuracy')
plt.title('Accuracy with SVD in different dimensions')
plt.show()

## question h
#### word cloud
categories = ['alt.atheism', 'comp.sys.mac.hardware', 'rec.sport.baseball', 'sci.med']
for category in categories:
    ## categories must be a list in here
    all_dataset1 = load_20newsgroups(data_home='./WI2020_Data', subset='all', categories=[category])
    string = all_dataset1.data.__str__()
    # print(string)
    wordcloud = WordCloud().generate(string)
    wordcloud.to_file(category + ".png")
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
# dataset = load_20newsgroups(data_home='./WI2020_Data', subset='all', categories=['sci.med'])




