import nltk
import random
from nltk.corpus import movie_reviews

docs=[(list(movie_reviews.words(fileid)),category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]
"""
docs=[]
for category in movie_reviews.categories():
	for fileid in movie_reviews.fileids(category):
		docs.append(list(movie_reviews.words(fileid),category))
"""

random.shuffle(docs)

#print(docs[1])

all_words=[]
for w in movie_reviews.words():
	all_words.append(w.lower())


all_words=nltk.FreqDist(all_words)
#15 most common words
#print(all_words.most_common(15))
#no of times word "stupid appears"
#print(all_words["stupid"])

word_features=list(all_words.keys())[:3000]

def find_features(docs):
	words=set(docs)
	features={}
	for w in word_features:
		features[w]=(w in words)
	return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
featuresets=[(find_features(rev),category) for rev,category in docs]

#print(featuresets)
training=featuresets[:1900]
testing=featuresets[1900:]

#using nltk's naive bayes classifier
clf=nltk.NaiveBayesClassifier.train(training)
print("original NB classifier:",nltk.classify.accuracy(clf,testing))
clf.show_most_informative_features(15)

#to use scikit learn algorithms following statement is used
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

#accuracy with different classifiers is calculated
#using Multinomial NB Classifier
MNB=SklearnClassifier(MultinomialNB())
MNB.train(training)
print("MNB classifier:",nltk.classify.accuracy(MNB,testing))

#using Multinomial NB Classifier
BC=SklearnClassifier(BernoulliNB())
BC.train(training)
print("BNBclassifier:",nltk.classify.accuracy(BC,testing))

#using Logistic Regression Classifier
LRC=SklearnClassifier(LogisticRegression())
LRC.train(training)
print("LRclassifier:",nltk.classify.accuracy(LRC,testing))

#using Support vector machine classifier
SVMC=SklearnClassifier(SVC())
SVMC.train(training)
print("SVMclassifier:",nltk.classify.accuracy(SVMC,testing))




