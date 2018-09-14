from nltk.corpus import movie_reviews
from random import shuffle
from nltk import FreqDist
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize

#no of reviews & types of reviews
print(len(movie_reviews.fileids()))
print(movie_reviews.categories())
"""
docs=[]
for category in movie_reviews.categories():
	for fileid in movie_reviews.fileids(category):
		docs.append(list(movie_reviews.words(fileid),category))
"""
docs=[(list(movie_reviews.words(fileid)),category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]

shuffle(docs)
print(len(docs))

allwords=[]
for w in movie_reviews.words():
	allwords.append(w.lower())

#print(allwords[0:100])
print(len(allwords))#total numbet of words present in the file(movie_review)

freq=FreqDist(allwords)
#most common 10 words and their frequency
#print(freq.most_common(10))

#as the words like a,the,in(called as stopwords) cant be useful in model building they should be removed.Same is the case with punctuation marks.
#to remove stopwords
stopwords=set(stopwords.words("english"))
sw=[word for word in allwords if word not in stopwords]
#print(sw[:10])
print(len(sw))

#to remove punctuation
import string
swp= [word for word in sw if word not in string.punctuation]
#print(swp[:10])
print(len(swp))
#the above swp list is the cleaned list which shall be further used for model building

#frequency distributon of the above cleaned list would be needed to create a 
#TOP N-WORDS model

swpf=FreqDist(swp)
#print(swpf.most_common(10))

word_features=list(swpf.keys())[:2000]

print(word_features[0:10])
def find_features(document):
	words=set(document)
	features={}
	for w in word_features:
		features[w]=(w in words)
	return features


featuresets=[(find_features(rev),category) for rev,category in docs]

#print(featuresets)
training=featuresets[:1900]
testing=featuresets[1900:]

#using nltk's naive bayes classifier
clf=nltk.NaiveBayesClassifier.train(training)
print("NB classifier:",nltk.classify.accuracy(clf,testing))
#clf.show_most_informative_features(15)

#print(clf.classify(testing[0][0]))

new = "The movie was very bad.The acting skills were the worst an the storyline was very bad."
newtokens = word_tokenize(new)
xx = find_features(newtokens)
print (clf.classify(xx))

new1 = "The movie was fantastic and icluded mind blowing stunts,the story line of the moviw was superb."
newtokens = word_tokenize(new1)
xx = find_features(newtokens)
print (clf.classify(xx))

#here the sentiment for postive review is predicted negative hence the second code file has been included to correctly predict the sentiment of the statement/review.



