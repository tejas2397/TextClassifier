from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
import nltk
from random import shuffle
from nltk import FreqDist
from nltk.corpus import stopwords
import string

posrev=[]
for fileid in movie_reviews.fileids('pos'):
	words=movie_reviews.words(fileid)
	posrev.append(words)

negrev=[]
for fileid in movie_reviews.fileids('neg'):
	words=movie_reviews.words(fileid)
	negrev.append(words)

#print(len(posrev))
#print(len(negrev))


#cleaning the data that is removing the stopwords and punctuation marks

stopwords=stopwords.words('english')

def bagofwords(words):
	cleandata=[]
	for word in words:
		word=word.lower()
		if word not in stopwords and word not in string.punctuation:
			cleandata.append(word)
	
	wordict=dict([word,True] for word in cleandata)
	return wordict

posfeatureset=[]
for words in posrev:
	posfeatureset.append((bagofwords(words),'pos'))

negfeatureset=[]
for words in negrev:
	negfeatureset.append((bagofwords(words),'neg'))

shuffle(posfeatureset)
shuffle(negfeatureset)

#now create training and testing data
training=posfeatureset[:900]+negfeatureset[:900]
testing=posfeatureset[900:]+negfeatureset[900:]

print(len(training))
print(len(testing))

#using nltk's naive bayes classifier
clf=nltk.NaiveBayesClassifier.train(training)
print("NB classifier:",nltk.classify.accuracy(clf,testing))
#clf.show_most_informative_features(15)

#print(clf.classify(testing[0][0]))

new = "The movie was very bad.The acting skills were the worst an the storyline was very bad."
newtokens = word_tokenize(new)
xx = bagofwords(newtokens)
print (clf.classify(xx))

new1 = "The movie was fantastic and icluded mind blowing stunts,the story line of the moviw was superb."
newtokens = word_tokenize(new1)
xx = bagofwords(newtokens)
print (clf.classify(xx))

new2=raw_input("enter a review")
newtokens = word_tokenize(new2)
xx = bagofwords(newtokens)
print (clf.classify(xx))

