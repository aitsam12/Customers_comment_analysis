## SKLEARN
import random
class Sentiment:
    NEGATIVE="NEGATIVE"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"

class Review:
    def __init__(self,text,score):
        self.text = text
        self.score = score
        self.sentiment = self.get_sentiment()

    def get_sentiment(self):
        if self.score <=2:
            return Sentiment.NEGATIVE
        elif self.score ==3:
            return Sentiment.NEUTRAL
        else: #score 4 and 5
            return Sentiment.POSITIVE
class ReviewContainer:
    def __init__(self, reviews):
        self.reviews = reviews
    def get_text(self):
        return [x.text for x in self.reviews]
    def get_sentiment(self):
        return [y.sentiment for y in self.reviews]
    def evenly_distribute(self):
        negative = list(filter(lambda x: x.sentiment == Sentiment.NEGATIVE, self.reviews))
        positive = list(filter(lambda x: x.sentiment == Sentiment.POSITIVE, self.reviews))
        positive_shrunk = positive[:len(negative)]
        self.reviews = negative + positive_shrunk
        random.shuffle(self.reviews)



##prep data
import json
from sklearn.model_selection import train_test_split


file_name = 'Books_small_10000.json'
reviews = []
with open (file_name) as f:
    for line in f:
        review = json.loads(line)
        reviews.append(Review(review['reviewText'], review['overall']))
#print(reviews[5].text)

training, test = train_test_split(reviews, test_size=0.33, random_state=42)

train_container = ReviewContainer(training)
test_container = ReviewContainer(test)
#print(training[0].sentiment)
train_container.evenly_distribute()
#print(len(train_container.reviews))
train_x = train_container.get_text()
train_y = train_container.get_sentiment()

test_container.evenly_distribute()
test_x = test_container.get_text()
test_y = test_container.get_sentiment()
#print(train_y.count(Sentiment.POSITIVE))
#print(train_y.count(Sentiment.NEGATIVE))

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
vectorizer = TfidfVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)
test_x_vectors = vectorizer.transform(test_x)


#print(train_x_vectors[0].toarray())

###CLASSIFICATION

#svm
from sklearn import svm

#clf_svm = svm.SVC(kernel = 'linear')
#clf_svm.fit(train_x_vectors, train_y)
#print(test_x[0])
#print(clf_svm.predict(test_x_vectors[0]))

'''
#decision tree
from sklearn.tree import DecisionTreeClassifier

clf_dec = DecisionTreeClassifier()
clf_dec.fit(train_x_vectors, train_y)

#print(clf_dec.predict(test_x_vectors[0]))

#naive bayes
#from sklearn.naive_bayes import GaussianNB

#clf_nb = GaussianNB()
#clf_nb.fit(train_x_vectors, train_y)

#print(clf_nb.predict(test_x_vectors[0]))

#logistic regression
from sklearn.linear_model import LogisticRegression

clf_log = LogisticRegression()
clf_log.fit(train_x_vectors, train_y)

#print(clf_log.predict(test_x_vectors[0]))
'''
###EVALUATION

#mean accuracy
#print(clf_svm.score(test_x_vectors,test_y))
#print(clf_dec.score(test_x_vectors,test_y))
#print(clf_log.score(test_x_vectors,test_y))

#F1 score
from sklearn.metrics import f1_score

#print(f1_score(test_y, clf_svm.predict(test_x_vectors), average =None, labels = [Sentiment.POSITIVE, Sentiment.NEUTRAL, Sentiment.NEGATIVE]))
#print(f1_score(test_y, clf_dec.predict(test_x_vectors), average =None, labels = [Sentiment.POSITIVE, Sentiment.NEUTRAL, Sentiment.NEGATIVE]))
#print(f1_score(test_y, clf_log.predict(test_x_vectors), average =None, labels = [Sentiment.POSITIVE, Sentiment.NEUTRAL, Sentiment.NEGATIVE]))

#print(train_y.count(Sentiment.POSITIVE))

test_set = ['I thoroughly enjoyed this 5 stars', 'bad book, donot buy', 'it is very good book, I really liked it']
new_test = vectorizer.transform(test_set)

#print(clf_svm.predict(new_test))


## to make the model more better
from sklearn.model_selection import GridSearchCV

parameters = {'kernel': ('linear', 'rbf'), 'C': (1,4,8,16,32)}

svc = svm.SVC()
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(train_x_vectors, train_y)
#print(clf.fit(train_x_vectors, train_y))
clf.score(test_x_vectors,test_y)
#print(clf.score(test_x_vectors,test_y))

### SAVING MODEL

import pickle

with open('sentiment_classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)


with open('sentiment_classifier.pkl', 'rb') as f:
   loaded_clf = pickle.load(f)

print(test_x[10])
print(loaded_clf.predict(test_x_vectors[10]))