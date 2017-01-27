from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import metrics
import numpy
import nltk


training = load_files("/home/vinothkumar/AI/Selected20NewsGroup/Training",description=None, categories=None, load_content=True,shuffle=True, encoding="latin1",decode_error='strict', random_state=42)

#print("\n".join(training.data[1].split("\n")[:3]))

def removeHeaders(trainingData):
	for i in range(0, len(trainingData)):
		header, body = trainingData[i].split("\n\n",1);
		trainingData[i] = body

removeHeaders(training.data)

#print("\n".join(training.data[1].split("\n")[:3]))

#vectorizer = CountVectorizer()
vectorizer = CountVectorizer(stop_words = "english")
trainedVector = vectorizer.fit_transform(training.data)

tfidf_transformer = TfidfTransformer()
transformedVector = tfidf_transformer.fit_transform(trainedVector)
#transformedVector = trainedVector

classifier = svm.LinearSVC(penalty = "l2", dual = False).fit(transformedVector, training.target)
#classifier = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#    decision_function_shape=None, degree=2, gamma='auto', kernel='rbf',
#    max_iter=-1, probability=False, random_state=None, shrinking=True,
#    tol=0.001, verbose=False).fit(transformedVector, training.target)
testing = load_files("/home/vinothkumar/AI/Selected20NewsGroup/Test",description=None, categories=None, load_content=True,shuffle=True, encoding="latin1",decode_error='strict', random_state=42)

removeHeaders(testing.data)
testVector = vectorizer.transform(testing.data)
transformedTestVector = tfidf_transformer.transform(testVector)
predicted = classifier.predict(transformedTestVector)

print(metrics.classification_report(testing.target, predicted,target_names=testing.target_names).split("\n")[7])

