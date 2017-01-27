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
from nltk.stem.porter import *
import sys
import pickle
from sklearn.externals import joblib


#### Configuration ####

tfid = True		#	True or False
linearkernel = True  	#	False == rbf
stemmer = True		#	True or False
featureSelection = None # 	can be "l1" or "l2" 
stopWords = "english"	# 	can be "english" or None

######################


def removeHeaders(trainingData):
	stemmer = PorterStemmer()
	for i in range(0, len(trainingData)):
		header, body = trainingData[i].split("\n\n",1);
		if stemmer == True:
			tokens = nltk.word_tokenize(body)
			singles = [stemmer.stem(token) for token in tokens]
			trainingData[i] = ' '.join(singles)
		else:
			trainingData[i] = body



def prepareClassifier(trainingFolder):

	training = load_files(trainingFolder,description=None, categories=None, load_content=True,shuffle=True, encoding="latin1",decode_error='strict', random_state=42)
	removeHeaders(training.data)

	vectorizer = CountVectorizer(stop_words = stopWords)
	trainedVector = vectorizer.fit_transform(training.data)
	transformedVector = trainedVector
	tfidf_transformer = None

	if tfid == True:
		tfidf_transformer = TfidfTransformer()
		transformedVector = tfidf_transformer.fit_transform(trainedVector)

	if linearkernel == True and featureSelection != None:
		classifier = svm.LinearSVC(penalty = featureSelection, dual = False).fit(transformedVector, training.target)
	elif linearkernel == True:
		classifier = svm.SVC(kernel = "linear").fit(transformedVector, training.target);		
	else:
		classifier = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape=None, degree=2, gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False).fit(transformedVector, training.target)
	return classifier, vectorizer, tfidf_transformer




def testData(testingFolder, classifier, vectorizer, tfidf_transformer):
	testing = load_files(testingFolder,description=None, categories=None, load_content=True,shuffle=True, encoding="latin1",decode_error='strict', random_state=42)

	removeHeaders(testing.data)

	testVector = vectorizer.transform(testing.data)
	transformedTestVector = testVector

        if tfid == True:
		transformedTestVector = tfidf_transformer.transform(testVector)

	predicted = classifier.predict(transformedTestVector)

	print(metrics.classification_report(testing.target, predicted,target_names=testing.target_names))



def main():
	if(len(sys.argv) == 3):
		trainingFolder = sys.argv[1]
		testingFolder = sys.argv[2]
		
		classifier,vectorizer, tfidf_transformer = prepareClassifier(trainingFolder)
		testData(testingFolder, classifier, vectorizer, tfidf_transformer)
	else:
		print("Format: python partB.py <trainingFolder> <testingFolder>\nP.S : Open python to change configuration !!")

main()
