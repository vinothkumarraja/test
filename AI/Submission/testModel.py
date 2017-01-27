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
	if(len(sys.argv) == 5):
		testingFolder = sys.argv[1]
		vectPath = sys.argv[2]
		tfidfPath = sys.argv[3]
		classifierPath = sys.argv[4]
		
		vectorizer = joblib.load(vectPath)
		tfidf_transformer = joblib.load(tfidfPath)
		classifier = joblib.load(classifierPath)
	
		testData(testingFolder, classifier, vectorizer, tfidf_transformer)
	else:
		print("Format: python testModel.py <testingFolder> <file_path.vec> <file_path.tfidf> <classifierPath.clf>")

main()
