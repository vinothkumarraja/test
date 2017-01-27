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

training = load_files("/home/vinothkumar/AI/Selected20NewsGroup/Training",description=None, categories=None, load_content=True,shuffle=True, encoding="latin1",decode_error='strict', random_state=42)


def removeHeaders(trainingData):
	for i in range(0, len(trainingData)):
		header, body = trainingData[i].split("\n\n",1);
		trainingData[i] = body

removeHeaders(training.data)

# for bigram
vectorizer = CountVectorizer(ngram_range = (2,2))

#for unigram
#vectorizer = CountVectorizer()
trainedVector = vectorizer.fit_transform(training.data)

tfidf_transformer = TfidfTransformer()
transformedVector = tfidf_transformer.fit_transform(trainedVector)


for i in range(1,9):
	#classifier = svm.SVC(kernel = 'linear').fit(transformedVector[:(transformedVector.shape[0]/8)*i], training.target[:(len(training.target)/8)*i])
	#classifier = MultinomialNB().fit(transformedVector[:(transformedVector.shape[0]/8)*i], training.target[:(len(training.target)/8)*i])
	#classifier = LogisticRegression().fit(transformedVector[:(transformedVector.shape[0]/8)*i], training.target[:(len(training.target)/8)*i])
	classifier = RandomForestClassifier().fit(transformedVector[:(transformedVector.shape[0]/8)*i], training.target[:(len(training.target)/8)*i])

	testing = load_files("/home/vinothkumar/AI/Selected20NewsGroup/Test",description=None, categories=None, load_content=True,shuffle=True, encoding="latin1",decode_error='strict', random_state=42)

	removeHeaders(testing.data)
	testVector = vectorizer.transform(testing.data)
	transformedTestVector = tfidf_transformer.transform(testVector)
	predicted = classifier.predict(transformedTestVector)

	print(metrics.classification_report(testing.target, predicted,target_names=testing.target_names).split("\n")[7])

