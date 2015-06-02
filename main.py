import sys, random, time
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import KFold
from sklearn.externals import joblib
from sklearn.metrics import hamming_loss, f1_score
from sklearn import preprocessing

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

CLFS_NAMES = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost"]
CLFS = [
    KNeighborsClassifier(n_neighbors=1, weights='distance'),
    LinearSVC(C=1000, loss='hinge'),
    SVC(C=1, gamma=2),
    DecisionTreeClassifier(),
    RandomForestClassifier(max_depth=5, n_estimators=100, n_jobs=4),
    AdaBoostClassifier()
]

def train(method=0):
	print 'command train'
	print 'use %s' % CLFS_NAMES[method]
	
	description_list = []
	tags_list = []
	
	for line in open('train.data') :
	    tmp = line.rstrip('\r\n').split('#$#')
	    description = tmp[1]
	    tags = tmp[2].rstrip(',').split(',') # TODO Semantic Web (RDF, OWL, etc.)
	    description_list.append(description)
	    tags_list.append(tags)
	
	all_tags = open('allTags.txt').read().splitlines() # TODO how to fit this ?
	lb = preprocessing.MultiLabelBinarizer()
	binary_tags_list = lb.fit_transform(tags_list)
	
	X_train = np.array(description_list)
	y_train = binary_tags_list
	
	clf = Pipeline([
	    ('vectorizer', CountVectorizer()),
	    ('tfidf', TfidfTransformer()),
	    ('clf', OneVsRestClassifier(CLFS[method]))])
		
	print 'train begin'
	
	clf.fit(X_train, y_train)
	
	print 'train end'
	
	joblib.dump(clf, 'models/model.pkl')
	joblib.dump(lb, 'models/lb.pkl')
	
def test():
	print 'command test'
	
	description_list = []
	
	for line in open('test.data') :
	    tmp = line.rstrip('\r\n').split('#$#')
	    description = tmp[1]
	    description_list.append(description)
	    
	X_test = np.array(description_list)
	
	clf = joblib.load('models/model.pkl')
	lb = joblib.load('models/lb.pkl')
	
	print 'test begin'
	
	predicted = clf.predict(X_test)
	all_labels = lb.inverse_transform(predicted)
	
	print 'test end'
	
	f = open('result.txt', 'w')
	
	for idx, labels in enumerate(all_labels):
		tags = ','.join(labels)
		line = str(idx + 1) + '#$#' + tags + '\n'
		f.write(line)
		
	f.close()
	
def compare(sample='all', method='all', test_on='test', n=10):
	if type(method) is int:
		clfs_names = [CLFS_NAMES[method]]
		clfs = [CLFS[method]]
	else:
		clfs_names = CLFS_NAMES
		clfs = CLFS
		
	print 'command compare'
	print 'sample %s, method %s, test on %s, use %d-fold' % (sample, clfs_names, test_on, n)
	
	description_list = []
	tags_list = []
	
	for line in open('train.data'):
		tmp = line.rstrip('\r\n').split('#$#')
		description = tmp[1]
		tags = tmp[2].rstrip(',').split(',') # TODO Semantic Web (RDF, OWL, etc.)
		description_list.append(description)
		tags_list.append(tags)
		
	if type(sample) is int:	
		random_idx = random.sample(range(len(description_list)), sample)
		description_list = [description_list[idx] for idx in random_idx]
		tags_list = [tags_list[idx] for idx in random_idx]
	
	lb = preprocessing.MultiLabelBinarizer()
	binary_tags_list = lb.fit_transform(tags_list)
	
	result = dict.fromkeys(clfs_names)
	for key in result:
		result[key] = [0, 0, 0]
	
	kf = KFold(len(description_list), n_folds=n)
	for train, test in kf:
		print "%%%%%%%%%%%%%%%%%%%%"
		
		if test_on == 'train':
			test = train
		
		X_train = np.array(description_list)[train]
		y_train = binary_tags_list[train]
    
		X_test = np.array(description_list)[test]
		y_test = binary_tags_list[test]
		
		for clf in zip(clfs_names, clfs):
			start_time = time.time()
			name = clf[0]
			clf = Pipeline([
			    ('vectorizer', CountVectorizer()),
			    ('tfidf', TfidfTransformer()),
			    ('clf', OneVsRestClassifier(clf[1]))])
			clf.fit(X_train, y_train)
			
			predicted = clf.predict(X_test)
			y_pred = np.array(predicted)
			hamming_loss_score = hamming_loss(y_test, y_pred)
			f1_macro = f1_score(y_test, y_pred, average='macro') 
			f1_micro = f1_score(y_test, y_pred, average='micro') 
			
			elapsed_time = time.time() - start_time
			
			print '%s: ' % name
			print 'hamming_loss: %s' % hamming_loss_score
			print 'f1 macro: %s' % f1_macro
			print 'f1 micro: %s' % f1_micro
			print 'use %d s' % elapsed_time
			print '==============='
			
			sys.stdout.flush()
			
			sum = result[name]
			
			sum[0] = sum[0] + hamming_loss_score
			sum[1] = sum[1] + f1_macro
			sum[2] = sum[2] + f1_micro
	for key in result:
		sum = result[key]
		sum[0] = sum[0] / n
		sum[1] = sum[1] / n
		sum[2] = sum[2] / n
		print key
		print sum

def IsNumber(x):
    try:
        _ = int(x)
    except ValueError:
        return False
    return True
		
try:
	command = sys.argv[1]
	params = [int(i) if IsNumber(i) else i for i in sys.argv[2:]]
	eval(command + '(*params)')
except IndexError:
	print "Please type a exist command (Eg: train, test, or compare) ..."




