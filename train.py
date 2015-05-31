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
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

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

#description_list = description_list[:100]
#binary_tags_list = binary_tags_list[:100]

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost"]
clfs = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier()]

# nfold
def nfold(n):
    kf = KFold(len(description_list), n_folds=n)
    result = dict.fromkeys(names)
    for key in result:
        result[key] = [0, 0, 0]
    for train, test in kf:
        print "%%%%%%%%%%%%%%%%%%%%"
        X_train = np.array(description_list)[train]
        y_train = binary_tags_list[train]
    
        X_test = np.array(description_list)[test]
        y_test = binary_tags_list[test] 
    
        for clf in zip(names, clfs):
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
            
            print '%s: ' % name
            print 'hamming_loss: %s' % hamming_loss_score
            print 'f1 macro: %s' % f1_macro
            print 'f1 micro: %s' % f1_micro
            print "==============="
            
            sum = result[name]
            
            sum[0] = sum[0] + hamming_loss_score
            sum[1] = sum[1] + f1_macro
            sum[2] = sum[2] + f1_micro
    for key in result:
        sum = result[key]
        sum[0] = sum[0] / n
        sum[1] = sum[1] / n
        sum[2] = sum[2] / n
    return result
        
result = nfold(10)
for key in result:
    print key
    print result[key]

# Persistence
#    joblib.dump(classifier, 'models/by_hand_%d.pkl' % split)
#    classifier = joblib.load('models/by_hand_%d.pkl' % split)


