
# coding: utf-8

# In[1]:


import sys
import pickle
sys.path.append("../tools/")
import warnings; warnings.simplefilter('ignore')


from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


# In[2]:


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'total_payments', 'bonus', 'total_stock_value', 'exercised_stock_options',
                 'long_term_incentive','from_poi_to_this_person', 'from_this_person_to_poi',
                 'shared_receipt_with_poi','to_messages', 'from_messages']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# In[3]:


### Task 2: Remove outliers

import pandas as pd
import numpy as np

data_dict.pop('TOTAL', 0) 
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0) # not a real person
data_dict.pop('LOCKHART EUGENE E', 0) # all value missing for this person

# convert to pandas dataframe
data_dict_df = pd.DataFrame.from_dict(data_dict, orient='columns')

# replacing 'NaN' string in dataframe
for column in data_dict_df:
    data_dict_df[column].replace('NaN',np.nan, inplace=True)
    
# only keeping columns that's in feature list
sel_data = data_dict_df.loc[features_list]
sel_data.fillna(0,inplace = True)


# In[4]:


### Task 3: Create new feature(s)

import copy

data_dict = copy.deepcopy(sel_data)

for key in data_dict:
    if (data_dict[key]['to_messages'] != 'NaN' and 
        data_dict[key]['to_messages'] != 0 and 
        data_dict[key]['from_poi_to_this_person'] != 'NaN' and 
        data_dict[key]['from_poi_to_this_person'] != 0) :
            data_dict[key]['poi_to_percent'] = data_dict[key]['from_poi_to_this_person'] 
            float((data_dict[key]['to_messages']))
    else:
        data_dict[key]['poi_to_percent'] = 'NaN'

for key in data_dict:
    if (data_dict[key]['from_messages'] != 'NaN' and 
        data_dict[key]['from_messages'] != 0 and 
        data_dict[key]['from_this_person_to_poi'] != 'NaN' and 
        data_dict[key]['from_this_person_to_poi'] != 0) :
            data_dict[key]['poi_from_percent'] = data_dict[key]['from_this_person_to_poi'] 
            float((data_dict[key]['from_messages']))
    else:
        data_dict[key]['poi_from_percent'] = 'NaN'


# In[5]:


features_list.pop()
features_list.pop()
features_list.append('poi_to_percent')
features_list.append('poi_from_percent')

### Store to my_dataset for easy export below.
my_dataset = dict(data_dict)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[6]:


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# split data into train and test sets
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=19)

# create a simple function to calculate list average
def avg_list(l):
    return sum(l) / float(len(l))

print '============================='
print 'Try a varity of classifiers'
print '============================='


# In[7]:


# try naive bayes
from sklearn.naive_bayes import GaussianNB

f1s = []
precisions = []
recalls = []

for train_index, test_index in kf.split(features, labels):
    features_train, features_test = np.array(features)[train_index], np.array(features)[test_index] 
    labels_train, labels_test = np.array(labels)[train_index], np.array(labels)[test_index]
    
    # fit the classifier using training data
    clf_naive_bayes = GaussianNB()
    clf_naive_bayes.fit(features_train, labels_train)
    
    # test/predict the classifier over test data
    pred_naive_bayes = clf_naive_bayes.predict(features_test)
    
    # calculate precision and recall on test data & append these scores to the precision and recall lists above
    f1s.append(f1_score(labels_test, pred_naive_bayes))
    precisions.append(precision_score(labels_test, pred_naive_bayes))
    recalls.append(recall_score(labels_test, pred_naive_bayes))
        
print "%s f1_score: %s" % ('naive_bayes', avg_list(f1s)) 
print "%s precision_score: %s" %('naive_bayes', avg_list(precisions))
print "%s recall_score: %s" % ('naive_bayes', avg_list(recalls))


# In[8]:


# try k-nearest neighbors
from sklearn.neighbors import KNeighborsClassifier

f1s = []
precisions = []
recalls = []

for train_index, test_index in kf.split(features, labels):
    features_train, features_test = np.array(features)[train_index], np.array(features)[test_index] 
    labels_train, labels_test = np.array(labels)[train_index], np.array(labels)[test_index]
    
    # fit the classifier using training data
    clf_neigh = KNeighborsClassifier()
    clf_neigh.fit(features_train, labels_train) 
    
    # test/predict the classifier over test data
    pred_neigh = clf_neigh.predict(features_test)
    
    # calculate precision and recall on test data & append these scores to the precision and recall lists above
    f1s.append(f1_score(labels_test, pred_neigh))
    precisions.append(precision_score(labels_test, pred_neigh))
    recalls.append(recall_score(labels_test, pred_neigh))
        
print "%s f1_score: %s" % ('KNeighbors', avg_list(f1s)) 
print "%s precision_score: %s" %('KNeighbors', avg_list(precisions))
print "%s recall_score: %s" % ('KNeighbors', avg_list(recalls))


# In[9]:


# try random forest
from sklearn.ensemble import RandomForestClassifier

f1s = []
precisions = []
recalls = []

for train_index, test_index in kf.split(features, labels):
    features_train, features_test = np.array(features)[train_index], np.array(features)[test_index] 
    labels_train, labels_test = np.array(labels)[train_index], np.array(labels)[test_index]
    
    # fit the classifier using training data
    clf_rf = RandomForestClassifier(max_depth=2, random_state=14, class_weight = 'balanced')
    clf_rf.fit(features_train, labels_train) 
    
    # test/predict the classifier over test data
    pred_rf = clf_rf.predict(features_test)
    
    # calculate precision and recall on test data & append these scores to the precision and recall lists above
    f1s.append(f1_score(labels_test, pred_rf))
    precisions.append(precision_score(labels_test, pred_rf))
    recalls.append(recall_score(labels_test, pred_rf))
        
print "%s f1_score: %s" % ('random forest', avg_list(f1s)) 
print "%s precision_score: %s" %('random forest', avg_list(precisions))
print "%s recall_score: %s" % ('random forest', avg_list(recalls))


# In[10]:


# try svm
from sklearn import svm

f1s = []
precisions = []
recalls = []

for train_index, test_index in kf.split(features, labels):
    features_train, features_test = np.array(features)[train_index], np.array(features)[test_index] 
    labels_train, labels_test = np.array(labels)[train_index], np.array(labels)[test_index]
    
    # fit the classifier using training data
    clf_svm = svm.SVC(class_weight = 'balanced')
    clf_svm.fit(features_train, labels_train) 
    
    # test/predict the classifier over test data
    pred_svm = clf_svm.predict(features_test)
    
    # calculate precision and recall on test data & append these scores to the precision and recall lists above
    f1s.append(f1_score(labels_test, pred_svm))
    precisions.append(precision_score(labels_test, pred_svm))
    recalls.append(recall_score(labels_test, pred_svm))
        
print "%s f1_score: %s" % ('svm', avg_list(f1s)) 
print "%s precision_score: %s" %('svm', avg_list(precisions))
print "%s recall_score: %s" % ('svm', avg_list(recalls))


# In[11]:


# try adaboost
from sklearn.ensemble import AdaBoostClassifier

f1s = []
precisions = []
recalls = []

for train_index, test_index in kf.split(features, labels):
    features_train, features_test = np.array(features)[train_index], np.array(features)[test_index] 
    labels_train, labels_test = np.array(labels)[train_index], np.array(labels)[test_index]
    
    # fit the classifier using training data
    clf_ada = AdaBoostClassifier(n_estimators=100)
    clf_ada.fit(features_train, labels_train) 
    
    # test/predict the classifier over test data
    pred_ada = clf_ada.predict(features_test)
    
    # calculate precision and recall on test data & append these scores to the precision and recall lists above
    f1s.append(f1_score(labels_test, pred_ada))
    precisions.append(precision_score(labels_test, pred_ada))
    recalls.append(recall_score(labels_test, pred_ada))
        
print "%s f1_score: %s" % ('adaboost', avg_list(f1s)) 
print "%s precision_score: %s" %('adaboost', avg_list(precisions))
print "%s recall_score: %s" % ('adaboost', avg_list(recalls))


# In[12]:


# try decision tree
from sklearn import tree

f1s = []
precisions = []
recalls = []

for train_index, test_index in kf.split(features, labels):
    features_train, features_test = np.array(features)[train_index], np.array(features)[test_index] 
    labels_train, labels_test = np.array(labels)[train_index], np.array(labels)[test_index]
    
    # fit the classifier using training data
    clf_tree = tree.DecisionTreeClassifier(max_leaf_nodes=3, class_weight = 'balanced')
    clf_tree.fit(features_train, labels_train) 
    
    # test/predict the classifier over test data
    pred_tree = clf_tree.predict(features_test)
    
    # calculate precision and recall on test data & append these scores to the precision and recall lists above
    f1s.append(f1_score(labels_test, pred_tree))
    precisions.append(precision_score(labels_test, pred_tree))
    recalls.append(recall_score(labels_test, pred_tree))
        
print "%s f1_score: %s" % ('decision tree', avg_list(f1s)) 
print "%s precision_score: %s" %('decision tree', avg_list(precisions))
print "%s recall_score: %s" % ('decision tree', avg_list(recalls))


# In[13]:


# build a pipeline to do multi-stage operations

from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

estimators = [('scaling', MinMaxScaler()), ('selectkbest', SelectKBest(k = 'all')), ('clf_rf',clf_rf)]
pipe = Pipeline(estimators)

f1s = []
precisions = []
recalls = []

for train_index, test_index in kf.split(features, labels):
    features_train, features_test = np.array(features)[train_index], np.array(features)[test_index] 
    labels_train, labels_test = np.array(labels)[train_index], np.array(labels)[test_index]
    
    # fit the classifier using training data
    pipe.fit(features_train,labels_train)
    
    # test/predict the classifier over test data
    pred_pipe = pipe.predict(features_test)
    
    # calculate precision and recall on test data & append these scores to the precision and recall lists above
    f1s.append(f1_score(labels_test, pred_pipe))
    precisions.append(precision_score(labels_test, pred_pipe))
    recalls.append(recall_score(labels_test, pred_pipe))
    
print "%s f1_score: %s" % ('decision tree pipe', avg_list(f1s)) 
print "%s precision_score: %s" %('decision tree pipe', avg_list(precisions))
print "%s recall_score: %s" % ('decision tree pipe', avg_list(recalls))
print ' ' 


# In[14]:


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.htm

# use tester to see how the current pipeline performs
import tester
print '============================='
print 'use tester to see how the current pipeline performs:'
print '============================='
print tester.test_classifier(pipe, data_dict, features_list, folds = 1000)


# In[20]:


# use GridSearchCV to find out best parameters to selectkbest, random forest and pipeline
from sklearn.model_selection import GridSearchCV

parameters = {'selectkbest__k': range(4,12), 'clf_rf__max_leaf_nodes': range(4,8),
              'clf_rf__criterion': ['gini'], 'clf_rf__n_estimators': [200],               
              'clf_rf__min_samples_split':range(2,5),'clf_rf__min_samples_leaf':range(4,10),              
              'clf_rf__class_weight': ['balanced']}

gs = GridSearchCV(pipe, param_grid = parameters, scoring = 'f1', n_jobs = -1)
gs.fit(features_train, labels_train)


# In[21]:


# use the parameters selected by GridSearchCV to refit random forest

clf = gs.best_estimator_

f1s = []
precisions = []
recalls = []

for train_index, test_index in kf.split(features, labels):
    features_train, features_test = np.array(features)[train_index], np.array(features)[test_index] 
    labels_train, labels_test = np.array(labels)[train_index], np.array(labels)[test_index]
    
    # fit the classifier using training data
    clf.fit(features_train,labels_train)
    
    # test/predict the classifier over test data
    pred_clf = clf.predict(features_test)
    
    # calculate precision and recall on test data & append these scores to the precision and recall lists above
    f1s.append(f1_score(labels_test, pred_clf))
    precisions.append(precision_score(labels_test, pred_clf))
    recalls.append(recall_score(labels_test, pred_clf))


print ' '
print "%s f1 score: %s" %('clf pipeline_refitted', avg_list(f1s))
print "%s precision_score: %s" %('clf pipeline_refitted', avg_list(precisions))
print "%s recall_score: %s" % ('clf pipeline_refitted', avg_list(recalls))


# In[22]:


print '============================='
print 'performance of refitted pipeline (using tester)'
print '============================='
print tester.test_classifier(clf, data_dict, features_list, folds = 1000)


# In[23]:


kbest_step= clf.named_steps['selectkbest']
idxs_selected= kbest_step.get_support()

features_dataframe_new = np.array(features_list[1:])[idxs_selected]
print ' '
print '============================='
print 'features used in final pipeline/algorithm'
print '============================='
print features_dataframe_new


# In[24]:


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

