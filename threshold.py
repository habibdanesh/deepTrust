from scipy.stats import *
import seaborn as sns
from sklearn.cluster import dbscan
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import trustscore

def find_threshold(ts, y, y_pred):

  # Visualization
  #norm_ts = [float(i)/sum(ts) for i in ts]
  #sns.distplot(norm_ts, color=['gray']*len(norm_ts));
  #sns.distplot(norm_ts, color=['gray']*len(norm_ts), kde=True);

  ts = np.array(ts).reshape(-1,1)
  
  # take trust scores that blong to false predictions
  false_ts = []
  for i in range(len(ts)):
    if(y_pred[i] != y[i]):
      false_ts.append(ts[i])
      
  # take trust scores that blong to true predictions
  true_ts = []
  for i in range(len(ts)):
    if(y_pred[i] == y[i]):
      true_ts.append(ts[i])

  # Clustering
  core_sample, label = dbscan(true_ts, eps=0.005, min_samples=2)
  num_clusters = len(np.unique(label)) - 1 # because label -1 belongs to outliers

  # Find largest cluster
  # remove -1s
  for i, val in enumerate(label):
    if val == -1:
      label[i] = num_clusters

  counts = np.bincount(label)
  largest_cl = np.argmax(counts[0:len(counts)-1]) # -1 to omit the outliers

  # threshold = larges value in the largest cluster
  threshold = 1000000
  for i, val in enumerate(true_ts):
    if(label[i] == largest_cl):
      if(val < threshold):
        threshold = val
  
  return threshold


if __name__ == '__main__':
  
  X, y = datasets.load_digits(return_X_y=True)
  #X, y = datasets.load_iris(return_X_y=True)
  #X, y = datasets.load_wine(return_X_y=True)
  #X, y = datasets.load_breast_cancer(return_X_y=True)

  print(X.shape)

  N_test_itr = 100
  avg_acc_train = 0.0
  avg_acc_test = 0.0
  avg_threshold = 0.0
  avg_acc_thr_train = 0.0
  avg_acc_thr_test = 0.0

  for itr in range(N_test_itr):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    N_train = len(X_train)
    N_test = len(X_test)

    # Train logistic regression
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Get model's accuracy
    y_pred_train = model.predict(X_train)
    acc_train = np.count_nonzero(y_pred_train == y_train) / float(N_train)
    avg_acc_train += acc_train

    y_pred_test = model.predict(X_test)
    acc_test = np.count_nonzero(y_pred_test == y_test) / float(N_test)
    avg_acc_test += acc_test

    # Trust score on training set
    trust_model = trustscore.TrustScore()
    #trust_model = trustscore.TrustScore(k=60, alpha=0.01, filtering="density")

    trust_model.fit(X_train, y_train)
    trust_score_train = trust_model.get_score(X_train, y_pred_train)

    threshold = find_threshold(trust_score_train, y_train, y_pred_train)
    avg_threshold += threshold

    # Trust score on testing set
    trust_score_test = trust_model.get_score(X_test, y_pred_test)

    # Validation of threshold on training and testing sets
    # Find the portion of true predictions that have TS>threshold
    # and the portion of false predictions that have TS<threshold
    true_true_preds_train = 0
    true_false_preds_train = 0
    for i in range(N_train):
      if((y_pred_train[i] == y_train[i]) and (trust_score_train[i] > threshold)):
        true_true_preds_train += 1
      if((y_pred_train[i] != y_train[i]) and (trust_score_train[i] < threshold)):
        true_false_preds_train += 1

    acc_thr_train = (true_true_preds_train + true_false_preds_train) / float(N_train)
    avg_acc_thr_train += acc_thr_train

    true_true_preds_test = 0
    true_false_preds_test = 0
    for i in range(N_test):
      if((y_pred_test[i] == y_test[i]) and (trust_score_test[i] > threshold)):
        true_true_preds_test += 1
      if((y_pred_test[i] != y_test[i]) and (trust_score_test[i] < threshold)):
        true_false_preds_test += 1

    acc_thr_test = (true_true_preds_test + true_false_preds_test) / float(N_test)
    avg_acc_thr_test += acc_thr_test

  # print results
  print('Training samples: %d' % N_train)
  print('Testing samples: %d' % N_test)

  print('Training Acc: %.2f %%' % ((avg_acc_train/N_test_itr)*100))
  print('Testing Acc: %.2f %%' % ((avg_acc_test/N_test_itr)*100))
  print('Threshold = %.2f' % (avg_threshold/N_test_itr))
  print('Threshold Training Acc: %.2f %%' % ((avg_acc_thr_train/N_test_itr)*100))
  print('Threshold Testing Acc: %.2f %%' % ((avg_acc_thr_test/N_test_itr)*100))

  print('\nDone.')




  
