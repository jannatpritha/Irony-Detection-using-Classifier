# Import Section
import csv
import codecs
import sys
import io
import numpy as np


# For Classifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier





# Python script for confusion matrix creation. 
from sklearn.metrics import *

import PreprocessingModule as PM



def main():
  tweets = []
  label = []
  csv.field_size_limit(500 * 1024 * 1024)  
  with open('/content/drive/MyDrive/Colab Notebooks/CodeForSKLearnClassifier/SemEval2018-IronyDetection_Train.txt', 'r') as f:
      next(f) # skip headings
      reader=csv.reader(f, dialect="excel-tab")
      for line in reader:
        # print(line[0])
        preProcessedTweetText= PM.preProcessingModule(line[2])
          #print(preProcessedTweetText)
        tweets.append(preProcessedTweetText)
        if(line[1]=="1"):
          label.append("irony")
        else:
          label.append("notIrony")
      #print(label)
      #print(tweets)





  X_train = np.array(tweets)
  Y_train = np.array(label)
  
  ## Define Classifier
  classifier = Pipeline([
     ('count_vectorizer', CountVectorizer(ngram_range=(1, 3))),
      ('tfidf', TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)),
     ('clf', OneVsRestClassifier(MultinomialNB()))]) 

  ## Train Classifier
  classifier.fit(X_train, Y_train)
  
  
  ## Test Set Prediction Module
  testTweets = []
  testLabelGold = []
  csv.field_size_limit(500 * 1024 * 1024)  
  with open('/content/drive/MyDrive/Colab Notebooks/CodeForSKLearnClassifier/SemEval2018-IronyDetectionSmallVersion.txt', 'r') as f:
      next(f) # skip headings
      reader=csv.reader(f, dialect="excel-tab")
      for line in reader:
          #print(line[2])
          preProcessedTweetText= PM.preProcessingModule(line[2])
          #print(preProcessedTweetText)
          testTweets.append(preProcessedTweetText)
          if(line[1]=="1"):
            testLabelGold.append("irony")
          else:
            testLabelGold.append("notIrony")
  
 
  # test_label_prediction
  X_test = np.array(testTweets)
  testLabelPredicted = classifier.predict(X_test)


  # Evaluation
  results = confusion_matrix(testLabelGold, testLabelPredicted) 
    
  print ('Confusion Matrix :')
  print (results) 

  print ('Recall Score :',recall_score(testLabelGold, testLabelPredicted, labels=['notIrony','irony'], pos_label='irony'))
  print ('Precision Score :',precision_score(testLabelGold, testLabelPredicted, labels=['notIrony','irony'], pos_label='irony'))
  print ('F1 Score :',f1_score(testLabelGold, testLabelPredicted, labels=['notIrony','irony'], pos_label='irony'))
  print ('Accuracy :',accuracy_score(testLabelGold, testLabelPredicted)) 
  
  
  print ('Evaluation Report : ')
  print (classification_report(testLabelGold, testLabelPredicted)) 

if __name__ == '__main__':
  main()