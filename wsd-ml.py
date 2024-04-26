"""
wsd-ml.py
Ricardo Garriga-Ramos
CMSC-416-001 - INTRO TO NATURAL LANG PROCESS - mar 20 Spring 2024

A multi algorithm implementation of word sense disambiguation 
The program uses a bag of words feature representation which is trained using 
GaussianNaiveBayes, RandomForestClassifier, and SVM from the scikit learn library

GaussianNaiveBayes uses the Naive assumption and Bayes theorem while assuming that the classification categories are Gaussian
RandomForestClassifier uses a set (currently set to 200) of random tree based classifiers
SVM uses a maximum margin separating hyperplane to decide between labels

Confusion Matrices were phone is positive and product is Negitive
GaussianNaiveBayes
True positive = 62      True Negitive = 45
False Positive = 9      False Negitive = 10
Accuracy = 0.8492063492063492

RandomForestClassifier
True positive = 65      True Negitive = 49
False Positive = 5      False Negitive = 7
Accuracy = 0.9047619047619048

SVM
True positive = 65      True Negitive = 45
False Positive = 9      False Negitive = 7
Accuracy = 0.873015873015873

All models beat both my discussion list form assignment 3 and the most frequent sense baseline
Discussion list
True positive = 55      True Negitive = 49
False Positive = 5      False Negitive = 17
Accuracy = 0.8253968253
Most frequent sense baseline
Accuracy = 0.4285714286


Run instructions:
python3 wsd.py line-train.txt line-test.txt [OPTIONAL: ml-model] > my-line-answers.txt

Examples:
python3 wsd-ml.py line-train.txt line-test.txt GaussianNaiveBayes > my-line-answers.txt
python3 wsd-ml.py line-train.txt line-test.txt RandomForestClassifier > my-line-answers.txt
python3 wsd-ml.py line-train.txt line-test.txt SVM > my-line-answers.txt
and
python3 wsd-ml.py line-train.txt line-test.txt  > my-line-answers.txt
which defults to
python3 wsd-ml.py line-train.txt line-test.txt GaussianNaiveBayes > my-line-answers.txt
"""

import sys
import re
import string
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.feature_extraction import text
from sklearn.preprocessing import LabelEncoder


def vectorize_training(stop_list):
    vectors_dict = {'senseid':[]} 
    has_context = False
    while True:
        cur_line = train_file.readline()
        if not cur_line:
            break
        # vectorize line
        if has_context:
            parse_context(stop_list, vectors_dict, cur_line)
            has_context = False
        # detect line
        elif re.search("<context>", cur_line):
            has_context = True
        # get senseid
        elif re.search("<answer instance=", cur_line):
            senseid = cur_line.split('"')[3]
            vectors_dict['senseid'].append(senseid)
    
    return vectors_dict
def parse_context(stop_list, vectors_dict, cur_line, adding = True):
    # add a new row in each numaric column

    for key in vectors_dict:
        if key != 'senseid' and key != 'instanceid':
            vectors_dict[key].append(0)
    
    words = cur_line.lower().split(" ")
    for word in words:
        is_head = re.search("<head>(lines?)</head>", word)
        if is_head:
            # is key word
            # not in use
            key = is_head.group(1)

        elif not re.search("<s>|</s>|<p>|</p>|<@>", word):

            # remove stop words
            if word not in stop_list:
                word = word.translate(str.maketrans('', '', string.punctuation))
                if word not in stop_list:
                    
                    
                    if re.search("[0-9]+", word):
                        # is a number
                        # not in use
                        num = "<numaric>"
                    else:
                        # is a word
                        first_key = list(vectors_dict.keys())[0]
                        i = len(vectors_dict[first_key])
                        known_word = True
                        if word not in vectors_dict.keys():
                            if adding:
                                # keep matrix square
                                vectors_dict[word] = []
                                while i > 0:
                                    vectors_dict[word].append(0)
                                    i -= 1
                            else:
                                known_word = False

                        if known_word:
                            if i> len(vectors_dict[word]):
                                print(word)
                                print(vectors_dict)
                                
                            vectors_dict[word][i-1] += 1
def vectorize_test(stop_list, train_vectors_dict):
    vectors_dict = {'instanceid': []} 
    for key in train_vectors_dict:
        if key != 'senseid':
            vectors_dict[key] = []

    has_context = False

    while True:
        cur_line = test_file.readline()
        if not cur_line:
            break
        # vectorize line
        if has_context:
            parse_context(stop_list, vectors_dict, cur_line, adding=False)
            has_context = False
        # detect line
        elif re.search("<context>", cur_line):
            has_context = True
        # get instance
        elif re.search("<instance id=", cur_line):
            instanceid = cur_line.split('"')[1]
            vectors_dict['instanceid'].append(instanceid)
    return vectors_dict


def train(algorithm, train_vectors_df):
    # return model
    X = train_vectors_df.drop('senseid', axis = 1)
    y = train_vectors_df['senseid']
    if algorithm == 'GaussianNaiveBayes':
        gnb = GaussianNB()
        return gnb.fit(X, y)
        
    elif algorithm == 'RandomForestClassifier':
        rfc = RandomForestClassifier(n_estimators=200)
        return rfc.fit(X, y)

    elif algorithm == 'SVM':
        clf = svm.SVC()
        return clf.fit(X, y)  
      
def test(model, test_vectors_df):
    inst_list = test_vectors_df['instanceid']
    lable_list = model.predict(test_vectors_df.drop('instanceid', axis = 1))
    for i in range(len(inst_list)):
        print(f"<answer instance=\"{inst_list[i]}\" senseid=\"{lable_list[i]}\"/>")

def main():

    # use stoplist from skikit
    my_words = ['','\n']
    stop_list = text.ENGLISH_STOP_WORDS.union(my_words)
    # Creeate vector representations for the training and testing data as a dictionary and dataframe
    train_vectors_dict = vectorize_training(stop_list)
    train_vectors_df = pd.DataFrame.from_dict(train_vectors_dict)
    test_vectors_dict = vectorize_test(stop_list, train_vectors_dict)
    test_vectors_df = pd.DataFrame.from_dict(test_vectors_dict)


    model = train(algorithm, train_vectors_df)
    test(model, test_vectors_df)










# parse arguments and start program
if len(sys.argv) >= 3:
    # open files
    train_file = open(sys.argv[1], 'r')
    test_file = open(sys.argv[2], 'r')

    # spesified l-model  len(sys.argv) = 4
    if len(sys.argv) == 4:
        algorithm = sys.argv[3]
    # unspesified l-model  len(sys.argv) = 3
    elif len(sys.argv) == 3:
        algorithm = 'GaussianNaiveBayes'
        

    main()

    train_file.close()
    test_file.close()
else:
    print('Not enough arguments')




