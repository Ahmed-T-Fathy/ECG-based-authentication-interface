from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
# first read data
import numpy as np
import pickle
from matplotlib import pyplot as plt

import helpers
import read_data

def testFromMain(type,x):
    x = x.reshape(1,-1)
    filename = ''
    accuracy=-1
    if type==1:
        filename='modelnonFiducial.pkl'
        accuracy=100
    elif type==2:
        filename = 'modelFiducial.pkl'
    loaded_model = pickle.load(open(filename, 'rb'))
    prediction = loaded_model.predict(x)
    print(prediction)

    return accuracy


signals, labels = read_data.read_data('./Data sets 2/*', 1500)


# label encoding
LabelEncoder = preprocessing.LabelEncoder()

def Training(type):
    encodedData = LabelEncoder.fit_transform(labels)
    # train svm model
    clf = svm.SVC(kernel='poly',degree=3, C=1)
    #
    #
    # # save the model to disk
    filename = 'modelFiducial.pkl'

    preprocessed_signals=filtered_signals=[]
    if type==1: #Non
        # preprocessing
        preprocessed_signals = helpers.data_preprocessing(signals, 1, 40, 500, 4)
        # feature extraction
        filtered_signals = helpers.extract_features(preprocessed_signals, 1500,1)
        filename = 'modelnonFiducial.pkl'
    elif type==2:
        filtered_signals=helpers.extract_features(signals,1500,type)
        encodedData,filtered_signals=helpers.editFeature(encodedData,filtered_signals,42)
        filename = 'modelFiducial.pkl'

    # split data
    X_train, X_test, y_train, y_test = train_test_split(filtered_signals, encodedData, test_size=0.2,shuffle=True)
    clf.fit(X_train, y_train)

    pickle.dump(clf, open(filename, 'wb'))
    loaded_model = pickle.load(open(filename, 'rb'))

    prediction =loaded_model.predict(X_test)

    print('prediction:', prediction)
    print('True:', y_test)

    accuracy=loaded_model.score(X_test,y_test)
    print('Accuracy :', accuracy)

# Training(2)
