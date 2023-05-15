from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
# first read data
import numpy as np
from matplotlib import pyplot as plt

import helpers
import read_data

signals, labels = read_data.read_data('./Data sets 2/*', 1500)

# preprocessing
preprocessed_signals = helpers.data_preprocessing(signals, 1, 40, 500, 4)

# label encoding
LabelEncoder = preprocessing.LabelEncoder()
encodedData = LabelEncoder.fit_transform(labels)

# feature extraction
filtered_signals = helpers.extract_features(signals, 1500,2)

qrs=helpers.QRS_Features(signals[0])

# split data
X_train, X_test, y_train, y_test = train_test_split(filtered_signals, encodedData, test_size=0.2,shuffle=True)

# train svm model
clf = svm.SVC(kernel='poly',degree=3, C=1)
clf.fit(X_train, y_train)
prediction =clf.predict(X_test)

print('prediction:', prediction)
print('True:', y_test)



accuracy=clf.score(X_test,y_test)
print('Accuracy :', accuracy)

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.plot(signals[0])
plt.subplot(132)
plt.plot(preprocessed_signals[0])
plt.title("Accuracy :"+str(accuracy))
plt.subplot(133)
plt.plot(filtered_signals[0])

plt.show()

# print((filtered_signals))
