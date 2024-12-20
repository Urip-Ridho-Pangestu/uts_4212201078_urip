import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn import datasets
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut

# Load the dataset (replace with your actual file path)
train_data = pd.read_csv('../archive (7)/emnist-bymerge-train2.csv', header=None)
test_data = pd.read_csv('../archive (7)/emnist-bymerge-test.csv', header=None)
l = train_data[0].head(4000)
d = train_data.drop(0, axis=1).head(4000)
lt = test_data[0].head(4000)
dt = test_data.drop(0, axis=1).head(4000)

print(d.shape)
print(l.shape)
print(dt.shape)
print(lt.shape)

class_namess = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
               'a','b','d','e','f','g','h','n','q','r','t']

hog_features_train = []
for image in d.values:
    fd,_ = hog(image.reshape(28, 28), orientations=9, pixels_per_cell=(7, 7), cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
    hog_features_train.append(fd)

hog_features_train = np.array(hog_features_train)

hog_features_test = []
for image in dt.values:
    fd,_ = hog(image.reshape(28, 28), orientations=9, pixels_per_cell=(7, 7), cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
    hog_features_test.append(fd)

hog_features_test = np.array(hog_features_test)

scaler = StandardScaler()
hog_features_train_scaled = scaler.fit_transform(hog_features_train)
hog_features_test_scaled = scaler.transform(hog_features_test)

lb = LabelBinarizer()
lb.fit(l)
y_train = l.values
y_test = lt.values


clf_svm = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True,C=6,gamma='scale',cache_size=1000,degree=5))


loo = LeaveOneOut()
loo_true1 = []
loo_pred1 = []
a=0
for train_index, test_index in loo.split(hog_features_train):
    X_train, X_test = hog_features_train[train_index], hog_features_train[test_index]
    y_train_split, y_test_split = y_train[train_index], y_train[test_index]

    # Train the SVM classifier
    clf_svm.fit(X_train, y_train_split)
    loo_true1.append(y_test_split[0])
    loo_pred1.append(clf_svm.predict(X_test)[0])
    print(y_test_split.shape)
    print(y_train_split.shape)


loo_true = np.array(loo_true1)
loo_pred = np.array(loo_pred1)

print("")
print("")
print(loo_true)
print("")
print(loo_pred)

conf_mat_svm = confusion_matrix(loo_true, loo_pred)
print('LOOCV Confusion Matrix:\n', conf_mat_svm)

fig, ax = plot_confusion_matrix(conf_mat=conf_mat_svm, class_names=class_namess)
plt.title('LOOCV Confusion Matrix')
plt.show()

precision_svm = precision_score(loo_true, loo_pred, average=None)
recall_svm = recall_score(loo_true, loo_pred, average=None)
accuracy_svm = accuracy_score(loo_true, loo_pred)
f1_svm = f1_score(loo_true, loo_pred, average='macro')
print(f'SVM Precision: {precision_svm}, Recall: {recall_svm}, Accuracy: {accuracy_svm}, F1 Score: {f1_svm}')



#clf_svm = SVC(kernel='rbf', probability=True,C=1,gamma='auto',cache_size=1000,)
clf_svm.fit(hog_features_train_scaled, y_train)

y_pred_svm = clf_svm.predict(hog_features_test_scaled)
x_pred_svm = clf_svm.predict(hog_features_train_scaled)
print("")
print("")
print(y_train)
print("")
print(x_pred_svm)

conf_mat_svmx = confusion_matrix(y_train, x_pred_svm)
print('SVM Confusion Matrix:\n', conf_mat_svmx)

fig, ax = plot_confusion_matrix(conf_mat=conf_mat_svmx, class_names=class_namess)
plt.title('SVM Confusion Matrix')
plt.show()

#precision_svmx = precision_score(y_train, x_pred_svm, average=None)
#recall_svmx = recall_score(y_train, x_pred_svm, average=None)
#accuracy_svmx = accuracy_score(y_train, x_pred_svm)
#f1_svmx = f1_score(y_train, x_pred_svm, average='macro')
#print(f'SVM Precision: {precision_svmx}, Recall: {recall_svmx}, Accuracy: {accuracy_svmx}, F1 Score: {f1_svmx}')

# SVM Classifier Performance
print("")
print("")
print(y_test)
print("")
print(y_pred_svm)

conf_mat_svm = confusion_matrix(y_test, y_pred_svm)
print('SVM Confusion Matrix:\n', conf_mat_svm)

fig, ax = plot_confusion_matrix(conf_mat=conf_mat_svm, class_names=class_namess)
plt.title('SVM Confusion Matrix')
plt.show()

precision_svm = precision_score(y_test, y_pred_svm, average=None)
recall_svm = recall_score(y_test, y_pred_svm, average=None)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm, average='macro')
print(f'SVM Precision: {precision_svm}, Recall: {recall_svm}, Accuracy: {accuracy_svm}, F1 Score: {f1_svm}')
