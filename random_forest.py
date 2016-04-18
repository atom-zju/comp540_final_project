import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
import utils

cifar10_data, cifar10_labels, cifar10_test_data, cifar10_test_label = utils.get_cifar10()

# print cifar10_labels.shape, cifar10_labels.dtype

# use random forest do prediction
# clf = RandomForestClassifier(n_estimators=50,max_depth=500)
# clf.fit(cifar10_data,cifar10_labels)
# print clf.score(cifar10_test_data,cifar10_test_label)


# use extra trees do prediction
# clf = ExtraTreesClassifier(n_estimators=300,max_depth=1500)
# clf.fit(cifar10_data,cifar10_labels)
# print clf.score(cifar10_test_data,cifar10_test_label)

# use ada boost do prediction
# clf = AdaBoostClassifier(n_estimators=300,learning_rate=0.1)
# clf.fit(cifar10_data,cifar10_labels)
# print clf.score(cifar10_test_data,cifar10_test_label)


# use k means do prediction
# clf = KMeans(n_clusters=10,max_iter=500, n_jobs=-1)
# clf.fit(cifar10_data,cifar10_labels)
# print clf.score(cifar10_test_data,cifar10_test_label)

# use knn to do prediction
# clf = KNeighborsClassifier(n_neighbors=50,n_jobs=-1)
# clf.fit(cifar10_data,cifar10_labels)
# print "start to predict"
# pred_labels = clf.predict(cifar10_test_data)
# print np.mean(pred_labels == cifar10_test_label)

# use SVM(one-vs-rest) to do prediction
clf = SVC(C=1.0,decision_function_shape='ovr')
clf.fit(cifar10_data,cifar10_labels)
print clf.score(cifar10_test_data,cifar10_test_label)

# bagging = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
