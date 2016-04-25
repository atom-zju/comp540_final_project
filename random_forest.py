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
N = cifar10_test_data.shape[0]
# X = np.zeros((N, 32, 32,3), dtype=np.uint8)
#
# for img_idx in range(N):
#     for chn in range(3):
#         X[img_idx,:,:,chn] = np.reshape(cifar10_test_data[img_idx,chn*1024:(chn+1)*1024],(32,32))
#
# X_test_patched = utils.extract_patches(X)
# np.save('X_test_patched',X_test_patched)
# X_test_patched = np.load('X_test_patched.npy')
#
# X_test_transformed = np.zeros((X_test_patched.shape[0],X_test_patched.shape[1],X_test_patched.shape[2],1),dtype=np.uint8)
# centroid = np.load('centroid.npy')
# print centroid.shape
# for img_dix in range(N):
#     for i in range(X_test_transformed.shape[1]):
#         for j in range(X_test_transformed.shape[2]):
#             feature_label = np.argmin(np.sum((centroid - X_test_patched[img_dix,i,j,:][None,:])**2,axis=1))
#             X_test_transformed[img_dix,i,j,:] = feature_label
# np.save('X_test_transformed',X_test_transformed)

# print centroid.shape


X_transformed = np.load("X_transformed.npy")
X_transformed = np.reshape(X_transformed, (X_transformed.shape[0], X_transformed.shape[1]*X_transformed.shape[2]*X_transformed.shape[3]))
X_test_transformed = np.load("X_test_transformed.npy")
X_test_transformed = np.reshape(X_test_transformed, (X_test_transformed.shape[0], X_test_transformed.shape[1]*X_test_transformed.shape[2]*X_test_transformed.shape[3]))
# clf = ExtraTreesClassifier(n_estimators=300,max_depth=1500)
# clf.fit(X_transformed,cifar10_labels)
# print clf.score(X_test_transformed,cifar10_test_label)

# use ada boost do prediction
# clf = AdaBoostClassifier(n_estimators=300,learning_rate=0.1)
# clf.fit(cifar10_data,cifar10_labels)
# print clf.score(cifar10_test_data,cifar10_test_label)


# use k means do prediction
# clf = KMeans(n_clusters=10,max_iter=500, n_jobs=-1)
# clf.fit(X_transformed,cifar10_labels)
# print clf.score(X_test_transformed,cifar10_test_label)

# use knn to do prediction
# clf = KNeighborsClassifier(n_neighbors=50,n_jobs=-1)
# clf.fit(X_transformed,cifar10_labels)
# print "start to predict"
# pred_labels = clf.predict(X_test_transformed)
# print np.mean(pred_labels == cifar10_test_label)

# use SVM(one-vs-rest) to do prediction
clf = SVC(C=1.0,decision_function_shape='ovr')
clf.fit(X_transformed,cifar10_labels)
print "started to do the prediction"
print clf.score(X_test_transformed,cifar10_test_label)

# bagging = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
