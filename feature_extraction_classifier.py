import numpy as np
import utils
from sklearn.cluster import KMeans
from sklearn.svm import SVC


def transform_features(X_patched,feature_num=100):
    X_transformed = np.zeros((X_patched.shape[0],X_patched.shape[1],X_patched.shape[2],1),dtype=np.uint8)
    total_features = X_patched.shape[0]*X_patched.shape[1]*X_patched.shape[2]
    features = np.zeros((total_features, X_patched.shape[3]), dtype=np.float16)
    global_idx = 0
    for idx in range(X_patched.shape[0]):
        for i in range(X_patched.shape[1]):
            for j in range(X_patched.shape[2]):
                features[global_idx,:] = X_patched[idx,i,j,:]
                global_idx += 1
    print "start to do k means"
    kmean_classifier = KMeans(n_clusters=feature_num, n_jobs=-1, max_iter=100,n_init=3)
    kmean_classifier.fit(features)
    print "start build x transformed"
    np.save("centroid",kmean_classifier.cluster_centers_)

    for idx in range(X_patched.shape[0]):
        for i in range(X_patched.shape[1]):
            for j in range(X_patched.shape[2]):
                feature_label = kmean_classifier.predict(X_patched[idx,i,j,:])
                X_transformed[idx,i,j,:] = feature_label

    return X_transformed

cifar10_data, cifar10_labels, cifar10_test_data, cifar10_test_label = utils.get_cifar10()

N = cifar10_data.shape[0]
X = np.zeros((N, 32, 32,3), dtype=np.uint8)

for img_idx in range(N):
    for chn in range(3):
        X[img_idx,:,:,chn] = np.reshape(cifar10_data[img_idx,chn*1024:(chn+1)*1024],(32,32))
y = cifar10_labels

# create feature patches
# print "start to do extraction"
# # X_patched = extract_patches(X)
# # np.save('X_patched',X_patched)
# X_patched = np.load('X_patched.npy')
# print X_patched.shape

# print "start to do transform"
# X_transformed = transform_features(X_patched)
# np.save('X_transformed',X_transformed)

X_transformed = np.load('X_transformed.npy')
print X_transformed.shape
X_transformed = np.reshape(X_transformed, (X_transformed.shape[0],
                                           X_transformed.shape[1]*X_transformed.shape[2]*X_transformed.shape[3]))

# use SVM(one-vs-rest) to do prediction
clf = SVC(C=1.0,decision_function_shape='ovr')
clf.fit(X_transformed,cifar10_labels)
print "started to do the prediction"
print clf.score(X_transformed,cifar10_labels)

