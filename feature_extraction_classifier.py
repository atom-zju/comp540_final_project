import numpy as np
import utils
from sklearn import preprocessing
from sklearn.cluster import KMeans

def extract_patches(X, stride=2, w_size=4):
    N = X.shape[0]
    K = (X.shape[1]-w_size)/stride + 1
    chns = X.shape[-1]
    X_patched = np.zeros((N, K, K, w_size*w_size*chns), dtype=float)
    # do whitening here
    for idx in range(N):
        for i in range(K):
            for j in range(K):
                X_patched[idx,i,j,:] = np.reshape(X[idx,i*stride:i*stride+w_size,j*stride:j*stride+w_size,:],(w_size*w_size*chns,))
                X_patched[idx,i,j,:] = preprocessing.scale(X_patched[idx,i,j,:])
    return X_patched

def transform_features(X_patched,feature_num=400):
    X_transformed = np.zeros((X_patched.shape[0],X_patched.shape[1],X_patched.shape[2],feature_num))
    total_features = X_patched.shape[0]*X_patched.shape[1]*X_patched.shape[2]
    features = np.zeros((total_features, X_patched.shape[3]), dtype=float)
    global_idx = 0
    for idx in range(X_patched.shape[0]):
        for i in range(X_patched.shape[1]):
            for j in range(X_patched.shape[2]):
                features[global_idx,:] = X_patched[idx,i,j,:]
                global_idx += 1

    kmean_classifier = KMeans(n_clusters=feature_num, n_jobs=-1)
    kmean_classifier.fit(features)

    for idx in range(X_patched.shape[0]):
        for i in range(X_patched.shape[1]):
            for j in range(X_patched.shape[2]):
                feature_label = kmean_classifier.predict(X_patched[idx,i,j,:])
                X_transformed[idx,i,j,feature_label] = 1

    return X_transformed

cifar10_data, cifar10_labels, cifar10_test_data, cifar10_test_label = utils.get_cifar10()

N = cifar10_data.shape[0]
X = np.zeros((N, 32, 32,3), dtype=np.uint8)

for img_idx in range(N):
    for chn in range(3):
        X[img_idx,:,:,chn] = np.reshape(cifar10_data[img_idx,chn*1024:(chn+1)*1024],(32,32))
y = cifar10_labels

# create feature patches
print "start to do extraction"
X_patched = extract_patches(X)
print "start to do transform"
X_transformed = transform_features(X_patched)

print X_transformed.shape

