#!/usr/bin/env python3

import numpy as np
from itertools import product
from skimage import io,color
def clustering(method,image,rate=1):
    """
    method : sklearn clustering method
    matrix : fitting data
    """
    lab = color.rgb2lab(image)
    mapping = np.meshgrid(np.arange(lab.shape[1]),np.arange(lab.shape[0]))
    print(image.shape,mapping[1].shape)
    data = np.concatenate((lab*rate,mapping[0][:,:,None],mapping[1][:,:,None]),axis = 2).reshape(lab.shape[0]*lab.shape[1],lab.shape[2]+2)
    print(data)
    model = method.fit(data)#clustering
    labels = model.labels_.reshape(lab.shape[0:2])
    clusters = [labels == c for c in range(method.n_clusters)]
    return [image*cluster[:,:,None] for cluster in clusters]

def clustered_clustering(method1,method2,image,rate1=1,rate2=1):
    result1 = clustering(method1,image,rate=rate1)
    result2 = list()
    for i in range(len(result1)):
        print(i)
        result2.extend(clustering(method2,result1[i],rate=rate2))
    return result1,result2


if __name__ == '__main__':
    #parameters
    clusters = 8

    import sys
    from PIL import Image
    img = np.array(Image.open(sys.argv[1]))
    from sklearn.cluster import KMeans
    result = clustering(KMeans(n_clusters=clusters),img,rate=1e10)
    import matplotlib.pyplot as plt
    for i,res in enumerate(result):
        plt.subplot(3,3,i+1)
        plt.imshow(res)
    plt.subplot(3,3,9)
    plt.imshow(img)
    plt.show()

    sys.exit(0)


    res1,res2 = clustered_clustering(KMeans(n_clusters=3),KMeans(n_clusters=3),img,rate1=1e2,rate2=1e5)
    import matplotlib.pyplot as plt
    for i,res in enumerate(res1):
        plt.subplot(4,3,i+1)
        plt.imshow(res)
    for i,res in enumerate(res2):
        plt.subplot(4,3,i+4)
        plt.imshow(res)
    plt.show()
