#!/usr/bin/env python3

import numpy as np
from itertools import product
from skimage import io,color
def clustering(method,image,rate=1):
    """
    method : sklearn clustering method
    matrix : fitting data
    """
    lab = color.rgb2lab(image)#use l*a*b* color-space(not RGB)
    mapping = np.meshgrid(np.arange(lab.shape[1]),np.arange(lab.shape[0]))#[0,1,2,...,0,1,2,...],[0,0,0,...,0,1,1,1,...]
    data = np.concatenate((lab*rate,*[m[:,:,None] for m in mapping]),axis = 2)\
            .reshape(lab.shape[0]*lab.shape[1],lab.shape[2]+len(mapping))#[[l,a,b,x,y],...]
    model = method.fit(data)#clustering
    labels = model.labels_.reshape(lab.shape[0:2])#reshape result
    segments = [labels == c for c in range(method.n_clusters)]
    return [image*segment[:,:,None] for segment in segments]

if __name__ == '__main__':
    #parameters
    from sklearn.cluster import KMeans,AgglomerativeClustering
    method = KMeans(n_clusters=12,   #k
                    max_iter=600,   #number of iteration
                    verbose=True,   #verbosity mode
                    n_jobs=-1)       #parallel computing (-1:full)

    #read image
    import sys
    from PIL import Image
    img = np.array(Image.open(sys.argv[1]))
    #clustering
    result = clustering(method,img,rate=1e3)
    #show result
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(ncols=4,nrows=4,figsize=(20,5))
    axes = axes.flatten(order='C')
    for i,res in enumerate(result):
        axes[i].imshow(res)
        axes[i].set_title('clustered_'+str(i))
        axes[i].get_xaxis().set_visible(False)
        axes[i].get_yaxis().set_visible(False)
    axes[-1].imshow(img)
    axes[-1].set_title('origin')
    axes[-1].get_xaxis().set_visible(False)
    axes[-1].get_yaxis().set_visible(False)
    plt.show()
    #end
    sys.exit(0)
