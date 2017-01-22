
def clustered_clustering(method1,method2,image,rate1=1,rate2=1):
    result1 = clustering(method1,image,rate=rate1)
    result2 = list()
    for i in range(len(result1)):
        print(i)
        result2.extend(clustering(method2,result1[i],rate=rate2))
    return result1,result2

#memo
res1,res2 = clustered_clustering(KMeans(n_clusters=3),KMeans(n_clusters=3),img,rate1=1e2,rate2=1e5)
import matplotlib.pyplot as plt
for i,res in enumerate(res1):
    plt.subplot(4,3,i+1)
    plt.imshow(res)
for i,res in enumerate(res2):
    plt.subplot(4,3,i+4)
    plt.imshow(res)
plt.show()
