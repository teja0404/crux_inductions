# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 22:22:25 2019

@author: TEJA
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans,AgglomerativeClustering
import matplotlib.pyplot as plt



dataset=pd.read_csv(r"C:\Users\TEJA\Desktop\Mall_Customers.csv")
X=dataset.iloc[:,[3,4]]
##### k-means ######
#finding the optimal n vallue using elbow method

eucl_dist_sum=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++")
    kmeans.fit(X)
    eucl_dist_sum.append(kmeans.inertia_)
    
    
plt.plot(range(1,11),eucl_dist_sum)#from this elbow plot we can find the optimal no of clusters
n_clusters=5

#fitting dataset to kmeans
kmeans=KMeans(n_clusters=5,init="k-means++",max_iter=500)
predicted=kmeans.fit_predict(X)

#visualising the results
X=np.array(X)
plt.title("k-means")
plt.scatter(X[predicted==0,0],X[predicted==0,1],c="red")
plt.scatter(X[predicted==1,0],X[predicted==1,1],c="blue")
plt.scatter(X[predicted==2,0],X[predicted==2,1],c="green")
plt.scatter(X[predicted==3,0],X[predicted==3,1],c="magenta")
plt.scatter(X[predicted==4,0],X[predicted==4,1],c="black")

####hierarchial clustering#####

""""In general manhattan dist is greater than eucledian"""

"""complete linkages considers the max distance betweeen the clusters to form new clusters"""




"""The distance matrix used here is L1 which uses manhattan distance to find the distance between the points
which is greater than eucledian.outliers will play major role in if we use complete linkage as it takes max dist b/w clusters.
using average linkage will give better results than complete linkage because we take the average of distance of all the points,
so that the effect of outliers will be demesified"""


aggclus1=AgglomerativeClustering(n_clusters=5,affinity="l1",linkage="average")
predicted2=aggclus1.fit_predict(X)

plt.title("l1 using average linkage")
plt.scatter(X[predicted2==0,0],X[predicted2==0,1],c="red")
plt.scatter(X[predicted2==1,0],X[predicted2==1,1],c="blue")
plt.scatter(X[predicted2==2,0],X[predicted2==2,1],c="green")
plt.scatter(X[predicted2==3,0],X[predicted2==3,1],c="magenta")
plt.scatter(X[predicted2==4,0],X[predicted2==4,1],c="black")




"""while we use L2 affinity which uses eucledian distance to find the similarity between the points which is less than eucledian
if we use average distance  the outliers will remain isolated.So complete linkage will give better result average"""

aggclus2=AgglomerativeClustering(n_clusters=5,affinity="l2",linkage="complete")
predicted3=aggclus2.fit_predict(X)


plt.title("l2 using complete linkage")
plt.scatter(X[predicted3==0,0],X[predicted3==0,1],c="red")
plt.scatter(X[predicted3==1,0],X[predicted3==1,1],c="blue")
plt.scatter(X[predicted3==2,0],X[predicted3==2,1],c="green")
plt.scatter(X[predicted3==3,0],X[predicted3==3,1],c="magenta")
plt.scatter(X[predicted3==4,0],X[predicted3==4,1],c="black")


"""considering above three we can say that agglomerative clustering performed better than k-means"""
"""while l1 and l2 gave almost equal results with l2 some much better result"""