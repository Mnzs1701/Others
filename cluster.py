
from sklearn import cluster , datasets
import pandas as pd
import numpy

import matplotlib.pyplot as plt
x=pd.read_csv("chennai.csv")
def dist(xa,ya,xb,yb):
    dist = numpy.sqrt((xa-xb)*(xa-xb) + (ya-yb)*(ya-yb))
    return dist
# import pdb; pdb.set_trace()
ram,prc=[],[]
for i in x.itertuples():
    #print(i)
    #import pdb; pdb.set_trace()
    ram.append(i[21])
    prc.append((i[14])/1250)

# n=len(clk)
# prc=[0]*n
plt.scatter(ram,prc)
plt.show()

k_means = cluster.KMeans(n_clusters=3)
# k_means.fit(X)
data=pd.DataFrame({
    'x':ram,
    'y':prc
})

#import pdb; pdb.set_trace()
print(k_means.fit(data))

labels = k_means.predict(data)
centroids = k_means.cluster_centers_
#import pdb; pdb.set_trace()

fig = plt.figure(figsize=(5, 5))
colmap = {1: 'r', 2: 'g', 3: 'b'}
colors = map(lambda x: colmap[x+1], labels)
# import pdb; pdb.set_trace()
for i in data.itertuples():
    if dist(i[1],i[2],centroids[0][0],centroids[0][1])== min(dist(i[1],i[2],centroids[0][0],centroids[0][1]),dist(i[1],i[2],centroids[1][0],centroids[1][1]),dist(i[1],i[2],centroids[2][0],centroids[2][1])):
        plt.scatter(i[1],i[2],color='r')
    elif dist(i[1],i[2],centroids[1][0],centroids[1][1])== min(dist(i[1],i[2],centroids[0][0],centroids[0][1]),dist(i[1],i[2],centroids[1][0],centroids[1][1]),dist(i[1],i[2],centroids[2][0],centroids[2][1])):
        plt.scatter(i[1],i[2],color='b')
    else:
        plt.scatter(i[1],i[2],color='g')
# plt.scatter(data['x'], data['y'], color=colors, alpha=0.5)
# plt.show()
# for idx, centroid in enumerate(centroids):
#     plt.scatter(*centroid, color=colmap[idx+1])
plt.xlim(0, 5.5)
plt.ylim(0, 4.1)
plt.show()
