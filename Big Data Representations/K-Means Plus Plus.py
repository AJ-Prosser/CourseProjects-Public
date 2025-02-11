from numpy.random import seed
import numpy as np
rngval=12
seed(rngval)
rng = np.random.default_rng(rngval)

import csv
import sys
import time

def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

data = np.genfromtxt(sys.argv[1], delimiter=',')
k = int(sys.argv[2])
r = int(sys.argv[3])

currerror = sum(sum((abs(data))))**3
#print(currerror)

realLabels = np.zeros(len(data))
labels = np.zeros(len(data))
#print(labels)

uniquelabels = range(k)

itercount = 1

for i in range(r):
    #select initial points
    centroids = rng.choice(data, 1, axis=0)
    for iter in range(1,k):
        #print("centroids = ", centroids)
        #print("iter = ", iter)
        kppweights = np.array([min([distance(point, center) for center in centroids]) for point in data])
        kppweights = kppweights/np.sum(kppweights)
        #print(kppweights)
        centroids = np.append(centroids, rng.choice(data, 1, p=kppweights), axis=0)
    #print(centroids)

    #loop until labels converge
    converged = False
    while converged == False:
        #for all point indexes in the data
        for pointidx in range(len(data)):
            bestdist = sum(sum(data))**3
            #find the closest cluster, set it as the center
            for groupnum in range(k):
                #print(centroids)
                #print(data[pointidx])
                if bestdist > distance(data[pointidx], centroids[groupnum]):
                    bestdist = distance(data[pointidx], centroids[groupnum])
                    labels[pointidx] = groupnum


        #select new centers
        newcentroids = centroids
        for label in uniquelabels:
            subset = data[labels == label]
            #print(label)
            #print(subset)
            #time.sleep(2)
            newcentroids[label] = np.mean(subset, axis=0)
            #print(newcentroids)
        
        #check for convergence
        if np.array_equal(newcentroids, centroids):
            converged = True
        #set new centers
        centroids = newcentroids
    

    #print(labels)

    testerror = 0.0
    for label in uniquelabels:
        subset = data[labels == label]
        #print(label)
        #print(subset)C
        subsetAvg = np.mean(subset, axis=0)
        #print(subsetAvg)
        for point in subset:
            testerror += np.linalg.norm(point - subsetAvg)**2

    if testerror < currerror:
        realLabels = labels
        #print("improvement from ", currerror, " to ", testerror, " on iteration ", i+1)
        currerror = testerror
        
#print(realLabels)

with open(sys.argv[4], 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    for val in realLabels:
        writer.writerow([int(val)])

print("Quantization error: ", currerror)