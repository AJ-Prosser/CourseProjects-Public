# minimize ratio of between-class scatter and within-class scatter
# AJ Prosser

import csv
import sys
import numpy as np

# get filein name

#print ('argument list', sys.argv)

data = np.genfromtxt(sys.argv[1], delimiter=',')
labels = np.genfromtxt(sys.argv[2], delimiter=',')
uniquelabels = np.unique(labels)
#print(uniquelabels)

#print(data)

#calculate mean(s)
def scattermatrix(subset):
    averages = np.mean(subset, axis=0)

    runningsum = 0
    for point in subset:
        yminusmean = point - averages
        #print(yminusmean)
        runningsum += np.outer(yminusmean,yminusmean)
    return runningsum

#appended = np.append(data, labels.reshape(-1, 1), axis=1)

#calc W
first = True
for label in uniquelabels:
    subset = data[labels == label]
    #print(subset)
    if first:
        W = scattermatrix(subset)
        first = False
    else: W += scattermatrix(subset)

#calc B
totalAvg = np.mean(data, axis=0)
runningsum = 0
for label in uniquelabels:
    subset = data[labels == label]
    subsetAvg = np.mean(subset, axis=0)
    avgdiff = subsetAvg - totalAvg

    runningsum += len(subset)*np.outer(avgdiff,avgdiff)

B = runningsum

#print(B)

#print(W)

evals, evecs = np.linalg.eigh(W)

#print(evals)
lambdahalf = np.diag(np.power(evals,-1/2))
C = np.matmul(np.matmul(np.matmul(np.matmul(lambdahalf,evecs.T),B),evecs),lambdahalf)

#print(C)

evalsC, evecsC = np.linalg.eigh(C)
evecsCt = np.matmul(np.matmul(evecs,lambdahalf),evecsC)

idx = np.argsort(evalsC)[::-1] # sort in reverse order
evalsC = evalsC[idx]
evecsCt = evecsCt[:,idx]

#minimize
selection = evecs[:,-2:]

dataout = np.matmul(data, selection)

with open(sys.argv[3], 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(dataout)

