# maximize within-class scatter
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

first = True
for label in uniquelabels:
    subset = data[labels == label]
    #print(subset)
    if first:
        W = scattermatrix(subset)
        first = False
    else: W += scattermatrix(subset)

#print(W)

evals,evecs = np.linalg.eigh(W) 

idx = np.argsort(evals)[::-1] # sort in reverse order
evals = evals[idx]
evecs = evecs[:,idx]

#print("Eigenvalues= \n", evals, "\nEigenvectors= \n", evecs)

#maximize
selection = evecs[:,0:2]

dataout = np.matmul(data, selection)

with open(sys.argv[3], 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(dataout)

