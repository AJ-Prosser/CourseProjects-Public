# maximize between-class scatter
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

totalAvg = np.mean(data, axis=0)
runningsum = 0
for label in uniquelabels:
    subset = data[labels == label]
    subsetAvg = np.mean(subset, axis=0)
    avgdiff = subsetAvg - totalAvg

    runningsum += len(subset)*np.outer(avgdiff,avgdiff)

B = runningsum

#print(B)

evals,evecs = np.linalg.eigh(B) 

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

