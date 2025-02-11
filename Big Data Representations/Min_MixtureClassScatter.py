# minimize mixture class scatter
# AJ Prosser

import csv
import sys
import numpy as np

# get filein name

#print ('argument list', sys.argv)

data = np.genfromtxt(sys.argv[1], delimiter=',')

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

M = scattermatrix(data)

evals,evecs = np.linalg.eigh(M) 

idx = np.argsort(evals)[::-1] # sort in reverse order
evals = evals[idx]
evecs = evecs[:,idx]

#print("Eigenvalues= \n", evals, "\nEigenvectors= \n", evecs)

#minimize
selection = evecs[:,-2:]

dataout = np.matmul(data, selection)

with open(sys.argv[3], 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(dataout)

#print(M)

