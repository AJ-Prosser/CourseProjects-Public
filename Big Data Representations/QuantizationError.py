from numpy.random import seed
seed(12)

import csv
import sys
import numpy as np

# get filein name

#print ('argument list', sys.argv)

data = np.genfromtxt(sys.argv[1], delimiter=',')
labels = np.genfromtxt(sys.argv[2], delimiter=',')

#print(data)
#print(labels)

uniquelabels = np.unique(labels)

totalerror = 0.0
for label in uniquelabels:
    subset = data[labels == label]
    #print(label)
    #print(subset)
    subsetAvg = np.mean(subset, axis=0)
    for point in subset:
        totalerror += np.linalg.norm(point - subsetAvg)**2

print("Quantization error: ", totalerror)