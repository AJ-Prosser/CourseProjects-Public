import csv
import sys
import numpy as np


data = np.genfromtxt(sys.argv[1], delimiter=',')

#distance matrix
i=0
j=0
#print(len(data))
alpha = float(sys.argv[3])
distances = np.zeros((len(data),len(data)))
for Xi in data:
    for Xj in data: 
        distances[i, j] = np.power(np.sum(np.square(Xi-Xj)),alpha/2)
        j+=1
    i+=1
    j=0
#print(distances)

#distances = np.matrix('0 1 1 1; 1 0 1 1; 1 1 0 1; 1 1 1 0')


#gram matrix
i=0
j=0
gram = np.zeros_like(distances)
amatrix = np.square(distances)
#print(amatrix)
for i in range(0,len(data)):
    for j in range(0,len(data)):
        gram[i, j] = -(amatrix[i, j])/2 + np.sum(amatrix[i])/(2 * len(data)) + np.sum(amatrix[:,j])/(2 * len(data)) - np.sum(np.sum(amatrix))/(2 * len(data) * len(data))

print("Gram Matrix: ",gram)

evals,evecs = np.linalg.eigh(gram) 

idx = np.argsort(evals)[::-1] # sort in reverse order
evals = evals[idx]
evecs = evecs[:,idx]

print("Eigenvalues= \n", evals, "\nEigenvectors= \n", evecs)

evals[0]

outmatrixT = np.zeros((2, len(data)))
#print(max(np.sqrt(evals[0]),0))
#outmatrixT[0] = np.multiply(evecs[0],max(np.sqrt(evals[0]),0))
#outmatrixT[1] = np.multiply(evecs[1],max(np.sqrt(evals[1]),0))
outmatrixT[0] = np.multiply(evecs[0],np.sqrt(evals[0]))
outmatrixT[1] = np.multiply(evecs[1],np.sqrt(evals[1]))
#print(outmatrixT)

with open(sys.argv[2], 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(np.transpose(outmatrixT))

