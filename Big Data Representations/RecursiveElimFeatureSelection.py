# Recursive elimination feature selection, linear discriminant
# AJ Prosser

import csv
import sys
import numpy as np

# get filein name

#print ('argument list', sys.argv)

data = np.genfromtxt(sys.argv[1], delimiter=',')
labels = np.genfromtxt(sys.argv[2], delimiter=',')
uniquelabels = np.unique(labels)

def solution(x, y):
    B = np.matmul(x.T,x)
    h = np.matmul(x.T, y)
    #print(B)
    #print(h)
    a = np.linalg.solve(B, h)
    #print(a)
    return a


selection = data
while len(selection.T) > 2:
    coeff = solution(selection, labels)
    #print(coeff)

    #print(selection[0:3])
    idx = np.argsort(coeff)[::-1]
    #print(idx)
    selection = np.abs(selection[:, idx])
    #print(selection[0:3])
    selection = selection[:, :-1]
    #print(selection)


#dataout = np.matmul(data, selection)

with open(sys.argv[3], 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(selection)