# Fischer selection
# AJ Prosser

import csv
import sys
import numpy as np

# get filein name

#print ('argument list', sys.argv)

data = np.genfromtxt(sys.argv[1], delimiter=',')
labels = np.genfromtxt(sys.argv[2], delimiter=',')
uniquelabels = np.unique(labels)

def scatter(x):
    sum = 0
    mean = x.mean()
    for element in x:
        sum += np.square(element - mean)
    return sum


def betweenscatter(j):
    totalAvg = np.mean(j, axis=0)
    B = 0
    for label in uniquelabels:
        subset = j[labels == label]
        subsetAvg = np.mean(subset)
        #print("subsetavg: ", subsetAvg, "\nTotalavg: ", totalAvg)
        avgdiff = subsetAvg - totalAvg

        B += len(subset)*np.square(avgdiff)
    return B

def withinscatter(j):
    W = 0
    for label in uniquelabels:
        subset = j[labels == label]
        #print(subset)
        W += scatter(subset)
    return W

def fisher(j,y):
    return betweenscatter(j)/withinscatter(j)

fisherlist = []
for column in data.T:
    #print(column)
    fisherlist.append(abs(fisher(column,labels)))

#zprint(fisherlist)

idx = np.argsort(fisherlist)[::-1]
#print(idx)
#pearsonlist = pearsonlist[idx]
sortedcols = data[:, idx]

#print(sortedcols)

selection = sortedcols[:,0:2]

#dataout = np.matmul(data, selection)

with open(sys.argv[3], 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(selection)