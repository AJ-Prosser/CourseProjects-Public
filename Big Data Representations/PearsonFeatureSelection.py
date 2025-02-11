# Pearson Feature selection
# AJ Prosser

import csv
import sys
import numpy as np

# get filein name

#print ('argument list', sys.argv)

data = np.genfromtxt(sys.argv[1], delimiter=',')
labels = np.genfromtxt(sys.argv[2], delimiter=',')

def scatter(x):
    sum = 0
    mean = x.mean()
    for element in x:
        sum += np.square(element - mean)
    return sum

def cjy(j, y):
    sum = 0
    jmean = j.mean()
    ymean = y.mean()
    for idx in range(len(y)):
        sum += (j[idx] - jmean)*(y[idx] - ymean)
    return sum

def pearson(j, y):
    return cjy(j, y)/np.sqrt(scatter(j)*scatter(y))

pearsonlist = []
for column in data.T:
    #print(column)
    pearsonlist.append(abs(pearson(column,labels)))

#print(pearsonlist)

idx = np.argsort(pearsonlist)[::-1]
#print(idx)
#pearsonlist = pearsonlist[idx]
sortedcols = data[:, idx]

#print(sortedcols)

selection = sortedcols[:,0:2]

#dataout = np.matmul(data, selection)

with open(sys.argv[3], 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(selection)