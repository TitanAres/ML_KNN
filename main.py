import kNN

group,labels = kNN.createDataset()
print kNN.classify_kNN([0, 0], group, labels, 3)

features, labels = kNN.filetomatrix("datingTestSet2.txt")
print features
print labels
