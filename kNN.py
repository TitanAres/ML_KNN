from numpy import *
import operator

#function that creates Data
def createDataset():
  group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
  labels = ['A', 'A', 'B', 'B']
  return group, labels

#classifer
def classify_kNN(input, dataset, labels, k):
  datasetSize = dataset.shape[0]
  #diffs between input and all dataset per feature
  diffMat = tile(input, (datasetSize, 1)) - dataset
  #calculate euclidean distance 
  sqdiffMat = diffMat**2;
  sqdistances = sqdiffMat.sum(axis = 1)
  distances = sqdistances**0.5
  #sort distance value
  sortdis = distances.argsort()
  classcounts = {}
  for i in xrange(k):
    temp_label = labels[sortdis[i]]
    classcounts[temp_label] = classcounts.get(temp_label, 0) + 1
  #get the final label
  sortclasscounts = sorted(classcounts.iteritems(), key = operator.itemgetter(1), reverse = True)
  return sortclasscounts[0][0] 

#trans .txt file to matrix used to training
def filetomatrix(filename):
  fr = open(filename)
  infstream = fr.readlines()
  line_num = len(infstream)
  features = zeros((line_num, 3))
  labels = []
  start_row = 0
  for line in infstream:
    temp_line = line.strip()
    temp_split = line.split('\t')
    features[start_row,:] = temp_split[0:3]
    labels.append(int(temp_split[-1]))
    start_row += 1
  return features, labels
