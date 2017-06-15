import pandas as pd
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
import numpy as np
import math

columns = list(range(1,31))
file = pd.read_csv("wdbc.data.txt",delimiter=',',header=None,usecols=columns)

def labels2index(y):
    classes = y.unique()
    classes_dict = {c:d for d,c in enumerate(classes)}
    return [classes_dict[y_] for y_ in y.values]

def feature_scaling(x):
   # print x
   for i in xrange(len(x.columns)):
       max = x.ix[i].max()
       min = x.ix[i].min()
       row = []
       for d in x.ix[i].values:
           scaled_value = ((d - min)/(max - min))
           if min == max:
               print d,min,max
           row.append(float(scaled_value))
       x.loc[i] = pd.DataFrame(row,dtype='float64').loc[0]
       # print x[20]
   return x


testset_split = 0.2

file = shuffle(file,random_state=42)

x = feature_scaling(file.ix[:,2:30])
y = labels2index(file[1])

print x.isnull().any()

split_index = int(len(x)*(1-testset_split))

x_train = x[:split_index]
y_train = y[:split_index]

x_test = x[split_index:]
y_test = y[split_index:]



km = KMeans(n_clusters=2,random_state=2,n_init=20,max_iter=300)
km.fit(x_train,y_train)
predictions =  km.predict(x_test)

print classification_report(y_test,predictions)

