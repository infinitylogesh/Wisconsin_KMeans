import pandas as pd
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.metrics import classification_report,accuracy_score
from sklearn.preprocessing import scale


columns = list(range(1,31))

# Reading the dataset content as csv
file = pd.read_csv("wdbc.data.txt",delimiter=',',header=None,usecols=columns)

# function that converts labels to indexes - 0 / 1
def labels2index(y):
    classes = y.unique()
    classes_dict = {c:d for d,c in enumerate(classes)}
    return [classes_dict[y_] for y_ in y.values]

# Dataset split
testset_split = 0.3

# File content is shuffled
file = shuffle(file,random_state=42)

# Feature scaling is done as many of the columns in the dataset have higher deviations in ranges
x = scale(file.ix[:,2:30])
y = labels2index(file[1])


split_index = int(len(x)*(1-testset_split))

x_train = x[:split_index]
y_train = y[:split_index]

x_test = x[split_index:]
y_test = y[split_index:]

# KMeans
km = KMeans(n_clusters=2,random_state=2,n_init=20,max_iter=300,n_jobs=-1)
km.fit(x_train,y_train)
predictions =  km.predict(x_test)

print "**** Confusion matrix ****"
print classification_report(y_test,predictions)
print "Accuracy: ",accuracy_score(y_test,predictions)

