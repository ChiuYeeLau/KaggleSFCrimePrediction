
__author__ = 'Monkey'
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
# import matplotlib.pylab as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.metrics import make_scorer
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
# from matplotlib.colors import LogNorm
from sklearn.decomposition import PCA

from copy import deepcopy
import pdb

import csv
import time
start_time = time.time()


# import keras
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils

# In[2]:

train = pd.read_csv("train.csv")

def get_data(fn):
  data = []
  with open(fn) as f:
    reader = csv.DictReader(f)
    data = [row for row in reader]
  return data

label_train = get_data('train.csv')

labels = 'ARSON,ASSAULT,BAD CHECKS,BRIBERY,BURGLARY,DISORDERLY CONDUCT,DRIVING UNDER THE INFLUENCE,DRUG/NARCOTIC,DRUNKENNESS,EMBEZZLEMENT,EXTORTION,FAMILY OFFENSES,FORGERY/COUNTERFEITING,FRAUD,GAMBLING,KIDNAPPING,LARCENY/THEFT,LIQUOR LAWS,LOITERING,MISSING PERSON,NON-CRIMINAL,OTHER OFFENSES,PORNOGRAPHY/OBSCENE MAT,PROSTITUTION,RECOVERED VEHICLE,ROBBERY,RUNAWAY,SECONDARY CODES,SEX OFFENSES FORCIBLE,SEX OFFENSES NON FORCIBLE,STOLEN PROPERTY,SUICIDE,SUSPICIOUS OCC,TREA,TRESPASS,VANDALISM,VEHICLE THEFT,WARRANTS,WEAPON LAWS'.split(',')

label_fields = {'Category': lambda x: labels.index(x.replace(',', ''))}

def get_fields(data, fields):
  extracted = []
  for row in data:
    extract = []
    for field, f in sorted(fields.items()):
      info = f(row[field])
      if type(info) == list:
        extract.extend(info)
      else:
        extract.append(info)
    extracted.append(np.array(extract, dtype=np.float32))
  return extracted

# pdb.set_trace()

labels_int = np.array(get_fields(label_train, label_fields))
# pdb.set_trace()
# labels_int = np_utils.to_categorical(labels_int)

labels_cmp = np_utils.to_categorical(labels_int.flatten())

# pdb.set_trace()

# In[3]:

train.head()


# In[4]:

xy_scaler = preprocessing.StandardScaler()
xy_scaler.fit(train[["X","Y"]])
train[["X","Y"]] = xy_scaler.transform(train[["X","Y"]])
train = train[abs(train["Y"]) < 100]
train.index = range(len(train))


# In[5]:

def parse_time (x):
    DD = datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    time = DD.hour
    day = DD.day
    month = DD.month
    year = DD.year
    return time, day, month, year

def get_season (x):
    summer = 0
    fall = 0
    winter = 0
    spring = 0
    if (x in [5, 6, 7]):
        summer = 1
    if (x in [8, 9, 10]):
        fall = 1
    if (x in [11, 0, 1]):
        winter = 1
    if (x in [2, 3, 4]):
        spring = 1
    return summer, fall, winter, spring


# In[6]:

## replace address by log odds

def parse_data (df, logodds, logoddsPA):
    feature_list = df.columns.tolist()
    if "Descript" in feature_list:
        feature_list.remove("Descript")
    if "Resolution" in feature_list:
        feature_list.remove("Resolution")
    if "Category" in feature_list:
        feature_list.remove("Category")
    if "Id" in feature_list:
        feature_list.remove("Id")
    clean_Data = df[feature_list]
    clean_Data.index = range(len(df))
    print("Creating address features")
    address_features = clean_Data["Address"].apply(lambda x: logodds[x])
    # pdb.set_trace()
    address_features.columns = ["logodds" + str(x) for x in range(len(address_features.columns))]
    print("Parsing dates")
    clean_Data["Time"], clean_Data["Day"], clean_Data["Month"], clean_Data["Year"] = zip(*clean_Data["Dates"].apply(parse_time))
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    print("Creating one-hot variables")
    dummy_ranks_PD = pd.get_dummies(clean_Data['PdDistrict'], prefix = 'PD')
    dummy_ranks_DAY = pd.get_dummies(clean_Data["DayOfWeek"], prefix = 'DAY')
    clean_Data["IsInterection"] = clean_Data["Address"].apply(lambda x: 1 if "/" in x else 0)
    clean_Data["logoddsPA"] = clean_Data["Address"].apply(lambda x: logoddsPA[x])
    print("droping processed columns")
    clean_Data = clean_Data.drop("PdDistrict",axis = 1)
    clean_Data = clean_Data.drop("DayOfWeek",axis = 1)
    clean_Data = clean_Data.drop("Address",axis = 1)
    clean_Data = clean_Data.drop("Dates",axis = 1)
    feature_list = clean_Data.columns.tolist()
    print("joining one-hot features")
    features = clean_Data[feature_list].join(dummy_ranks_PD.ix[:,:]).join(dummy_ranks_DAY.ix[:,:]).join(address_features.ix[:,:])
    print("creating new features")
    features["IsDup"] = pd.Series(features.duplicated()|features.duplicated(take_last = True)).apply(int)
    features["Awake"] = features["Time"].apply(lambda x: 1 if (x==0 or (x>=8 and x<=23)) else 0)
    features["Summer"], features["Fall"], features["Winter"], features["Spring"] = zip(*features["Month"].apply(get_season))
    if "Category" in df.columns:
        labels = df["Category"].astype(pd.core.categorical.Categorical)
    else:
        labels = None
    return features, labels


# In[7]:

addresses = sorted(train["Address"].unique())
categories = sorted(train["Category"].unique())
C_counts = train.groupby(["Category"]).size()
A_C_counts = train.groupby(["Address","Category"]).size()
A_counts = train.groupby(["Address"]).size()
logodds = {}
logoddsPA = {}
MIN_CAT_COUNTS = 2
default_logodds = np.log(C_counts/len(train)) - np.log(1.0-C_counts/float(len(train)))

for addr in addresses:
    PA = A_counts[addr]/float(len(train))
    logoddsPA[addr] = np.log(PA)-np.log(1.-PA)
    logodds[addr] = deepcopy(default_logodds)
    for cat in A_C_counts[addr].keys():
        if (A_C_counts[addr][cat] > MIN_CAT_COUNTS) and A_C_counts[addr][cat] < A_counts[addr]:
            PA = A_C_counts[addr][cat]/float(A_counts[addr])
            logodds[addr][categories.index(cat)] = np.log(PA) - np.log(1.0-PA)
    logodds[addr] = pd.Series(logodds[addr])
    logodds[addr].index = range(len(categories))

# pdb.set_trace()

features, labels = parse_data(train, logodds, logoddsPA)

from sklearn import cross_validation
kf = cross_validation.KFold(len(features), n_folds = 10)

collist = features.columns.tolist()
scaler = preprocessing.StandardScaler()
scaler.fit(features)
features[collist] = scaler.transform(features)


# features_val = features.values
# labels_val = labels.values

'''
from sklearn import metrics

model_kfolds = []
error = []

for train_index, test_index in kf:
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = features.iloc[train_index], features.iloc[test_index]
    y_train = labels_int[train_index]
    y_train = y_train.flatten()
    y_test = labels_cmp[test_index]
    # X_train.index = range(len(X_train))
    # X_test.index = range(len(X_test))
    # y_train.index = range(len(y_train))
    # y_test.index = range(len(y_test))
    model = LogisticRegression()
    model.fit(X_train, y_train)
    pred = model.predict_proba(X_test)
    err = metrics.log_loss(y_test, pred)
    model_kfolds.append(model)
    error.append(err)
'''

# model = model_kfolds[np.argmin(error)]
# pdb.set_trace()

'''
sss = StratifiedShuffleSplit(labels, train_size = 0.9)
for train_index, test_index in sss:
    features_train,features_test = features.iloc[train_index],features.iloc[test_index]
    labels_train,labels_test = labels[train_index],labels[test_index]
features_test.index = range(len(features_test))
features_train.index = range(len(features_train))
labels_train.index = range(len(labels_train))
labels_test.index = range(len(labels_test))
features.index = range(len(features))
labels.index = range(len(labels))

# model = LogisticRegression()
# model.fit(features, labels)
'''

labels_binary = pd.get_dummies(labels)


'''
print("all", log_loss(labels, model.predict_proba(features.as_matrix())))
print("train", log_loss(labels_train, model.predict_proba(features_train.as_matrix())))
print("test", log_loss(labels_test, model.predict_proba(features_test.as_matrix())))
'''

print 'keras models'

def build_model(input_dim, output_dim, hn, dp = 0.5, layers = 1):
    model = Sequential()
    model.add(Dense(hn, input_dim=input_dim))
    model.add(Activation('relu'))
    model.add(Dropout(dp))

    model.add(Dense(hn))
    model.add(Activation('relu'))
    model.add(Dropout(dp))

    model.add(Dense(output_dim))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

'''

def build_model(input_dim, output_dim, hn, dp = 0.5, layers = 1):
    model = Sequential()
    model.add(Dense(input_dim, hn, init='glorot_uniform'))
    model.add(PReLU((hn,)))
    model.add(Dropout(dp))

    for i in range(layers):
      model.add(Dense(hn, hn, init='glorot_uniform'))
      model.add(PReLU((hn,)))
      model.add(BatchNormalization((hn,)))
      model.add(Dropout(dp))

    model.add(Dense(hn, output_dim, init='glorot_uniform'))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model
'''

EPOCHS = 350
BATCHES = 512
# 512: 2.235
# 256: 2.25
# 1024: 2.2319
HN = 750
# 512: best till now 
input_dim = features.shape[1]
output_dim = labels_binary.shape[1]

# pdb.set_trace()

print 'keras fit'

from sklearn.cross_validation import StratifiedKFold


model = build_model(input_dim, output_dim, HN)
model.fit(features.as_matrix(), labels_binary.as_matrix(), nb_epoch = EPOCHS, batch_size = BATCHES, verbose = 0)


test = pd.read_csv("test.csv")
test[["X","Y"]] = xy_scaler.transform(test[["X","Y"]])

test["X"] = test["X"].apply(lambda x: 0 if abs(x)>5 else x)
test["Y"] = test["Y"].apply(lambda y: 0 if abs(y)>5 else y)

new_addresses = sorted(test["Address"].unique())
new_A_counts=test.groupby("Address").size()
only_new = set(new_addresses + addresses) - set(addresses)
only_old = set(new_addresses + addresses) - set(new_addresses)
in_both = set(new_addresses).intersection(addresses)
for addr in only_new:
    PA = new_A_counts[addr]/float(len(test) + len(train))
    logoddsPA[addr] = np.log(PA) - np.log(1.-PA)
    logodds[addr] = deepcopy(default_logodds)
    logodds[addr].index = range(len(categories))
for addr in in_both:
    PA = (A_counts[addr] + new_A_counts[addr])/float(len(test)+len(train))
    logoddsPA[addr] = np.log(PA) - np.log(1.-PA)


features_sub, _ = parse_data(test, logodds, logoddsPA)

collist = features_sub.columns.tolist()
print(collist)


features_sub[collist] = scaler.transform(features_sub[collist])

pred = pd.DataFrame(model.predict_proba(features_sub.as_matrix()), columns = sorted(labels.unique()))

import gzip
with gzip.GzipFile('SFCrimePredictionNeuralNetwork.csv.gz',mode = 'w',compresslevel = 9) as gzfile: pred.to_csv(gzfile,index_label = "Id",na_rep = "0")

print("--- %s seconds ---" % (time.time() - start_time))

