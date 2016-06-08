#!/bin/env python
# coding: utf-8

# In[1]:

# In[5]:

from copy import deepcopy
import time
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split, KFold, cross_val_predict
from sklearn.base import clone
from sklearn.metrics import log_loss
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.grid_search import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator
import holidays
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from scipy.stats import expon as sp_expon
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics
import xgboost as xgb

# Turns off annoying warnings. If there are ever serious data
# errors, trying removing this line.
pd.options.mode.chained_assignment = None


# In[8]:

class BasicFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, logodds, logoddsPA, date_features=True, DayOfWeek_features=True, PdDistrict_features=True, Address_features=True, Interaction_feature=True, Holiday_features=True):
        """
        Add more features to the dataset. All of are stateless transforms and can be performed outside
        of a CV loop.
        """

        print 'Basic!'

        self.date_features = date_features
        self.DayOfWeek_features = DayOfWeek_features
        self.PdDistrict_features = PdDistrict_features
        self.Address_features = Address_features
        self.Interaction_feature = Interaction_feature
        self.Holiday_features = Holiday_features
        self.logodds = logodds
        self.logoddsPA = logoddsPA

    def fit(self, X, y=None):
        return self

    def transform(self, X_df, y=None):
        # Features from dates
        X = X_df.copy()

        if self.date_features:
            X['Year'] = X.Dates.apply(lambda x: int(x[:4])) # Hypothesis: The distribution of crimes changed over time
            X['Month'] = X.Dates.apply(lambda x: int(x[5:7])) # H: Certain crimes occur during some months more than others
            X['Hour'] = X.Dates.apply(lambda x: int(x[11:13])) # H: Certain crimes occur at day, others at night
            X['Minute'] = X.Dates.apply(lambda x: int(x[14:16])) # H: Certain crimes are rounded to the nearest hour
            # Idea: Is holiday feature. H: Holidays --> Tourists --> Different types of crimes

        # Features from DayOfWeek
        if self.DayOfWeek_features:
            X['DayOfWeekNum'] = X["DayOfWeek"].map({"Tuesday":0, "Wednesday":1,
                                                 "Thursday":2, "Friday":3,
                                                 "Saturday":4, "Sunday":5,
                                                 "Monday":6}) # H: Different days have different crime distributions
            X['IsWeekend'] = X["DayOfWeekNum"].apply(lambda x: 1*((x == 4) | (x == 5))) # H: Weekends are special

        # Features from PdDistrict
        if self.PdDistrict_features:
            X['PdDistrictNum'] = LabelEncoder().fit_transform(X.PdDistrict) # H: Different districts have different crimes

        # Features from Address
        if self.Address_features:
            X['Intersection'] = X.Address.apply(lambda x: 1*("/" in x)) # H: Intersections have unique crimes
            address_features = X["Address"].apply(lambda x: self.logodds[x])
            address_features.columns = ["logodds" + str(x) for x in range(len(address_features.columns))]
            X["logoddsPA"] = X["Address"].apply(lambda x: logoddsPA[x])
            X = X.join(address_features.ix[:,:])
            print 'logodds!'

        if self.Interaction_feature:
            X["Interaction"] = ((X.X-X.X.mean())/X.X.std())*((X.Y-X.Y.mean())/X.Y.std())

        if self.Holiday_features:
            us_holidays = holidays.UnitedStates()
            date = X.Dates.apply(lambda x: str(x[:10])) # yyyy-mm-dd
            holiday = date.apply(lambda x: x in us_holidays) #False/True
            X['Holiday'] = LabelEncoder().fit_transform(holiday) #Recode to numbers
        return X


class DuplicateCrimeCounts(BaseEstimator, TransformerMixin):
    """
    For each crime, count the number of other crimes that occurred at the exact time and location
    """
    def __init__(self):
        print 'Duplicate!'
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = X.groupby(["Dates", "X", "Y"]).size()
        result = result.reset_index(name="crime_count")
        X = X.merge(result, how="left", on=["Dates", "X", "Y"])
        return X


class JustNumerics(BaseEstimator, TransformerMixin):
    """
    Drops all columns that are objects
    """
    def __init__(self):
        print 'JustNumerics!'
        pass

    def fit(self, X, y=None):
        self.numeric_columns = X.dtypes[X.dtypes != "object"].index
        return self

    def transform(self, X, y=None):
        return X[self.numeric_columns]


class PCATransform(BaseEstimator, TransformerMixin):
    """
    PCA with an argument that allows the user to skip the transform
    altogether.
    """
    def __init__(self, n_components=.1, skip=False, whiten=False, standard_scalar=True):
        print 'PCA!'
        self.n_components = n_components
        self.skip = skip
        self.whiten = whiten
        self.standard_scalar = standard_scalar

    def fit(self, X, y=None):
        if not self.skip:
            if self.standard_scalar:
                self.std_scalar = StandardScaler().fit(X)
                X = self.std_scalar.transform(X)
            self.pca = PCA(n_components=self.n_components, whiten=self.whiten).fit(X)
        return self

    def transform(self, X, y=None):
        if not self.skip:
            if self.standard_scalar:
                X = self.std_scalar.transform(X)
            return self.pca.transform(X)
        return X

class ModelBasedFeatures(BaseEstimator, TransformerMixin):
    """
    Adds a feature to the dataset based on the output of a model.

    Should include in FeatureUnion.
    """
    def __init__(self, model, feature_name, skip=False, train_cv=None):
        print 'Model Based Features!'
        self.model = model
        self.feature_name = feature_name
        self.skip = skip
        self.train_cv = train_cv

    def _get_random_item(self, items):
        # Currently only handles scipy distributions
        return items.rvs()

    def fit(self, X, y=None, *args, **kwargs):
        # Purpose of skip is to skip the estimator
        if self.skip:
            return self

        # Hash train data. If test data equals train data,
        # use cv predictions.
        if isinstance(X, pd.DataFrame):
            self.hashed_value = hash(X.values.data.tobytes())
        elif isinstance(X, np.ndarray):
            self.hashed_value = hash(X.data.tobytes())
        else:
            print("Can't hash data")

        # Get specific model param combo for this iteration
        self.model_params = {key: self._get_random_item(kwargs[key]) for key in kwargs}

        # Set params of model to these parameters
        self.model.set_params(**self.model_params)

        # Fit model
        self.model.fit(X, y)

        # Save y values
        self.y = y

        return self

    def transform(self, X):
        # Purpose of skip is to skip the estimator
        if self.skip:
            return X

        # Is the data being transformed the same as the training data
        is_train_data = False
        if isinstance(X, pd.DataFrame) and self.hashed_value == hash(X.values.data.tobytes()):
            is_train_data = True
        if isinstance(X, np.ndarray) and self.hashed_value == hash(X.data.tobytes()):
            is_train_data = True

        # If the dataset is the training data, use CV predictions
        if is_train_data:
            feature = cross_val_predict(clone(self.model), X, self.y)#, cv=self.train_cv)

        # Otherwise, use the model to predict
        else:
            feature = self.model.predict(X)

        # Add feature to dataset
        if isinstance(X, pd.DataFrame):
            X[self.feature_name] = feature
        if isinstance(X, np.ndarray):
            X = np.c_[X, feature]
        return X


def gridsearch_parameter_results(gridsearch):
    """
    Evaluate how each feature impacts the final prediction.
    """
    scores = [val[1] for val in gridsearch.grid_scores_]
    params = gridsearch.param_distributions.keys()
    for param in params:
        arg_values = [val[0][param] for val in gridsearch.grid_scores_]
        no_strings = all([type(arg) != str for arg in arg_values])
        if no_strings:
            plt.scatter(arg_values, scores, linewidth=0, s=100, alpha=.25)
            plt.title(param)
            plt.show()
        else:
            # Make all arg values strings
            arg_values = [str(arg) for arg in arg_values]
            results = pd.Series(data=scores, index=arg_values)
            group_means = results.groupby(level=0).mean()
            group_means.plot(kind="bar", title=param, rot=0, linewidth=0, colormap="viridis")
            plt.show()


def make_predictions(X_predict, clf, splits=20):
    """
    Memory friendly way of making predictions. If you run into memory errors,
    increase the value of splits.
    """

    print 'Make predictions!'

    n = X_predict.shape[0]
    step_size = n//splits + 1
    indices = [e for e in range(0, n + splits + 1, step_size)]
    results = []
    for start, end in [e for e in zip(indices, indices[1:])]:
        results.append(clf.predict_proba(X_predict[start:end]))
    predictions = np.concatenate(results)
    return predictions


def make_Kaggle_file(predict_probabilities, columns, output_file_name="auto", decimal_limit=3):
    """
    Outputs a file that can be submitted to Kaggle. This takes a long time to run, so you
    shouldn't run it that often. Instead, just have good internal validation techniques so you
    don't have to check the public leaderboard.

    Required imports:
    import time
    import pandas as pd

    predict_probabilities: array-like of shape = [n_samples, n_classes]. Is the output of a
        predict_proba method of a sklearn classifier

    columns: array or list of column names that are in the same order as the columns of the
        predict_probabilities method. If LabelEncoder was used, is accessed via the classes_
        attribute. Don't include an "Id" column.

    output_file_name: If "auto" names it sf_crime_test_predictions_<YearMonthDay-HourMinuteSecond>,
        else uses the string entered as the file name.

    decimal_limit: If None uses full precision, else formats predictions based on that precision.
        Can significantly reduce the filesize and make writing the file faster.
        i.e. actual prediction = .2352452435, decimal_limit=2 --> .24, decimal_limit=3 --> .235, etc.
    """

    print 'Kaggle File!'

    predictions = pd.DataFrame(predict_probabilities, columns=columns)
    predictions.index.name = "Id"
    if output_file_name == "auto":
        timestr = time.strftime("%Y%m%d-%H%M%S")
        output_file_name = "SFCrimeEnsemble.csv"
    if decimal_limit:
        decimal_limit = '%%.%df' % decimal_limit
    predictions.to_csv(output_file_name, float_format=decimal_limit)
    print("Finished writing file: ", output_file_name)


# In[9]:

print 'Data Prepare!'

X = pd.read_csv("train.csv")
X = X.sample(frac=1, random_state=57).reset_index(drop=True)

addresses = sorted(X["Address"].unique())
categories = sorted(X["Category"].unique())
C_counts = X.groupby(["Category"]).size()
A_C_counts = X.groupby(["Address","Category"]).size()
A_counts = X.groupby(["Address"]).size()
logodds = {}
logoddsPA = {}
MIN_CAT_COUNTS = 5
default_logodds = np.log(C_counts/len(X)) - np.log(1.0-C_counts/float(len(X)))

for addr in addresses:
    PA = A_counts[addr]/float(len(X))
    logoddsPA[addr] = np.log(PA)-np.log(1.-PA)
    logodds[addr] = deepcopy(default_logodds)
    for cat in A_C_counts[addr].keys():
        if (A_C_counts[addr][cat] > MIN_CAT_COUNTS) and A_C_counts[addr][cat] < A_counts[addr]:
            PA = A_C_counts[addr][cat]/float(A_counts[addr])
            logodds[addr][categories.index(cat)] = np.log(PA) - np.log(1.0-PA)
    logodds[addr] = pd.Series(logodds[addr])
    logodds[addr].index = range(len(categories))


# In[10]:

X = DuplicateCrimeCounts().fit_transform(X)
X = BasicFeatures(logodds, logoddsPA).fit_transform(X)

# Pop off target variable and convert to integer representation
y = X.pop('Category')
labels = LabelEncoder()
y = labels.fit_transform(y)

# Keep only the numeric features
X = JustNumerics().fit_transform(X)


X_predict = pd.read_csv("test.csv")

new_addresses = sorted(X_predict["Address"].unique())
new_A_counts=X_predict.groupby("Address").size()
only_new = set(new_addresses + addresses) - set(addresses)
only_old = set(new_addresses + addresses) - set(new_addresses)
in_both = set(new_addresses).intersection(addresses)
for addr in only_new:
    PA = new_A_counts[addr]/float(len(X_predict) + len(X))
    logoddsPA[addr] = np.log(PA) - np.log(1.-PA)
    logodds[addr] = deepcopy(default_logodds)
    logodds[addr].index = range(len(categories))
for addr in in_both:
    PA = (A_counts[addr] + new_A_counts[addr])/float(len(X_predict)+len(X))
    logoddsPA[addr] = np.log(PA) - np.log(1.-PA)

X_predict = DuplicateCrimeCounts().fit_transform(X_predict)
X_predict = BasicFeatures(logodds, logoddsPA).fit_transform(X_predict)
X_predict = JustNumerics().fit_transform(X_predict)
X_predict.drop(["Id"], axis=1, inplace=True)

# In[ ]:

rnd = 57
param1 = {'max_depth': 6,
         'eta': 0.05,
         # next step: eta to 0.15
         # 'silent': 1,
         'objective': 'multi:softprob',
         'eval_metric': "mlogloss",
         'min_child_weight': 5,
         'subsample': 0.7,
         'colsample_bytree': 0.7,
         'num_class': 39,
         'seed': rnd
        }

param2 = {'max_depth': 7,
         'eta': 0.08,
         # next step: eta to 0.15
         # 'silent': 1,
         'objective': 'multi:softprob',
         'eval_metric': "mlogloss",
         'min_child_weight': 5,
         'subsample': 0.7,
         'colsample_bytree': 0.7,
         'num_class': 39,
         'seed': rnd
        }


model2 = RandomForestClassifier(n_estimators=2500, max_depth=None, min_samples_split=10, min_samples_leaf=10, n_jobs=-1)#, oob_score=True)
# model2 = ExtraTreesClassifier(n_estimators=500, max_depth=None, min_samples_split=10, min_samples_leaf=10, n_jobs=-1)
model1 = GradientBoostingClassifier(learning_rate = 0.1, max_depth = 2, min_samples_leaf = 10, min_samples_split = 10,n_estimators = 10, random_state = rnd, subsample = 0.7, verbose = 0,warm_start = False)
model3 = 'xgboost_1'
model4 = 'xgboost_2'
model5 = LogisticRegression()

clfs = [model3, model4]

final_submit = []

xgboostRounds = 300 
# 300

print 'Ensemble!'

skf = list(StratifiedKFold(y, 5))

for i, (train_idx, test_idx) in enumerate(skf):
    print '------------' + str(i) + '--------------'
    trainX = X.iloc[train_idx]
    cvX = X.iloc[test_idx]
    trainy = y[train_idx]
    cvy = y[test_idx]

    indice = 0
    #run models

    preds = []
    predstest = []

    for model in clfs:
        print indice
        if model == 'xgboost_1':
            dtrain = xgb.DMatrix(trainX, label = trainy)
            dcv = xgb.DMatrix(cvX, label = cvy)
            watchlist = [(dtrain,'train'), (dcv,'eval')]
            clf_1 = xgb.train(param1, dtrain, num_boost_round = xgboostRounds, evals = watchlist, early_stopping_rounds = 5)
            preds.append(clf_1.predict(dcv, ntree_limit = clf_1.best_iteration))
        elif model == 'xgboost_2':
            dtrain = xgb.DMatrix(trainX, label = trainy)
            dcv = xgb.DMatrix(cvX, label = cvy)
            watchlist = [(dtrain,'train'), (dcv,'eval')]
            clf_2 = xgb.train(param2, dtrain, num_boost_round = xgboostRounds, evals = watchlist, early_stopping_rounds = 5)
            preds.append(clf_2.predict(dcv, ntree_limit = clf_2.best_iteration))
        else:
            # pdb.set_trace()
            model.fit(trainX, trainy)
            preds.append(model.predict_proba(cvX))

        text_file = open("Output.txt", "w")
        text_file.write("Done! " + str(indice))
        text_file.close()

        print('model ',indice,': loss=',metrics.log_loss(cvy, preds[indice]))

        if model == 'xgboost_1':
            dtest = xgb.DMatrix(X_predict)
            predstest.append(clf_1.predict(dtest, ntree_limit = clf_1.best_iteration))
        elif model == 'xgboost_2':
            dtest = xgb.DMatrix(X_predict)
            predstest.append(clf_2.predict(dtest, ntree_limit = clf_2.best_iteration))
        else:
            predstest.append(model.predict_proba(X_predict))

        indice += 1

    # find best weights
    step = 0.1 * (1./len(preds))
    print("step: ", step)
    poidsref = np.zeros(len(preds))
    poids = np.zeros(len(preds))
    poidsreftemp = np.zeros(len(preds))
    poidsref = poidsref + 1./len(preds)

    bestpoids = poidsref.copy()
    blend_cv = np.zeros(preds[0].shape)

    for k in range(0, len(preds), 1):
        blend_cv = blend_cv + bestpoids[k] * preds[k]
        bestscore = metrics.log_loss(cvy, blend_cv)

    getting_better_score = True
    while getting_better_score:
        getting_better_score = False
        for i in range(0, len(preds), 1):
            poids = poidsref
            if poids[i] - step>-step:
                # decrease weight in position i
                poids[i] -= step
                for j in range(0, len(preds), 1):
                    if j != i:
                        if poids[j] + step <= 1:
                            # try an increase in position j
                            poids[j] += step
                            # score new weights
                            blend_cv = np.zeros(preds[0].shape)
                            for k in range(0, len(preds), 1):
                                blend_cv = blend_cv + poids[k] * preds[k]
                            actualscore = metrics.log_loss(cvy, blend_cv)
                            # if better, keep it
                            if actualscore < bestscore:
                                bestscore = actualscore
                                bestpoids = poids.copy()
                                getting_better_score = True
                            poids[j] -= step
                poids[i] += step
        poidsref = bestpoids.copy()

    print("weights: ", bestpoids)
    print("optimal blend loss: ", bestscore)

    blend_to_submit = np.zeros(predstest[0].shape)
    for i in range(0, len(preds), 1):
        blend_to_submit = blend_to_submit + bestpoids[i] * predstest[i]

    final_submit.append(blend_to_submit)

final_submit = np.mean(final_submit, axis = 0)

# print(log_loss(y, clf.oob_decision_function_))


# In[ ]:




# In[ ]:

# final_predictions = make_predictions(X_predict, clf)


# In[ ]:

# Export predictions to file to be submitted to Kaggle
make_Kaggle_file(final_submit, labels.classes_, decimal_limit=10)



