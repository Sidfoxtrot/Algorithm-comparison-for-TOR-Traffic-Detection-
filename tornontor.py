#Import the required libraries
import pandas as pd
import numpy as np
from sklearn import tree
import sklearn.ensemble as ske
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn import model_selection
import warnings
warnings.filterwarnings("ignore")

data='TOR-NonTOR.csv'
dataframe=pd.read_csv(data,low_memory=False)
#Data Cleansing
def dfnormalize(df):
    for feature in df.columns:
         df.loc[:,feature]=pd.to_numeric(df.loc[:,feature],errors='coerce').fillna(0)
         maxvalue=df[feature].max()
         minvalue=df[feature].min()
         if (maxvalue-minvalue)>0:
             df.loc[:,feature]=(df.loc[:,feature]-minvalue)/(maxvalue-minvalue)
         else:
             df.loc[:,feature]=(df.loc[:,feature]-minvalue)    
    return df
columns=dataframe.keys()
print(columns)
datatobeused=dataframe[columns[4:len(columns)-1]].copy()
normalise=dfnormalize(datatobeused)
print(normalise.describe())
change_labels = lambda x: 1 if x == 'nonTOR' else 0
x_train = normalise.sample(frac=0.8,replace=True)
x_test = normalise.drop(x_train.index)
y_train=dataframe['label'].apply(change_labels).loc[x_train.index]
y_test=dataframe['label'].apply(change_labels).loc[x_test.index]

#Training and Comparing the models
outcome=[]
model_names=[]
models = [('DecTree',DecisionTreeClassifier()),('KNN',KNeighborsClassifier()),('RandomForest',ske.RandomForestClassifier())]
for model_name, model in models:
    k_fold_validation=model_selection.KFold(n_splits=10, random_state=12)
    results=model_selection.cross_val_score(model,x_train,y_train,cv=k_fold_validation,scoring='accuracy')
    outcome.append(results)
    model_names.append(model_name)
    output="%s|Accuracy=%f "%(model_name,results.mean()*100)
    print(output)


fig=plt.figure()
fig.suptitle('Machine Learning Model Comparison')
ax = fig.add_subplot(111)
plt.boxplot(outcome)
ax.set_xticklabels(model_names)
plt.show()







