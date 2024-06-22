###############################################################################
# Random Forest for Classification - Credit Card Fraud Detection
###############################################################################


###############################################################################
# Import Required Packages
###############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

###############################################################################
# Import Data
###############################################################################

# Import
data = pd.read_csv("creditcard.csv")

###############################################################################
# Data Preprocessing
###############################################################################

# Check data head
data.head()

# Let's see all the columns (instead of seeing ... for a couple of them)
pd.options.display.max_columns = None
data.head()

# Check data tail
data.tail()

# Check data shape
data.shape
print(f'Number of rows: {data.shape[0]}')
print(f'Number of columns: {data.shape[1]}')

# Check data type and missing values
data.info()
data.isna().sum()

# Standardisation

scale_standard = StandardScaler()

# Convert the Result from Array to DataFrame
data['Amount'] = scale_standard.fit_transform(pd.DataFrame(data['Amount']))         # or data['Amount'] = scale_standard.fit_transform(data[['Amount']])

# Drop Time column
data.drop('Time', axis = 1, inplace = True)

# Find duplicated data
data.duplicated().any()

# Drop duplicated data
data = data.drop_duplicates()
data.shape

# Class balance
data['Class'].value_counts()                                   # Checking if the data are balanced

###############################################################################
# Data Visualization
###############################################################################

sns.countplot(data['Class'])
plt.show()

###############################################################################
# Split Input Variables and Output Variables
###############################################################################

X = data.drop(['Class'], axis = 1)
y = data['Class']

###############################################################################
# Split out Training and Test Sets
###############################################################################

X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.2, random_state = 42)

###############################################################################
# Model Training and Assessment
###############################################################################

classifier = {
       'LogisticRegression': LogisticRegression(random_state = 42),
       'DecisionTreeClassifier': DecisionTreeClassifier(random_state = 42),
       'RandomForestClassifier': RandomForestClassifier(random_state = 42)}              # n_estimators: number of trees
                                                                                         # max_features: number of variables offered for splitting in each splitting point
                                                                                         # If we do not enter these parameters, the defaults will be applied (100, auto)

for name, clf in classifier.items():
    
    print(f'\n==========={name}==========')
    
    clf.fit(X_train, y_train)
    y_pred_class = clf.predict(X_test)

    # Accuracy (the number of correct classification out of all attempted classifications)
    accuracy = accuracy_score(y_test, y_pred_class) 
    print(f'\n Accuracy: {accuracy}')
    
    # Precision (of all observations that were predicted as positive, how many were actually positive)
    precision = precision_score(y_test, y_pred_class)
    print(f'\n Precision: {precision}')

    # Recall (of all positive observations, how many did we predict as positive)
    recall = recall_score(y_test, y_pred_class) 
    print(f'\n Recall: {recall}')

    # F1-Score (the harmonic mean of precision and recall)
    f1 = f1_score(y_test, y_pred_class)
    print(f'\n F1_score: {f1}')

###############################################################################
# Dealing with Class Imbalance
###############################################################################

normal = data[data['Class']==0]
fraud = data[data['Class']==1]

normal.shape
fraud.shape

# Undersampling: will result in data lost
###############################################################################

normal.sample = normal.sample(n=473)

    # Create new undersampled dataset

new_data  = pd.concat([normal.sample, fraud], ignore_index=True)

new_data.head()

new_data['Class'].value_counts()

X = new_data.drop(['Class'], axis = 1)
y = new_data['Class']

X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.2, random_state = 42)

classifier = {
       'LogisticRegression': LogisticRegression(random_state = 42),
       'DecisionTreeClassifier': DecisionTreeClassifier(random_state = 42),
       'RandomForestClassifier': RandomForestClassifier(random_state = 42)}              # n_estimators: number of trees
                                                                                         # max_features: number of variables offered for splitting in each splitting point
                                                                                         # If we do not enter these parameters, the defaults will be applied (100, auto)

for name, clf in classifier.items():
    
    print(f'\n==========={name}==========')
    
    clf.fit(X_train, y_train)
    y_pred_class = clf.predict(X_test)

    # Accuracy (the number of correct classification out of all attempted classifications)
    accuracy = accuracy_score(y_test, y_pred_class) 
    print(f'\n Accuracy: {accuracy}')
    
    # Precision (of all observations that were predicted as positive, how many were actually positive)
    precision = precision_score(y_test, y_pred_class)
    print(f'\n Precision: {precision}')

    # Recall (of all positive observations, how many did we predict as positive)
    recall = recall_score(y_test, y_pred_class) 
    print(f'\n Recall: {recall}')

    # F1-Score (the harmonic mean of precision and recall)
    f1 = f1_score(y_test, y_pred_class)
    print(f'\n F1_score: {f1}')

# Oversampling: new sample data will be produced and added to the original data set to reach the size of majority
###############################################################################

X = data.drop(['Class'], axis = 1)
y = data['Class']

X.shape
y.shape

X_res, y_res = SMOTE().fit_resample(X,y)

y_res.value_counts()

X_train, X_test, y_train, y_test = train_test_split (X_res, y_res, test_size = 0.2, random_state = 42)

classifier = {
       'LogisticRegression': LogisticRegression(random_state = 42),
       'DecisionTreeClassifier': DecisionTreeClassifier(random_state = 42),
       'RandomForestClassifier': RandomForestClassifier(random_state = 42)}              # n_estimators: number of trees
                                                                                         # max_features: number of variables offered for splitting in each splitting point
                                                                                         # If we do not enter these parameters, the defaults will be applied (100, auto)

for name, clf in classifier.items():
    
    print(f'\n==========={name}==========')
    
    clf.fit(X_train, y_train)
    y_pred_class = clf.predict(X_test)

    # Accuracy (the number of correct classification out of all attempted classifications)
    accuracy = accuracy_score(y_test, y_pred_class) 
    print(f'\n Accuracy: {accuracy}')
    
    # Precision (of all observations that were predicted as positive, how many were actually positive)
    precision = precision_score(y_test, y_pred_class)
    print(f'\n Precision: {precision}')

    # Recall (of all positive observations, how many did we predict as positive)
    recall = recall_score(y_test, y_pred_class) 
    print(f'\n Recall: {recall}')

    # F1-Score (the harmonic mean of precision and recall)
    f1 = f1_score(y_test, y_pred_class)
    print(f'\n F1_score: {f1}')

###############################################################################
# Saving our Model
###############################################################################

rfc = RandomForestClassifier()
rfc.fit(X_res, y_res)

pickle.dump(rfc, open('Credit Card Fraud Detection/credit_card_model.p', 'wb'))                 # .p:  pickle
                                                                                                # open('address')
                                                                                                # 'wb': write the file
                                                                                        
###############################################################################
# Prediction
###############################################################################                                                                                        

model = pickle.load(open('Credit Card Fraud Detection/credit_card_model.p', 'rb'))         
                 
prediction = model.predict([[-1.3598071336738,-0.0727811733098497,2.53634673796914,1.37815522427443,-0.338320769942518,0.462387777762292,0.239598554061257,0.0986979012610507,0.363786969611213,0.0907941719789316,-0.551599533260813,-0.617800855762348,-0.991389847235408,-0.311169353699879,1.46817697209427,-0.470400525259478,0.207971241929242,0.0257905801985591,0.403992960255733,0.251412098239705,-0.018306777944153,0.277837575558899,-0.110473910188767,0.0669280749146731,0.128539358273528,-0.189114843888824,0.133558376740387,-0.0210530534538215,149.62]])                                                                     
prediction[0]     

if prediction[0]==0:
    print ('Normal Transaction')
else:
    print ('Fraud Transaction')

                                                                                   