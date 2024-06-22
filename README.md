---
layout: post
title: Credit Card Fraud Detection Using ML
image: "/posts/classification-title-img.png"
tags: [Fraud Detection, Machine Learning, Classification, Python]
---

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>
Welcome to the Credit Card Fraud Detection Project! 
Credit card fraud poses a major threat to both financial institutions and consumers. As online transactions become more prevalent, identifying fraudulent activities has become more complex. 

Here, I created a fraud detection system using Python and widely-used machine learning libraries like scikit-learn. The primary goal was to build a robust fraud detection system using Random Forest, Logistic Regression, and Decision Tree classifiers. This project also addresses the class imbalance problem inherent in credit card fraud datasets.

### Project Structure
creditcard.csv: The dataset containing transaction details and fraud labels.
Credit Card Fraud Detection.py: The main script containing data preprocessing, model training, evaluation, and model saving.
credit_card_model.p: The serialized Random Forest model saved using pickle.## Import Libraries.

### Actions <a name="overview-actions"></a>

I first needed to compile the necessary data from tables in the database.

As I was predicting a binary output, I tested three classification modeling approaches, namely:

* Logistic Regression
* Decision Tree
* Random Forest

I imported the data, defined models using a dictionary, trained & tested each model, and then measured this predictive performance based on several metrics to give a well-rounded overview of which was the best.
I also used undersampling and oversampling techniques to address the class imbalance problem inherent in credit card fraud datasets.
<br>
<br>

### Results <a name="overview-results"></a>

Here, I aimed to build a model that would accurately predict the fraud transaction.
Based upon these, the chosen model is the Random Forest as it was the most consistently performant on the test set across classification accuracy, precision, recall, and f1-score. 

<br>
**Metric 1: Classification Accuracy**

* Random Forest = 0.99
* Decision Tree = 0.99
* Logistic Regression = 0.99

<br>
**Metric 2: Precision**

* Random Forest = 0.90
* Logistic Regression = 0.89
* Decision Tree = 0.66

<br>
**Metric 3: Recall**

* Random Forest = 0.76
* Decision Tree = 0.75
* Logistic Regression = 0.6

<br>
**Metric 4: F1 Score**

* Random Forest = 0.82
* Logistic Regression = 0.72
* Decision Tree = 0.7

<br>
<br>
___

**Import Required Packagess.**


```python
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
```

## Get the Data
** Read the 'Creditcard.csv file into a dataframe **


```python
data = pd.read_csv("creditcard.csv")
```

# Modelling Overview  <a name="modelling-overview"></a>

We built a model that looked to accurately predict fraud transaction.

As we were predicting a binary output, we tested three classification modeling approaches, namely:

* Logistic Regression
* Decision Tree
* Random Forest


**Data Preprocessing**


```python
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
data['Amount'] = scale_standard.fit_transform(pd.DataFrame(data['Amount']))         

# Drop Time column
data.drop('Time', axis = 1, inplace = True)

# Find duplicated data
data.duplicated().any()

# Drop duplicated data
data = data.drop_duplicates()
data.shape
```

I also investigated the class balance of the dependent variable - which is important when assessing classification accuracy.


```python
# Class balance
data['Class'].value_counts()   
```

<br>
From the last step in the above code, I saw that **0.2% of the data were in class 1 and the rest of them were in class 0**. This showed me a clear class imbalance. I made sure to not rely on classification accuracy alone when assessing results - also analyzing Precision, Recall, and F1-Score.

<br>

# Create Input and Output Variables


```python
X = data.drop(['Class'], axis = 1)
y = data['Class']
```


# Train Test Split

**Use train_test_split to split your data into a training set and a testing set.**


```python
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.2, random_state = 42)
```

# Model Training and Assessment

Three classifiers were trained and evaluated:

Logistic Regression
Decision Tree Classifier
Random Forest Classifier

Each model's performance was evaluated using the following metrics:

Accuracy
Precision
Recall
F1-Score

```python
classifier = {
       'LogisticRegression': LogisticRegression(random_state = 42),
       'DecisionTreeClassifier': DecisionTreeClassifier(random_state = 42),
       'RandomForestClassifier': RandomForestClassifier(random_state = 42)}

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
```

**Fit this KNN model to the training data.**


```python
clf.fit(X_train, y_train)
```




<style>#sk-container-id-7 {color: black;background-color: white;}#sk-container-id-7 pre{padding: 0;}#sk-container-id-7 div.sk-toggleable {background-color: white;}#sk-container-id-7 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-7 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-7 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-7 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-7 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-7 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-7 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-7 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-7 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-7 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-7 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-7 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-7 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-7 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-7 div.sk-item {position: relative;z-index: 1;}#sk-container-id-7 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-7 div.sk-item::before, #sk-container-id-7 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-7 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-7 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-7 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-7 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-7 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-7 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-7 div.sk-label-container {text-align: center;}#sk-container-id-7 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-7 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-7" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>KNeighborsClassifier(n_neighbors=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" checked><label for="sk-estimator-id-7" class="sk-toggleable__label sk-toggleable__label-arrow">KNeighborsClassifier</label><div class="sk-toggleable__content"><pre>KNeighborsClassifier(n_neighbors=1)</pre></div></div></div></div></div>



# Predictions and Evaluations
Let's evaluate our KNN model!

**Use the predict method to predict values using your KNN model and X_test.**


```python
y_predict = clf.predict (X_test)
```

** Create a confusion matrix and classification report.**


```python
from sklearn.metrics import classification_report,confusion_matrix
```


```python
print(confusion_matrix(y_test, y_predict))
```

    [[67 31]
     [23 79]]
    


```python
print(classification_report(y_test, y_predict))
```

                  precision    recall  f1-score   support
    
               0       0.74      0.68      0.71        98
               1       0.72      0.77      0.75       102
    
        accuracy                           0.73       200
       macro avg       0.73      0.73      0.73       200
    weighted avg       0.73      0.73      0.73       200
    
    

# Choosing a K Value
Let's go ahead and use the elbow method to pick a good K Value!

** Create a for loop that trains various KNN models with different k values, then keep track of the error_rate for each of these models with a list. Refer to the lecture if you are confused on this step.**


```python
from sklearn.metrics import f1_score

accuracy_scores = []
k_list = list(range(1,40))

for i in k_list:
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(X_train, y_train)
    y_predict_i = clf.predict(X_test)
    accuracy = f1_score(y_test, y_predict_i)
    accuracy_scores.append(accuracy)
```

**Now create the following plot using the information from your for loop.**


```python
plt.figure(figsize=(10,6))
plt.plot(range(1,40),accuracy_scores,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Accuracy Score vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy Score')
```




    Text(0, 0.5, 'Accuracy Score')




    
![png](output_40_1.png)
    



```python
max_accuracy = max(accuracy_scores)
max_accuracy_idx = accuracy_scores.index(max_accuracy)
optimal_k_value = k_list[max_accuracy_idx]
```


```python
plt.plot(k_list, accuracy_scores)
plt.scatter(optimal_k_value, max_accuracy, marker = 'x', color = 'red')
plt.title(f'Accuracy (F1 Score) by K \n Optimal value for K: {optimal_k_value} (Accuracy: {round(max_accuracy, 4)})')
plt.xlabel('k')
plt.ylabel('Accuracy (F1 Score)')
plt.tight_layout()
plt.show()
```


    
![png](output_42_0.png)
    


## Retrain with new K Value

**Retrain your model with the best K value (up to you to decide what you want) and re-do the classification report and the confusion matrix.**


```python
clf = KNeighborsClassifier(n_neighbors=11)

clf.fit(X_train,y_train)
y_predict = clf.predict(X_test)

print('WITH K=11')
print('\n')
print(confusion_matrix(y_test,y_predict))
print('\n')
print(classification_report(y_test,y_predict))
```

    WITH K=11
    
    
    [[77 21]
     [24 78]]
    
    
                  precision    recall  f1-score   support
    
               0       0.76      0.79      0.77        98
               1       0.79      0.76      0.78       102
    
        accuracy                           0.78       200
       macro avg       0.78      0.78      0.77       200
    weighted avg       0.78      0.78      0.78       200
    
    

# Great Job!
