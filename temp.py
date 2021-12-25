# import biblio

import numpy as np
import pandas as pd
import random as rnd

# visualization biblio
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
# machine learning biblio
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
# import data and delet of usefull data 
df_train=pd.read_csv('/home/lotfi/Downloads/train (1).csv')

df_train=df_train.drop(['Name','Ticket','Cabin','PassengerId'],axis=1)

df_test=pd.read_csv('/home/lotfi/Downloads/test.csv')
df_test=df_test.drop(['Name','Ticket','Cabin','PassengerId'],axis=1)
y_class=pd.read_csv('/home/lotfi/Downloads/gender_submission.csv')

#handling missing values age and fare

df_train['Age']=df_train['Age'].fillna(df_train['Age'].mean())
df_train['Fare']=df_train['Fare'].fillna(df_train['Fare'].mean())
df_train=df_train.dropna()
df_test['Age']=df_test['Age'].fillna(df_test['Age'].mean())
df_test['Fare']=df_test['Fare'].fillna(df_test['Fare'].mean())
# 
df_train['Sex'].replace(['male', 'female'],[0,1],inplace=True)
df_test['Sex'].replace(['male', 'female'],[0,1],inplace=True)

# 
ports = {"S": 0, "C": 1, "Q": 2}
data = [df_train, df_test]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)
    
    
#    
data = [df_train, df_test]

for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)

#  
data = [df_train, df_test]
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6

#declaration
X_train = df_train.drop("Survived", axis=1)
Y_train = df_train["Survived"]
X_test  = df_test

y_test=y_class['Survived'].values

# KNN 
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train) 
knn_pred = knn.predict(X_test) 


#print(Y_pred)




#arbre de décision

decision_tree = DecisionTreeClassifier() 
decision_tree.fit(X_train, Y_train)  
Y_pred = decision_tree.predict(X_test)  


 #SVM

linear_svc = SVC()
linear_svc.fit(X_train, Y_train)
svm_pred = linear_svc.predict(X_test)

#quel est le meilleur Modele


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions = cross_val_predict(decision_tree, X_train, Y_train, cv=3)
confusion_matrix(Y_train, predictions)
#calcul de précision
from sklearn.metrics import precision_score

knn_prec=precision_score(y_test, knn_pred)
y_prec=precision_score(y_test, Y_pred)
svm_prec=precision_score(y_test, svm_pred)


results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Decision Tree'],
    'Score': [knn_prec, y_prec, svm_prec]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head()

print(svm_prec)
print(y_prec)
print (knn_prec)
