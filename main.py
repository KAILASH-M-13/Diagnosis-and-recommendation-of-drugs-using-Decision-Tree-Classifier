import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
df=pd.read_csv("/content/drug200-1.csv")
df['Sex']=df['Sex'].map({'M':1,'F':0})
print(df)
df['BP']=df['BP'].map({'HIGH':1,'LOW':0,'NORMAL':3})
print(df)
df['Cholesterol']=df['Cholesterol'].map({'HIGH':1,'LOW':0,'NORMAL':3})
print(df)
df['Drug']=df['Drug'].map({'DrugA' : 1 ,'drugA' : 1 ,'DrugB' : 2 ,'drugB' : 2 ,'DrugC' : 3 ,'drugC' : 3,'DrugX' : 4 ,'drugX' : 4,'DrugY' : 5 ,'drugY' : 5 })
print(df)
from sklearn.model_selection import train_test_split
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state=0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(X_test)
print(y_pred)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
ag=int(input("Enter your age"))
sx=input("Enter your gender, Male as M and Female as F :")
if sx in ("M","male","m"):
  a=1
elif sx=="F":
    a=0
else:
    print("Not defined")
bp=input("Enter your blood pressure level HIGH,LOW,NORMAL :").upper()
if bp=="HIGH":
  b=1
elif bp=="LOW":
    b=0
elif bp=="NORMAL":
    b=3
else:
    print("Not defined")
Cho=input("Enter your Cholesterol level HIGH,LOW,NORMAL :")
if Cho=="HIGH":
  bo=1
elif Cho=="LOW":
    bo=0
elif Cho=="NORMAL":
    bo=3
else:
    print("Not defined")
so=input("Enter the sodium level at your blood :")
data = {
  "Age": [ag],
  "Sex": [a],
  "BP": [b],
  "Cholesterol": [bo],
  "Na_to_K": [so]
}
pre=pd.DataFrame(data)
y_pred_1 = classifier.predict(pre)

if y_pred_1 == 1:
  print("Drug A")
elif y_pred_1 == 2:
  print("Drug B")
elif y_pred_1 == 3:
  print("Drug C")
if y_pred_1 == 4:
  print("Drug X")
if y_pred_1 == 5:
  print("Drug Y")
