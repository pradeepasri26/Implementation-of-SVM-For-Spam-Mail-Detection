# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary packages.
2.Read the given csv file and display the few contents of the data.
3.Assign the features for x and y respectively.
4.Split the x and y sets into train and test sets.
5.Convert the Alphabetical data to numeric using CountVectorizer.
6.Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
7.Find the accuracy of the model.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: PRADEEPASRI S
RegisterNumber: 212221220038 
*/
print("Result Output:")
import chardet 
file='/content/spam.csv'
with open(file, 'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding='Windows-1252')

print("data head:")
data.head()

print("data info:")
data.info()

print("data isnull:")
data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

print("y_prediction  value:")
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
y_pred

print("Accuracy Value:")
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
![image](https://github.com/pradeepasri26/Implementation-of-SVM-For-Spam-Mail-Detection/assets/131433142/732f5171-f63e-4177-8c00-70155ea5cd7e)
![Screenshot (194)](https://github.com/pradeepasri26/Implementation-of-SVM-For-Spam-Mail-Detection/assets/131433142/cd7f7e85-ac80-4cf7-b5e9-3f80563ef5f7)
![Screenshot (195)](https://github.com/pradeepasri26/Implementation-of-SVM-For-Spam-Mail-Detection/assets/131433142/9802123a-ffcb-4f4a-bb66-499fa418e70e)
![Screenshot (196)](https://github.com/pradeepasri26/Implementation-of-SVM-For-Spam-Mail-Detection/assets/131433142/745b59c7-2b96-4a6b-8807-3d45adc9db86)
![Screenshot (197)](https://github.com/pradeepasri26/Implementation-of-SVM-For-Spam-Mail-Detection/assets/131433142/76be027a-5ad3-4a82-b715-4c8e6454287b)
![Screenshot (198)](https://github.com/pradeepasri26/Implementation-of-SVM-For-Spam-Mail-Detection/assets/131433142/6d8de98b-52c2-46d1-a45e-21b84f0dba96)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
