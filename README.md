# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the employee dataset and display basic information, including null values and class distribution of the left column.

2.Encode the categorical salary column using Label Encoding.

3.Define the features (X) and target (y) by selecting relevant columns.

4.Split the data into training and testing sets (80-20 split).

5.Initialize a Decision Tree Classifier with the entropy criterion and train it on the training data.

6.Predict the target values for the test set.

7.Calculate and display the model's accuracy.

8.Compute and display the confusion matrix for the predictions.

9.Predict the left status for a new employee sample.

## Program:
```python
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: M.Chandru
RegisterNumber:  24900224
```
```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
data=pd.read_csv('Employee.csv')
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
print(data.head())
x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours',
'time_spend_company','Work_accident','left','promotion_last_5years']]
print(x.head())
y=data[['left']]
print(y.head())
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print(accuracy)
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

![Screenshot 2024-11-28 170610](https://github.com/user-attachments/assets/79a7ee51-74a3-4d12-8af1-c65db7d84cae)

![Screenshot 2024-11-28 170631](https://github.com/user-attachments/assets/326737f0-453f-46b1-a476-5eb54f71f11d)

![Screenshot 2024-11-28 170638](https://github.com/user-attachments/assets/a3379a45-9257-4b37-b861-4dffc53cbd49)

![Screenshot 2024-11-28 170647](https://github.com/user-attachments/assets/e919622d-4915-4922-a25e-7fa2430287bc)

![Screenshot 2024-11-28 171213](https://github.com/user-attachments/assets/38e22183-44ff-4f09-b677-223bd994c3c1)

![Screenshot 2024-11-28 170835](https://github.com/user-attachments/assets/28eaf883-f1e7-43bc-a415-3116ec4b9b71)

![Screenshot 2024-11-28 170715](https://github.com/user-attachments/assets/d2bbbe51-60fc-4349-89bd-d1defecea96e)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
