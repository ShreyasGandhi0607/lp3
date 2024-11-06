import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("diabetes.csv")

print(data.head())

#split the dataset into features(x) and target(y)
X = data.drop('Outcome', axis = 1)
Y = data['Outcome']

#split the dataset into training and testing set

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

#initialize KNN with k = 5
k = 5
knn = KNeighborsClassifier(n_neighbors = k)

#fit the model

knn.fit(x_train, y_train)

#make predictions

y_pred = knn.predict(x_test)

#compute the evaluation metrics

confusion_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
error_rate = 1 - accuracy

#print the evaluation metrics

print(f"\nconfusion matrix for k = {k}\n", confusion_matrix)
print(f"\naccuracy = {accuracy * 100:.2f}%")
print(f"\nprecision = {precision * 100:.2f}%")
print(f"\nrecall = {recall * 100:.2f}%")
print(f"\nerror_rate = {error_rate * 100:.2f}%")

#checking the accuracy for the different values of k
print("\nAccuracy for the different values of K:")

for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for k = {k} : {accuracy * 100:.2f}%")


#function to calculate the manual euclidean distance

def euclidean_distance(row1, row2):
    return np.sqrt(np.sum((row1 - row2)**2))

sample1 = x_train.iloc[0]
sample2 = x_train.iloc[1]
distance = euclidean_distance(sample1, sample2)
print(f"manual euclidean distance is: {distance:.4f}")
