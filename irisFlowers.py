import numpy
import matplotlib.pyplot
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Load DataSet from SKlearn for iris flowers
iris = datasets.load_iris()
data = iris.data
target = iris.target


#Print some of the iris Dataset from SKlearn
#print(iris.feature_names)
#print(data[:5])
#print(target[:5])


#Splitiung our Dataset into training and testing sets. using sklearn module
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2, random_state=42)


#Setting a mean of 0 and a deviation of 1 of our data from iris dataset using sklearn module
scaler = StandardScaler()
data_train = scaler.fit_transform(data_train)
data_test = scaler.transform(data_test)

#using the K-Nearest Neighbors algorithn to train our model, this is also from sklearn
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(data_train, target_train)


#make a prediction based on the trained model
target_predictions = knn.predict(data_test)

#Evaluate the models performance
accuracy = accuracy_score(target_test, target_predictions)
print(f'Accuracy: {accuracy:.2f}')
print('\nConfusion Matrix:')
print(confusion_matrix(target_test, target_predictions))
print('\nClassification Report:')
print(classification_report(target_test, target_predictions))
