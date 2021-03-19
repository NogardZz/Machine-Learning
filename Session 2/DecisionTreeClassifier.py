from sklearn.datasets import load_iris
iris_dt = load_iris()
iris_dt.data[1:5] #thuoc_tinh
iris_dt.target[1:5] #nhan
# print(iris_dt.data[1:5])
# print(iris_dt.target[1:5])
from sklearn.model_selection import train_test_split
X_train, X_text, y_train, y_test = train_test_split(iris_dt.data,
                                                    iris_dt.target,
                                                    test_size=1/3.0,
                                                    random_state=5)
"""
X_train, y_train: tap thuoc tinh
x_test, y_test: tap kiem tra
"""

X_train[1:6]
X_train[1:6, 1:3]
y_train[1:6]
X_text[6:10]
y_test[6:10]

from sklearn.tree import DecisionTreeClassifier
# max_depth = 3 do sau cua cay bang 3
# min_samples_leaf dung qua trinh phan hoach
clf_gini = DecisionTreeClassifier(criterion="gini",
                                  random_state=100,
                                  max_depth=3,
                                  min_samples_split=5)
clf_gini.fit(X_train, y_train) #Huan luyen mo hinh

y_pred = clf_gini.predict((X_text))
y_test
clf_gini.predict([[4, 4, 3, 3]])

from sklearn.metrics import  accuracy_score
print("Accuracy is: ", accuracy_score(y_test, y_pred)*100)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred, labels=[2,0,1])
