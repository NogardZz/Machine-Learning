"""
  ENTROPY
"""
# Lab_1
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
dt = pd.read_csv("winequality-white.csv", delimiter=';')
# print(dt)
print("So phan tu: ", len(dt)) # Có 4898 phần tử
print("So nhan: ", dt.quality.unique()) # Có 7 nhãn
from sklearn.model_selection import train_test_split
X = dt.drop("quality", 1)
y = dt["quality"]
# print(X)
# print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
print("So phan tu trong tap kiem tra: ", len(y_test))
print("So phan tu trong tap hoc: ", len(y_train))
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion='entropy', random_state=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy is: ", accuracy_score(y_test, y_pred)*100)
print(confusion_matrix(y_test, y_pred))
X_test_small = X_test[:10]
y_test_small = y_test[:10]
y_pred_small = clf.predict(X_test_small)
print("Accuracy is: ", accuracy_score(y_test_small, y_pred_small)*100)
print(confusion_matrix(y_test_small, y_pred_small))

# Lab_2
X2 = [[180, 15, 0],
      [167, 42, 1],
      [136, 35, 1],
      [174, 25, 0],
      [141, 20, 1]]
Y2 = ['Nam', 'Nu', 'Nu', 'Nam', 'Nu']
clf2 = DecisionTreeClassifier(criterion='entropy', random_state=5)
clf2.fit(X2, Y2)
print("Dự báo phần tử mới tới có thông tin chiều cao=130, độ dài mái tóc = 38 và giọng nói có giá trị là 1 thì người này là nam hay nữ? ", clf2.predict([[130, 38, 1]]))
