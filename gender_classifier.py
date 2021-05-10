from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier 

dt = tree.DecisionTreeClassifier()
kn = KNeighborsClassifier()
mlp = MLPClassifier()
# [height, weight, shoe_size]

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
    [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42], 
    [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
    'female', 'male', 'male']

dt_clf = dt.fit(X,Y)
kn_clf = kn.fit(X,Y)
mlp_clf = mlp.fit(X,Y)

kn_prediction = kn_clf.predict([[190,70,43]])
mlp_prediction = mlp_clf.predict([[190,70,43]])
dt_prediction = dt_clf.predict([[190,70,43]])

print("predicted type for decision tree classifier: ", dt_prediction)
print("predicted type for K-Neighbor classifier: ", kn_prediction)
print("predicted type for MLP classifier: ", mlp_prediction)