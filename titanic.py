import numpy as np
import pandas as pd
from sklearn import tree
from IPython.display import Image
from sklearn.externals.six import StringIO
import pydotplus
from sklearn.ensemble import RandomForestClassifier

input_file = "~/udemy/playground/titanic.csv"
png_filename='/Users/ddarios/udemy/playground/titanic_dt.png'
df = pd.read_csv(input_file, header = 0)
np.random.seed(10)
print(df.head())

d = {'male': 1, 'female': 0}
df['Sex'] = df['Sex'].map(d)
print(df.head())

df = df.drop(columns=['Name']) # what does axis do?
print(df.head())

features = list(df.columns[:6])
# print(features)

y = df["Survived"]
X = df[features]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=features)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
data = graph.create_png()
# with open(png_filename, "wb") as png:
#         writer = png.write(data)

clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, y)

#Predict survived of an passenger that is 50 years old
print (clf.predict([[2, 0, 65, 0, 0, 10.00]]))
#...and a passenger that is 10 years old
print (clf.predict([[2, 0, 1, 0, 0, 10.00]]))
