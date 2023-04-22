import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import metrics
import graphviz

# Load the CSV file into a pandas dataframe
dataset = pd.read_csv("suv_data.csv")
gender=pd.get_dummies(dataset["Gender"],drop_first=1)
dataset=pd.concat([dataset ,gender] , axis=1 )
dataset.drop('Gender', axis=1 , inplace=True)

# Separate the target variable from the features
#independent variable
X = dataset.iloc[: , [1,2,4]].values
X=pd.DataFrame(X)
#dependent variables
y = dataset.iloc[: ,3 ].values
y=pd.DataFrame(y)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create a decision tree classifier
dt = DecisionTreeClassifier()

# Train the classifier on the training data
dt.fit(X_train, y_train)

# Predict the target variable on the testing data
y_pred = dt.predict(X_test)

# Evaluate the performance of the classifier
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Visualize the decision tree
dot_data = export_graphviz(dt, out_file=None, feature_names=['Age', 'EstimatedSalary', 'Male'], class_names=['0', '1'], filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("decision_tree") # Save the visualization to a file
