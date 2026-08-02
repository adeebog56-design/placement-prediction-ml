# Importing requered modules

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import joblib

# Reading datset and assigning lables and feature

df = pd.read_csv("Dataset/Placement.csv")

y = df['Placement']
X = df.drop(columns=['College_ID','Placement'])

# Assigning training and testing data

x_train,x_test,y_train,y_test = train_test_split(X,y, test_size=0.2 ,random_state=42)

# Making Preprocessor and Pipeline

Preprocessor = ColumnTransformer([ ('cat',OneHotEncoder(),
                                   X.select_dtypes(include='object').columns),
                                   ('num',StandardScaler(),
                                   X.select_dtypes(include="number").columns),])

pipeline = Pipeline([('encoder',Preprocessor),('model',LogisticRegression())])

# Fitting the model

pipeline.fit(x_train,y_train)

# Saving and Loading the model

joblib.dump(pipeline,"Models/Placement_Logestic_Regression.pkl")
loaded_pipeline = joblib.load("Models/Placement_Logestic_Regression.pkl")

# Metrics

print("===============METRICS===============")

y_pred = loaded_pipeline.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)
matrix   = confusion_matrix(y_test,y_pred)
cross_val = cross_val_score(loaded_pipeline ,X,y , cv= 5 )

print("Accuracy_Score : ",round(accuracy,2))
print()

print("Confusion_matrix : \n\n",matrix)
print()

print("Classification Report :\n")
print(classification_report(y_test, y_pred))
print()

print("Cross_Valdation : \n\n",cross_val)
print("Average : ",cross_val.mean(),2)
print()

print("===============END===============")