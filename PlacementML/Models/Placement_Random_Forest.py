# Importing requered modules

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
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

pipeline = Pipeline([('encoder',Preprocessor),('model',RandomForestClassifier(n_estimators=300,max_depth=20))])

# Fitting the model

pipeline.fit(x_train,y_train)

# Saving and Loading the model

joblib.dump(pipeline,"Models/Placement_Random_Forest.pkl")
loaded_pipeline = joblib.load("Models/Placement_Random_Forest.pkl")

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

print("===============FEATURE_IMPORTANCE===============")

Features = loaded_pipeline.named_steps['encoder'].get_feature_names_out()
Importance = loaded_pipeline.named_steps['model'].feature_importances_

df_Importance = pd.DataFrame({'feature': Features,
                              'Importance':Importance})

df_Importance = df_Importance.sort_values(by='Importance',ascending=False)

plt.figure(figsize=(10,8))

sns.barplot(
    data=df_Importance.head(15),
    x="Importance",
    y="feature"
)


plt.savefig("Outputs/Graphs/Feature_Importance.png")
plt.close()

print("===============END===============")