# Importing Modules

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading and overview of dataset

print("===============OVERVIEW===============")

df = pd.read_csv("Dataset/Placement.csv")

print("Viewing the dataset : \n\n",df.head(10))
print()

print("Inormation about the datset : \n")
df.info()
print()

print("Basic Statistics of data : \n\n",df.describe())
print()

print("NaN value counts : \n\n",df.isna().sum())
print()

print("Duplicate Counts : ",df.duplicated().sum())
print()

print("Placement Count :\n")
print(df["Placement"].value_counts())
print()

# Graphs

print("===============GRAPHS===============")
print()

plt.figure(figsize=(8,5))

sns.histplot(df['IQ'],bins=5)
plt.savefig("Outputs/Graphs/IQ_Hist.png")

plt.close()

print("IQ Histograph...")
print()

plt.figure(figsize=(8,5))

sns.histplot(df['CGPA'],bins=5,color='r')
plt.savefig("Outputs/Graphs/CGPA_Hist.png")

plt.close()

print("CGPA Histograph...")
print()

plt.figure(figsize=(8,5))

num_column = df.select_dtypes(include='number').corr()
sns.heatmap(num_column,annot=True,cmap='Blues')

plt.savefig("Outputs/Graphs/Heatmap.png")
plt.close()

print('Heatmap...')
print()

print("=================END=================")
