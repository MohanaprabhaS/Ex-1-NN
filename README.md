<H3>NAME: MOHANAPRABHA S</H3>
<H3>REGISTER NO: 212224040197</H3>
<H3>EX. NO.1</H3>
<H3>DATE: 28.01.2026</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df= pd.read_csv("Churn_Modelling.csv")
print(df)

X=df.iloc[:,:-1].values
print(X)

y=df.iloc[:,-1].values
print(y)

print(df.isnull().sum())
df.fillna(df.select_dtypes(include='number').mean(), inplace=True)

print(df.isnull().sum())
y=df.iloc[:,-1].values
print(y)

df.duplicated()
print(df['EstimatedSalary'].describe())

scaler=MinMaxScaler()
df1 = pd.DataFrame(
    scaler.fit_transform(df.select_dtypes(include='number')),
    columns=df.select_dtypes(include='number').columns
)
print(df1)

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

print(X_train)
print(len(X_train))
print(X_test)
print("Lenght of X_test ",len(X_test))
```



## OUTPUT:
## DATASET:

<img width="887" height="418" alt="Screenshot 2026-01-28 160550" src="https://github.com/user-attachments/assets/e31292fe-3de7-41f8-b154-413554de14cb" />

<img width="510" height="290" alt="Screenshot 2026-01-28 204827" src="https://github.com/user-attachments/assets/201a6b53-07ac-4c56-aace-306e57dcfc4e" />

## X VALUES:

<img width="894" height="251" alt="Screenshot 2026-01-28 160648" src="https://github.com/user-attachments/assets/901e346a-8191-44eb-98df-34d46e2efcd8" />

## Y VALUES:

<img width="862" height="107" alt="Screenshot 2026-01-28 160712" src="https://github.com/user-attachments/assets/1781e3a0-ea56-4a63-9e93-a8fb8fadfbc5" />

## NULL VALUES:

<img width="754" height="691" alt="Screenshot 2026-01-28 160757" src="https://github.com/user-attachments/assets/f39d4d5a-9a17-4631-94e3-1e9568242d76" />

## DUPLICATED VALUES:

<img width="761" height="257" alt="Screenshot 2026-01-28 160840" src="https://github.com/user-attachments/assets/410bc3e7-8033-43f4-a321-8a61e276a4d4" />

## DESCRIPTION:

<img width="757" height="208" alt="Screenshot 2026-01-28 160904" src="https://github.com/user-attachments/assets/76fd319e-85ed-42b3-ab50-caa1bb3bde89" />

## TRAINING DATA:

<img width="788" height="608" alt="Screenshot 2026-01-28 160934" src="https://github.com/user-attachments/assets/c6e13c34-2790-494d-b1c3-4e78f638660b" />


## TESTING DATA:

<img width="787" height="420" alt="Screenshot 2026-01-28 161011" src="https://github.com/user-attachments/assets/1b43dee0-ff5b-4fb9-bc0b-57db1edd84e9" />



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


