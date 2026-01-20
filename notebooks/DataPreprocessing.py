#Load csv
import pandas as pd

df= pd.read_csv("Dataset_ATS_v2.csv")
df.head()
df.info()
df.isnull().sum()

# binary column by Label encoding
from sklearn.preprocessing import LabelEncoder

binary_column= ['gender','Dependents','PhoneService','MultipleLines','Churn']
le= LabelEncoder()
for col in binary_column:
    df[col]= le.fit_transform(df[col])

#for multi-class columns (One-hot encoding)

df= pd.get_dummies(df, columns=['InternetService','Contract'], drop_first=True)
df.info()

df.to_csv("Processed_data.csv", index=False)


#Train/Test Split
from sklearn.model_selection import train_test_split

X=df.drop('Churn', axis=1) # Features
y= df['Churn'] # Target variables
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42, stratify=y)

# Combine features and target for training set
train_df = pd.concat([X_train, y_train], axis=1)

# Combine features and target for test set
test_df = pd.concat([X_test, y_test], axis=1)

#Save train and test file
train_df.to_csv("train_set.csv", index=False)
test_df.to_csv("test_set.csv", index=False)
