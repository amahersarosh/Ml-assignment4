#!/usr/bin/env python
# coding: utf-8

# In[26]:



path_to_csv = "C:/Users/MaherSarosh/Downloads/Dataset (1)/Dataset"
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
# Read the provided CSV file ‘data.csv’
df = pd.read_csv(path_to_csv)
df
# Show the basic statistical description about the data.
df.describe()


# In[27]:


# Check if the data has null values.
print('Are there any null values: ',df.isnull().values.any())
# Replace the null values with the mean
df.fillna(df.mean(),inplace=True)
print('Are there any null values after using fillna: ',df.isnull().values.any())
# Select at least two columns and aggregate the data using: min, max, count, mean.
aggre = df.groupby('Duration').agg({'Calories':['mean','min','max','count']})
aggre


# In[28]:


# Filter the dataframe to select the rows with calories values between 500 and 1000
df[(df['Calories']>=500) & (df['Calories']<=1000)]


# In[29]:


# Filter the dataframe to select the rows with calories values > 500 and pulse < 100
df[(df['Calories']>500) & (df['Pulse']<100)]


# In[30]:


# Create a new “df_modified” dataframe that contains all the columns from df except for “Maxpulse”
df_modified = df[['Duration', 'Pulse', 'Calories']]
df_modified


# In[31]:


# Delete the “Maxpulse” column from the main df dataframe
df = df.drop('Maxpulse', axis=1)
df


# In[32]:


# Convert the datatype of Calories column to int datatype
df['Calories'] = df['Calories'].astype('int64')
df.dtypes


# In[33]:


# Using pandas create a scatter plot for the two columns (Duration and Calories)
df.plot.scatter(x='Duration', y='Calories')


# In[6]:


import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot as plt

titanic = pd.read_csv(""C:/Users/MaherSarosh/Downloads/Dataset (1)/Dataset/train.csv"")
print(titanic)

# Finding the correlation between ‘survived’ (target column) and ‘sex’ column

# To get a correlation we need to convert our categorical features to numerical ones.
# Of course the choice of order will affect the correlation but luckily all of our categories seem to be binary
titanic['Survived'] = titanic['Survived'].astype('category').cat.codes
titanic['Sex'] = titanic['Sex'].astype('category').cat.codes

# Used corr() function to find the correlation
correlation_Value = titanic['Sex'].corr(titanic['Survived'])
print("\nThe correlation between ‘survived’ (target column) and ‘sex’ column is : ", correlation_Value)

# ------------------------------------------------------------------------------------------------------------------------------
# a. Do you think we should keep this feature?
#    Yes we should keep this because a large negative correlation is just as useful as a large positive correlation.
#   The only difference is that for a positive correlation, as the feature increases, the target will increase.
#   For a negative correlation, as the feature decreases, the target will increase.

# ------------------------------------------------------------------------------------------------------------------------------
# Visualization of the correlation using heatmap, histplot, scatterplot

# Dropping the unnecessary columns
data = titanic.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis='columns')

# Removing all the data that has missing values
processed_data = data.dropna(axis=0)

# Converts categorical values in the 'Sex' column into numerical values.
data1 = pd.get_dummies(processed_data, drop_first=True)

# Converting the datatype to Float
data1["Survived"] = data1["Survived"].astype(float)
data1["Pclass"] = data1["Pclass"].astype(float)
data1["Sex"] = data1["Sex"].astype(float)

# Calculated the correlation matrix using corr() function
correlation_matrix = data1.corr().round(2)  # Round to 2 decimal places
print("\n Correlation Matrix : \n", correlation_matrix)  # display correlation matrix

# Creating plot
sns.heatmap(data=correlation_matrix, annot=True)  # Set annot = True to print the values inside the squares
# show plot
plt.show()

# Creating plot
sns.histplot(data=correlation_matrix)
# show plot
plt.show()

# Creating plot
sns.scatterplot(data=correlation_matrix)
# show plot
plt.show()

# From the correlation matrix , we can observe that 'Pclass' and 'Fare' have a correlation of -0.55.
# This suggests that these feature pairs are strongly correlated to each other.
# Considering multicollinearity, let's drop the 'Fare' column since it has lower correlation
# with 'Survived' compared to 'Pclass'.
final_df = data1.drop('Fare', axis=1)

# ------------------------------------------------------------------------------------------------------------------------------
# Classification using Gaussian Naive Bayes
x = final_df.drop('Survived', axis=1)
y = final_df['Survived']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
classifier = GaussianNB()
y_pred = classifier.fit(x_train, y_train).predict(x_test)

# Summary of the predictions made by the Gaussian Naive Bayes classifier
print("\nClassification using Gaussian Naive Bayes")
print("Classification Report : \n", classification_report(y_test, y_pred))
print("Confusion Matrix : \n", confusion_matrix(y_test, y_pred))

# Accuracy score
print('accuracy is', accuracy_score(y_pred, y_test))


# In[5]:


import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings  # current version generates a bunch of warnings that we'll ignore
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
import seaborn as sns

warnings.filterwarnings("ignore")

# glass is a dataframe that we load the glass.csv data into.
glass = pd.read_csv(""C:/Users/MaherSarosh/Downloads/Dataset (1)/Dataset/glass.csv"")

x = glass.iloc[:, :-1].values
y = glass.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Classification using Gaussian Naive Bayes
classifier = GaussianNB()
y_pred = classifier.fit(x_train, y_train).predict(x_test)

# Summary of the predictions made by the Gaussian Naive Bayes classifier
print("Classification using Gaussian Naive Bayes\n")
print("Classification Report : \n", classification_report(y_test, y_pred))
print("Confusion Matrix : \n", confusion_matrix(y_test, y_pred))
# Accuracy score
print('accuracy is', accuracy_score(y_pred, y_test))

# ------------------------------------------------------------------------------------------------------------------------------
# Classification using Linear Support Vector Machine's
classifier = LinearSVC(verbose=0)
y_pred = classifier.fit(x_train, y_train).predict(x_test)

# Summary of the predictions made by the LinearSVC classifier
print("\n-----------------------------------------------------------------------------------------\n"
      "Classification using Linear Support Vector Machine's\n")
print("Classification Report : \n ", classification_report(y_test, y_pred))
print("Confusion Matrix : \n", confusion_matrix(y_test, y_pred))
# Accuracy score
print('accuracy is', accuracy_score(y_pred, y_test))

# -----------------------------------------------------------------------------------------------------------------------------
# Visualization of the correlation using heatmap, histplot, scatterplot
correlation_matrix = glass.corr().round(2)  # Round to 2 decimal places
print("\n Correlation Matrix : \n", correlation_matrix)  # display correlation matrix

# Creating plot
sns.heatmap(data=correlation_matrix, annot=True)  # Set annot = True to print the values inside the squares
# show plot
plt.show()

# Creating plot
sns.histplot(data=correlation_matrix)
# show plot
plt.show()

# Creating plot
sns.scatterplot(data=correlation_matrix)
# show plot
plt.show()


# In[ ]:




