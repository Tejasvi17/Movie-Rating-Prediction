# Movie-Rating-Prediction


Here’s a step-by-step breakdown of what we’ll do :
Data Collection: Gather a comprehensive dataset of movies that includes information about genres, directors, actors, and movie ratings. You can acquire this data from sources like IMDb, Kaggle, or other movie databases.

Data Preprocessing: Clean and prepare the dataset for analysis. This includes handling missing data, removing duplicates, and converting categorical variables (e.g., genres) into a suitable format for modeling.

Feature Engineering: Identify and create relevant features from the dataset. Consider the impact of variables such as the number of famous actors in a movie or the director's track record on the movie's rating.

Exploratory Data Analysis (EDA): Conduct an in-depth analysis of the dataset to understand the relationships between features and movie ratings. Visualize the data to discover patterns and correlations.

Model Selection: Choose appropriate regression models to predict movie ratings. Experiment with different algorithms, such as linear regression, decision trees, random forests, or gradient boosting.

Model Training: Split the dataset into training and testing sets. Train the selected models on the training data, and fine-tune hyperparameters to optimize their performance.

Model Evaluation: Evaluate the models using appropriate metrics such as Mean Absolute Error (MAE), Root Mean Square Error (RMSE), or R-squared. Select the best-performing model for the final prediction.

## 1. Importing Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

## 2. Loading Dataset
data = pd.read_csv('IMDb_Movies_India.csv')

## 3. Data Exploration
%%HTML
<style type="text/css">
table.dataframe td, table.dataframe th {
border: 1px solid black !important;
}
</style>

data.head()
data.shape
data.describe()
data.info()
data.isnull().mean()*100


## 4. Data PreProcessing

data.dropna(inplace= True)
data.head()
# Remove commas and convert 'Votes' to integer
data['Votes'] = data['Votes'].str.replace(',', '', regex=True).astype(int)
data.head()
data.shape

### 4.2 - Removing Duplicates from Dataset :
data.drop_duplicates(inplace = True)
data.shape
data.info()
data.describe()


### 4.3 - Data Visualizations:
**1 - Rating Ditribution :**
plt.figure(figsize=(8, 6))
plt.hist(data['Rating'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Movie Ratings')
plt.show()

**2 - Count of Movies Released Each Year:**
plt.figure(figsize=(12, 6))
sns.countplot(data=data, x='Year')
plt.xlabel('Year')
plt.ylabel('Number of Movies')
plt.title('Number of Movies Released Each Year')
plt.xticks(rotation=90)
plt.show()

**3. Movie Duration vs Rating :**
sns.scatterplot(x='Duration', y ='Rating', data=data)

**4. Movie Rating vs Votes :**
sns.scatterplot(x='Rating', y ='Votes', data=data)


### 4.4 - Encoding text data into Numerical form
data.head()
categorical_features = ['Genre', 'Director', 'Actor 1','Actor 2','Actor 3']
for feature in categorical_features:
    le = LabelEncoder()
    data[feature] = le.fit_transform(data[feature])


### 4.5 - Defining Features and and Target variable
x = data[['Genre', 'Director', 'Actor 1','Actor 2','Actor 3']]
y = data['Rating']
print(x)
print(y)


