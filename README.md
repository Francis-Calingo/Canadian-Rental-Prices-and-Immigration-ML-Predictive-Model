# GROUP PROJECT OVERVIEW: Machine Learning Predictive Model, Canadian Immigration and Rental Prices

# Table of Contents
* [Introduction](#introduction)
* [Code and Setup](#code-and-setup)
* [Step-by-Step Guide](#step-by-step-guide)
* [Problem Definition](#problem-definition)
* [Data Gathering](#data-gathering)
* [Data Preparation and Exploratory Data Analysis](#data-preparation-and-exploratory-data-analysis)
* [Model Building](#model-building)
* [Model Performance](#model-performance)
* [Model Validation and Hyperparamater Tuning](#model-validation-and-hyperparameter-tuning)
* [Predictions](#predictions)
* [Discussion](#discussion)
* [Credits and Acknowledgements](#credits-and-acknowledgements)

---

# Introduction
With both housing prices and immigration levels reaching record (or near-record) highs in Canada, it's no surprise that immigration and housing have become hotly-debated topics in the country in recent years. Some have gone as far as to claim that the high housing costs are directly connected to high levels of immigration. We decided that this presented a good opportunity to use our knowledge in machine learning models, and check the veracity of those claims by attempting to build a predictive regression model to predict rental prices across 6 major Canadian metropolitan areas given permanent resident data and rental price data.

[<b>Back to Table of Contents</b>](#table-of-contents)

---

# Code and Setup
 
  <ul>
    <li><b>IDEs Used:</b> Google Colab, Jupyter Notebook</li>
    <li><b>Python Version:</b> 3.10.12</li>
    <li><b>Libraries and Packages:</b>
    <ul>
      <li><b><i>sklearn Packages:</i> </b> ColumnTransformer, Pipeline, StandardScaler, OneHotEncoder, train_test_split, LinearRegression, mean_squared_error, r2_score, RandomForestRegressor, make_scorer, cross_val_score, GridSearchCV</li>
      <li><b> <i>Other Packages:</i> </b> pandas, files (from google.colab), seaborn</li>
    </ul></li>
    <li><b>ChatGPT version:</b> GPT-4</li>
  </ul>

```bash
git clone https://github.com/Francis-Calingo/Canadian-Rental-Prices-and-Immigration-ML-Predictive-Model.git
cd Canadian-Rental-Prices-and-Immigration-ML-Predictive-Model
```

To install the necessary Python libraries and packages:
```bash
pip install -r requirements.txt
```

[<b>Back to Table of Contents</b>](#table-of-contents)

---

# Step-by-step Guide

**STEP 1: Problem Definition:** Will new permanent resident targets set out by the federal government decrease rental prices across Canada? Since rental prices are a continuous variable, this is a regression problem.

**STEP 2: Data Gathering:** We scraped Permanent Resident Admissions data (since 2015) from Immigration, Refugees and Citizenship Canada, and enerated synthetic rental price data using a comprehensive ChatGPT prompt to compensate for lack of proprietary data.

**STEP 3: Data Preparation:** We prepared the data by using scaling methods.

**STEP 4: EDA (Exploratory Data Analysis):** It is important to use descriptive and graphical statistics to look for patterns, correlations, and comparisons in the dataset. In this step, we used Seaborn pairplots and boxplots to analyze the data.

**STEP 5: Data Modelling:** In this project, we used Linear Regression and Random Forest Regressor.

**STEP 6: Validate Model:** After training the model, we carried out cross-validation using k-fold cross validation technique to assess model performance and generalization.

**STEP 7: Optimize Model with Hyperparameter Tuning:** We performed Hyperparameter Tuning on Linear Regression and Random Forest Regressor models using GridSearchCV to reach the best models.

**STEP 8: Evaluate Model:**

**STEP 9: Make Predictions:**


[<b>Back to Table of Contents</b>](#table-of-contents)

---

# Problem Definition

In October 2024, Canadian Prime Minister Justin Trudeau announced a significant reduction in immigration targets previously set for 2025 - 2027, including a 21% decrease in the number of permanent residents (PR) being admitted into the country. As cited by the federal government, in order to relieve major sectors of Canada’s infrastructure, such as the rental and housing markets, permanent resident targets will be reduced from half a million admissions to 395,000. The objective of this report was to determine whether or not reductions in immigration targets set out by the Government of Canada will in fact reduce prices in the rented accommodation market. By using supervised learning, and specifically regression to predict rental prices in Canada, this analysis can determine the degree of impact that changes in PR admissions will have on the average cost of renting a home in Canada.

**H<sub>0</sub>: "New PR admissions targets will not reduce rental accommodation costs in Canada"**


[<b>Back to Table of Contents</b>](#table-of-contents)

---

# Data Gathering

## Webscraping Immigration Data

Permanent Residence admission Data was scraped from this website: https://open.canada.ca/data/en/dataset/f7e5498e-0ad8-4417-85c9-9b8aff9b9eda/resource/81021dfd-c110-42cf-a975-1b9be8b82980 

## Feature Engineering with Synthetic Rental Data

The following ChatGPT prompt was used to generate our synthetic data:

```
Generate a realistic dataset of rental prices for major Canadian cities, including Vancouver, Toronto, Montreal, Calgary, Ottawa, Edmonton, and Halifax. The dataset should include:

1.	Data Columns:

-City: Major cities like Toronto, Vancouver, Montreal, Calgary, etc.
-Province: Corresponding provinces (e.g., Ontario, British Columbia).
-Year: From 2019 to 2023.
-Month: January to December.
-Rental Type: Apartment, Condo, Detached House, Townhouse.
-Number of Bedrooms: 1, 2, 3, 4, etc.
-Number of Bathrooms: 1, 2, 3, etc.
-Square Footage: Ranges for different rental types.
-Furnished: Yes/No.
-Pet Friendly: Yes/No.
-Parking Included: Yes/No.
-Distance to City Center (km): Numeric value.
-Monthly Rent (Target): Dependent variable, with realistic pricing trends.
-Walk Score: A score between 0 and 100 indicating walkability.
-Transit Score: A score between 0 and 100 indicating access to public transit.
-Age of Building: Number of years since the building was constructed.
-Energy Efficiency Rating: Numeric score (e.g., 0–10).
-Lease Term: Length of the lease in months (e.g., 6, 12, 24).
-Noise Level: Numeric score (e.g., 1–10, with 10 being very noisy).
-Nearby Schools Rating: Average rating of schools in the area (1–10).
-Internet Availability: Yes/No indicating high-speed internet availability.
-Crime Rate Index: A score representing the area's safety.
-Annual Property Tax: Approximation based on rent and location.

2.	Realism:

-Average monthly rent should reflect the general cost of living in each city. For example, Vancouver and Toronto should have higher average rents compared to Edmonton or Halifax.
-Include a range of rental prices within cities to capture variability (e.g., downtown areas vs. suburban neighborhoods).
-Use realistic distributions for rental prices, square footage, and proximity to transit. For instance, apartments should generally be smaller and less expensive than single-family homes.

3.	Additional Notes:

-Include 10,000 rows of data distributed proportionally across cities.
-Reflect seasonality and trends where applicable (e.g., higher prices in Toronto and Vancouver for smaller units due to demand).
-Ensure property types align with city norms (e.g., more condos in downtown Toronto, more single-family homes in Calgary suburbs).

```

[<b>Back to Table of Contents</b>](#table-of-contents)

---

# Data Preparation and Exploratory Data Analysis

## Loading Data

Since we used Google Colab, this is how we uploaded our data and saw a preview of the data:

```jupyter
from google.colab import files
uploaded = files.upload()
# steps:
#1.Download three files to the local:  'synthetic_rental_prices_canada_updated.csv' and 'PR_Admissions_unpivoted.xlsx' and 'PR_Admissions_unpivoted_added_2025'
#2.upload the files to the files under this colab notebook
```

```jupyter
# read csv file with rent by city
import pandas as pd
csv_file = 'synthetic_rental_prices_canada_updated.csv'
df_rental = pd.read_csv(csv_file)
df_rental.tail()
```

<img src="https://github.com/Francis-Calingo/Canadian-Rental-Prices-and-Immigration-ML-Predictive-Model/blob/main/Figures/EDA1.png"/>

```jupyter
# read xlsx file for the pr data with city and date
xlsx_file = 'PR_Admissions_unpivoted.xlsx'
df_pr= pd.read_excel(xlsx_file)
df_pr.tail()
```
<img src="https://github.com/Francis-Calingo/Canadian-Rental-Prices-and-Immigration-ML-Predictive-Model/blob/main/Figures/EDA2.png"/>


## Exploratory Data Analysis

For the exploratory data analysis on high-impact columns:

* Target Variable=MonthlyRent
 
* Feature variables categorized into three types: Categorical Variables, Discrete Variables and Continuous Variables.
 
* Pairwise plots were produced using Seaborn to identify patterns and relationships.
  
<img src="https://github.com/Francis-Calingo/Canadian-Rental-Prices-and-Immigration-ML-Predictive-Model/blob/main/Figures/Figure1.png"/>

## Data Preparation
City, RentalType, Year, Month, Bedrooms, SquareFootage, and AnnualPropertyTax were the features that had the most influence on MonthlyRent, and are therefore to be used for model building.

But before proceeding, we performed Categorical Encoding and Standardized Scaling for pre-processing pipeline, where:

* **Dependant variable:** MonthlyRent

* **Categorical variables:** City, RentalType

* **Discrete variables:** Year, Month, Bedrooms, SquareFootage, Admissions

* **Continuous variables:** AnnualPropertyTax

Code snippet:

```jupyter
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

dep_var = ['MonthlyRent']

categorical_vars_comb = ['City', 'RentalType']

discrete_vars_comb = ['Year', 'Month', 'Bedrooms', 'SquareFootage','Admissions']

continuous_vars_comb = ['AnnualPropertyTax']

# Categorical - one hot encode
cat_ohe_step = ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
cat_steps = [cat_ohe_step]
cat_pipe = Pipeline(cat_steps)
cat_transformers = [('cat', cat_pipe, categorical_vars_comb)]

# Numerical -  scale
num_scl_step = ('scl', StandardScaler())
num_steps = [num_scl_step]
num_pipe = Pipeline(num_steps)
num_transformers = [('num', num_pipe, discrete_vars_comb + continuous_vars_comb)]

ct = ColumnTransformer(transformers=cat_transformers + num_transformers)

ct.fit(df_combine[categorical_vars_comb + discrete_vars_comb + continuous_vars_comb])
X=ct.transform(df_combine[categorical_vars_comb + discrete_vars_comb + continuous_vars_comb])
y=df_combine[['MonthlyRent']].values

```

The we executed a train-test split. We split the data into test and train sets. The train set will be used to train the model, while the test set will be used to test the model's performance.

```jupyter
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```


[<b>Back to Table of Contents</b>](#table-of-contents)

---

# Model Building
To make our model selection varied, we opted to go for one model that worked best for determining linear relationships, and for the other model that worked best for handling non-linear relationships. We used the following two supervised learning regression models: Linear Regression and Random Forest Regressor. 

**Linear Regression:** Following feature engineering, we split our data into training and testing sets (test size=20%).  Then, we trained the linear regression model, then made predictions on the training set.
**Random Forest Regressor:** The algorithm is virtually the same as for the linear regression model, this time setting model=RandomForestRegressor() with n_estimators=100 and random_state=0.

## Model Building Steps

Steps for model selection (both linear regression and random forest regressor):

* (1) Define target (y) and features (X)

* (2) Encode categorical features</li>

* (3) Split data into training and testing sets, size=0.2

* (4) Initialize, then train model

* (5) Get model coefficients and intercept
 
* (6) Make predictions on test set

* (7) Evaluate model

* (8) Perform k-fold cross-validation (k=5)

[<b>Back to Table of Contents</b>](#table-of-contents)

---

# Model Performance

<b>Linear Regression Model:</b>
  <ul>
    <li><b>RMSE:</b> 114799.009005</li>
    <li><b>R^2 Score:</b> 0.935382</li>
   <li><b>Average MSE from Cross-Validation:</b> 626996.754116</li>
    </ul>

<img src="https://github.com/Francis-Calingo/Canadian-Rental-Prices-and-Immigration-ML-Predictive-Model/blob/main/Figures/Eval1.1.png"/>

<img src="https://github.com/Francis-Calingo/Canadian-Rental-Prices-and-Immigration-ML-Predictive-Model/blob/main/Figures/Eval1.2.png"/>

 
<b>Random Forest Regressor:</b>
  <ul>
    <li><b>RMSE:</b> 34611.563041</li>
    <li><b>R^2 Score:</b> 0.980581</li>
   <li><b>Average MSE from Cross-Validation:</b> 716345.255222</li>
    </ul>

<img src="https://github.com/Francis-Calingo/Canadian-Rental-Prices-and-Immigration-ML-Predictive-Model/blob/main/Figures/Eval2.1.png"/>

<img src="https://github.com/Francis-Calingo/Canadian-Rental-Prices-and-Immigration-ML-Predictive-Model/blob/main/Figures/Eval2.2.png"/>

Random Forest Regressor performed better. As it is more suited for non-linear data, suggesting data's non-linearity.

[<b>Back to Table of Contents</b>](#table-of-contents)

---

# Model Validation and Hyperparameter Tuning

Performed hyperparameter tuning on random forest regressor using GridSearchCV.

* (1) We ensured that our categorical columns were consistent (i.e., datatype set as string) 

* (2) Set target variable name (i.e., target_variable=’MonthlyRent’) 

* (3) Identify and set categorical and numerical columns 

* (4) Exclude target variable from features 

* (5) Separate target and features 

* (6) Define preprocessing for categorical data 

* (7) Define the Random Forest Model 

* (8) Create a Pipeline, then define the parameter grid for tuning:

* (9) Split the dataset into training and test sets 

* (10) Perform grid search with Cross-Validation 

* (11) Print the best parameters and score, then train the model with said parameters

Code snippet:

```jupyter
# Hyperparameter Tuning for Random Forest Regressor 
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


df_combine = pd.merge(df_rent_clean, df_pr_agg, on=['City', 'Year', 'Month'], how='left')
df_combine.drop(['Date', 'Province'], inplace=True, axis=1)  # Drop non-important columns

# Ensure categorical columns are consistent
for col in df_combine.select_dtypes(include=['object', 'category']).columns:
    df_combine[col] = df_combine[col].astype(str)

# Set the actual target variable name
target_variable = 'MonthlyRent'  # Replace with the correct target column name

# Identify categorical and numerical columns
categorical_columns = df_combine.select_dtypes(include=['object']).columns.tolist()
numerical_columns = df_combine.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Exclude the target variable from the features
numerical_columns = [col for col in numerical_columns if col != target_variable]

# Separate features and target
X = df_combine.drop(columns=[target_variable])
y = df_combine[target_variable]

# Define preprocessing for categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ],
    remainder='passthrough'  # Keep numerical columns untouched
)

# Define the Random Forest model
model = RandomForestRegressor(random_state=42)

# Create a pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# Define the parameter grid for tuning
param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [10, 20, None],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform Grid Search with Cross-Validation
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Display the best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Negative Mean Squared Error:", grid_search.best_score_)

# Train the model with the best parameters
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print("Test Set Score (R^2):", test_score)
```

Results:

<img src="https://github.com/Francis-Calingo/Canadian-Rental-Prices-and-Immigration-ML-Predictive-Model/blob/main/Figures/Tuning.png"/>

[<b>Back to Table of Contents</b>](#table-of-contents)

---

# Predictions

## Testing Best Model

```jupyter
## Testing the best model 

y_pred = best_model.predict(X_test) 
residuals = y_test - y_pred  

fig = plt.figure(figsize=(10, 6))  
fig.subplots_adjust(hspace=.5)
ax1 = fig.add_subplot(211) 
ax2 = fig.add_subplot(212)
ax1.scatter(y_pred, residuals, alpha=0.6)
ax1.axhline(y=0, color='r', linestyle='--')
ax1.set_xlabel('Predicted Values')
ax1.set_ylabel('Residuals')
ax1.set_title('Residual Plot')

ax2.hist(residuals, bins = 35,)
ax2.set_xlabel('Predicted Values')
ax2.set_title('Residuals Histogram')

fig.show()
```

<img src="https://github.com/Francis-Calingo/Canadian-Rental-Prices-and-Immigration-ML-Predictive-Model/blob/main/Figures/PredictTest1.png"/>

Residuals are well distributed along the predicted value.


```jupyter
## Checking for overfittng  

# Evaluate R^2 on the training set
train_score = best_model.score(X_train, y_train)

# Print train and test scores
print("Training Set Score (R^2):", train_score)
print("Test Set Score (R^2):", test_score)
```

<img src="https://github.com/Francis-Calingo/Canadian-Rental-Prices-and-Immigration-ML-Predictive-Model/blob/main/Figures/PredictTest2.png"/>

Training score is slightly better, but it is not enough to indicate overfitting. Overall the model is generalizing well and R2 score is exceptional.

```jupyter
## Checking for feature importances 
import numpy as np 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Get the feature names after preprocessing
preprocessor = best_model.named_steps['preprocessor']
ohe = preprocessor.named_transformers_['cat']

# Extracting names for one-hot encoded features
ohe_feature_names = ohe.get_feature_names_out(categorical_columns)

# Combine numerical and one-hot encoded feature names
all_feature_names = list(ohe_feature_names) + numerical_columns

# Extract feature importance from the Random Forest model
importances = best_model.named_steps['model'].feature_importances_

# Sort and plot feature importance
import numpy as np
import matplotlib.pyplot as plt

sorted_indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))

plt.bar(range(len(importances)), importances[sorted_indices], align='center')
plt.xlim((0,10))
plt.xticks(range(len(importances)), np.array(all_feature_names)[sorted_indices], rotation=90)
plt.title('Feature Importance')
plt.show()
```

<img src="https://github.com/Francis-Calingo/Canadian-Rental-Prices-and-Immigration-ML-Predictive-Model/blob/main/Figures/PredictTest3.png"/>

## Prediction Results

Steps:

* (1) Use the best_model.predict() function on the combined dataset (with ‘MonthlyRent’ column dropped)

* (2) Then the prediction dataset was combined with the historic rent dataset via concatenation.

* (3) Admissions data was prepared for plotting (for the purposes of this project, we assumed that admissions levels will remain the same for 2024 and 2025).

* (4) Admissions data and combined rent data for Canada’s largest cities (i.e., Toronto, Montreal, Vancouver, Calgary, Edmonton, Ottawa) were plotted side-by-side

<img src="https://github.com/Francis-Calingo/Canadian-Rental-Prices-and-Immigration-ML-Predictive-Model/blob/main/Figures/Toronto%20Prediction.png"/>

<img src="https://github.com/Francis-Calingo/Canadian-Rental-Prices-and-Immigration-ML-Predictive-Model/blob/main/Figures/Montreal%20Prediction.png"/>

<img src="https://github.com/Francis-Calingo/Canadian-Rental-Prices-and-Immigration-ML-Predictive-Model/blob/main/Figures/Vancouver%20Prediction.png"/>

<img src="https://github.com/Francis-Calingo/Canadian-Rental-Prices-and-Immigration-ML-Predictive-Model/blob/main/Figures/Calgary%20Prediction.png"/>

<img src="https://github.com/Francis-Calingo/Canadian-Rental-Prices-and-Immigration-ML-Predictive-Model/blob/main/Figures/Edmonton%20Prediction.png"/>

<img src="https://github.com/Francis-Calingo/Canadian-Rental-Prices-and-Immigration-ML-Predictive-Model/blob/main/Figures/Ottawa%20Prediction.png"/>

[<b>Back to Table of Contents</b>](#table-of-contents)

---

# Discussion


<P>After completing our prediction using Random Forest Regressor, we can observe that 2025 rental prices appear lower in many cities (i.e. Vancouver, Montreal and Calgary). Additionally, by observing feature importances, it is clear that PR admissions, while not the most important feature, does meaningfully contribute to the model’s ability to predict rental prices. As such, we can reject the null hypothesis that the new PR admission targets will not reduce rental accommodation costs in Canada. </P>

<P>By using synthetic data for rental prices, we were able to build a well performing, generalized model that can be used to predict economic impacts relative to immigration policies. Synthetic data is a useful tool businesses and policy makers can leverage to demonstrate technical capabilities, and serve as a proof of concept in order to justify any capital expenses required to gather real-world, proprietary data. While the prompt required ChatGPT to provide realistic data, future EDA could be done with real-world proprietary data to compare how realistic synthetic data was.</P>

<P>Visualizing rental prices side-by-side with admissions data does show that rental prices in 2025 would decrease, but not by a significant margin. While the scope of this report was aimed to examine the impact of permanent resident admissions, future studies could leverage temporary resident (TR) immigration targets. TR admissions and other immigration categories could be included to develop a more holistic analysis to understand the relationship between immigration policy and Canada’s economic outlook.</P>

[<b>Back to Table of Contents</b>](#table-of-contents)

---

# Credits and Acknowledgements

Great Learning. “RMSE: What Does It Mean?” Medium, 26 Apr. 2021, https://medium.com/@mygreatlearning/rmse-what-does-it-mean-2d446c0b1d0e.

“R-Squared in Regression Analysis in Machine Learning.” GeeksforGeeks, 7 May 2019, https://www.geeksforgeeks.org/ml-r-squared-in-regression-analysis/.

"What Is Synthetic Data?" MOSTLY AI. 24 Sept. 2021, https://mostly.ai/what-is-synthetic-data.

"What Is Prompt Engineering?" McKinsey. https://www.mckinsey.com/featured-insights/mckinsey-explainers/what-is-prompt-engineering. 

[<b>Back to Table of Contents</b>](#table-of-contents)
