# GROUP PROJECT OVERVIEW: Machine Learning Predictive Model, Canadian Immigration and Rental Prices

# Table of Contents
* [Introduction](#introduction)
* [Code and Setup](#code-and-setup)
* [Step-by-Step Guide](#step-by-step-guide)
* [Problem Definition](#problem-definition)
* [Data Gathering](#data-gathering)
* [Data Preparation & Exploratory Data Analysis](#data-cleaning-and-exploratory-data-analysis)
* [Model Building](#model-building)
* [Model Performance](#model-performance)
* [Model Validation and Hyperparamater Tuning](#model-building-and-hyperparameter-tuning)
* [Model Evaluation](#model-evaluation)
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

**STEP 4: EDA (Exploratory Data Analysis):** It is important to use descriptive and graphical statistics to look for patterns, correlations, and comparisons in the dataset. In this step, we used heatmaps and correlation matrices to analyze the data.

**STEP 5: Data Modelling:** In this project, we used Linear Regression and Random Forest Regressor.

**STEP 6: Validate Model:** After training the model, we carried out cross-validation using k-fold cross validation technique to assess model performance and generalization.

**STEP 7: Optimize Model:** We performed Hyperparameter Tuning on Linear Regression and Random Forest Regressor models using GridSearchCV to reach the best models.

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

  <ul>
    <li>Target Variable=MonthlyRent</li>
    <li>Feature variables categorized into three types: Categorical Variables, Discrete Variables and Continuous Variables.</li>
    <li>Pairwise plots were produced using Seaborn to identify patterns and relationships.</li>
  </ul>
  
<img src="https://github.com/Francis-Calingo/Canadian-Rental-Prices-and-Immigration-ML-Predictive-Model/blob/main/Figures/Figure1.png"/>

  <ul>
    <li>The following features had the most influence on MonthlyRent, and to be used for model building: City, RentalType, Year, Month, Bedrooms, 
SquareFootage, and AnnualPropertyTax.</li>
    <li>Performed Categorical Encoding and Standardized Scaling for pre-processing pipeline, where:</li>
    <ul>
      <li>Dependant variable=MonthlyRent</li>
      <li>Categorical variables=City, RentalType</li>
      <li>Discrete variables=Year, Month, Bedrooms, SquareFootage, Admissions</li>
      <li>Continuous variables=AnnualPropertyTax</li>
    </ul>

[<b>Back to Table of Contents</b>](#table-of-contents)

---

# Model Building

## Model Building Steps

<ul>
    <li>Steps for model selection (both linear regression and random forest regressor):</li>
    <ul>
      <li>(1) Define target (y) and features (X) </li>
      <li>(2) Encode categorical features</li>
      <li>(3) Split data into training and testing sets, size=0.2</li>
      <li>(4) Initialize, then train model</li>
      <li>(5) Get model coefficients and intercept</li>
      <li>(6) Make predictions on test set</li>
      <li>(7) Evaluate model</li>
      <li>(8) Perform k-fold cross-validation (k=5)</li>
      </ul>
    <li>Performed hyperparameter tuning on random forest regressor using GridSearchCV.</li>
  </ul>

[<b>Back to Table of Contents</b>](#table-of-contents)

---

# Model Performance

<b>Linear Regression Model:</b>
  <ul>
    <li><b>RMSE:</b> 114799.009005</li>
    <li><b>R^2 Score:</b> 0.935382</li>
   <li><b>Average MSE from Cross-Validation:</b> 626996.754116</li>
    </ul>
<b>Random Forest Regressor:</b>
  <ul>
    <li><b>RMSE:</b> 34611.563041</li>
    <li><b>R^2 Score:</b> 0.980581</li>
   <li><b>Average MSE from Cross-Validation:</b> 716345.255222</li>
    </ul>

Random Forest Regressor performed better. As it is more suited for non-linear data, suggesting data's non-linearity.

[<b>Back to Table of Contents</b>](#table-of-contents)

---

# Model Validation & Hyperparameter Tuning

1) We ensured that our categorical columns were consistent (i.e., datatype set as string) 
2) Set target variable name (i.e., target_variable=’MonthlyRent’) 
3) Identify and set categorical and numerical columns 
4) Exclude target variable from features 
5) Separate target and features 
6) Define preprocessing for categorical data 
7) Define the Random Forest Model 
8) Create a Pipeline, then define the parameter grid for tuning:
 9) Split the dataset into training and test sets 
10) Perform grid search with Cross-Validation 
11) Print the best parameters and score, then train the model with said parameters 

[<b>Back to Table of Contents</b>](#table-of-contents)

---

# Model Evaluation

---

# Predictions

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
