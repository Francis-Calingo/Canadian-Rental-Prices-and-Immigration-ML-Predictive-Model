# GROUP PROJECT OVERVIEW: Machine Learning Predictive Model, Canadian Immigration and Rental Prices

# Table of Contents
* [Introduction](#introduction)
* [Code and Setup](#code-and-setup)
* [Web Scraping](#web-scraping)
* [Feature Engineering](#feature-engineering)
* [Data Cleaning & Exploratory Data Analysis](#data-cleaning--exploratory-data-analysis)
* [Model Building](#model-building)
* [Model Performance](#model-performance)
* [Predictions](#predictions)
* [Discussion](#discussion)
* [Credits and Acknowledgements](#credits-and-acknowledgements)

---

# Introduction

  <ul>
    <li>Predict Rental Prices across 6 major Canadian cities in 2025 given permanent resident data and rental price data.</li>
    <li>Scraped Permanent Resident Admissions data (since 2015) from Immigration, Refugees and Citizenship Canada.</li>
    <li>Generated synthetic rental price data using a comprehensive ChatGPT prompt to compensate for lack of proprietary data.</li>
    <li>Performed Hyperparameter Tuning on Linear Regression and Random Forest Regressor models using GridSearchCV to reach the best models.</li>
  </ul>

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

# Web Scraping

Permanent Residence admission Data was scraped from this website: https://open.canada.ca/data/en/dataset/f7e5498e-0ad8-4417-85c9-9b8aff9b9eda/resource/81021dfd-c110-42cf-a975-1b9be8b82980 

---

# Feature Engineering

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

# Data Cleaning & Exploratory Data Analysis

  <ul>
    <li>Target Variable=MonthlyRent</li>
    <li>Feature variables categorized into three types: Categorical Variables, Discrete Variables and Continuous Variables.</li>
    <li>Pairwise plots were produced using Seaborn to identify patterns and relationships.</li>
  </ul>
  
![image](https://github.com/user-attachments/assets/49096bf0-eb45-4adb-bafb-11d2a8a7bd10)

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
</details>

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

# Predictions

![image](https://github.com/user-attachments/assets/75376878-ca52-4a8a-b331-13c643545bc9)

![image](https://github.com/user-attachments/assets/df9238ad-b39d-4980-80db-531eb75f56d3)

![image](https://github.com/user-attachments/assets/9b177835-2b3f-4a67-965d-f05782729f03)

![image](https://github.com/user-attachments/assets/20560199-0a42-4c34-ac35-cef9e795058a)

![image](https://github.com/user-attachments/assets/1e33eec0-f6db-4a03-a0b3-18add97aebf1)

![image](https://github.com/user-attachments/assets/9820949b-34ca-497c-94fd-241e8bf13eb1)

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
