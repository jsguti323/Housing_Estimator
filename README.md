
## Housing Estimator




## Overview
Our final project creates a model to predict median house values in California based on the US government's Census data from 1990. The objective is to identify which features impact housing values and to train a model which can predict median house prices in any area with those features. 

Following the sequence of data preparation, database creation, feature engineering and selection, the team focused analysis on three machine learning models: Linear Regression, Random Forest Regressor, and Gradient Boosting Regression. After optimization and comparison, the team concluded the optimized Gradient Boosting Regression to be the most successful model for the study. 

This summary presents the process the team undertook from topic and data selection, through database creation and modeling, to results and conclusion. In it, we detail the technologies we used, our decision-making process, the various iterations of models we considered, and finally lay out our recommendations for the data and the process.  

<details><summary> Preliminary Design and Development </summary>
<p>
  
## Study Design    
The study design followed 5 main steps:  

- Identify the topic  
- Identify our data sources    
- Identify the question to be answered     
- Specify the target variable     
- Determine the model  
  
## Topic and Data Selection    
The topic was selected was housing price trends. Given most recent trends, the team felt that the topic was interesting and relevant, providing a rich opportunity in terms of available data and the broad array of features which can be modeled.  

 ### Criteria for Data Selection    

We selected the California Housing Prices database from Kaggle (https://www.kaggle.com/datasets/camnugent/california-housing-prices, details below), which is a modified version of the 1990 Census showing per-block housing, population, and income information. The Census data  data includes house features (age of the house, number of rooms, number of bedrooms), and community characteristics (median income, number of households, and geographic location). It offers relevant features, and encompasses a wide geographic area. Within those geographic areas it is deeply saturated because of the per-block dimension. It includes geographical location coordinates which can link to a wide range of other data sources.  
	
We added additional context features through weather API calls and county employment rates to expand the scope of the Census data and enlarge the pool of potential influential factors.  In addition to the features above, the team researched availability of community crime statistics and economic indicators.  The deciding factor of whether to include more variables was based on accessibility of the data and on ease with which external data could be merged into the larger dataset. In the end, the team opted for a weather API call and US Census Bureau data on business establishments and number of employees.  

## The Database    

### Description  
Size of the database was the first consideration.  Both overfitting and underfitting are primary concerns in machine learning modelling.  Having a large enough data set helps control for both those conditions. The team set the minimal standard of 10,000 rows to meet this requirement.  This size specification narrowed the number of sources suitable for analysis which led to selecting the California Housing Prices database from Kaggle as the main data source. The external data for county employment figures were derived from census data (Census.gov) and weather from openweathermap.org, both called using APIs. The population information is the Kaggle California cities dataset. After cleaning, restructuring, refining and merging the individual datasets, these four datasets became the production database and subsequently housed in AWS and connected in pgAdmin.

#### Component Datasets: Details
- **Census.csv**: 
    1990 Census data on communities   
    Selected features (3):      
    - counties 
    - Employees 
    - Establishments 
    Observations: 60

 - **Housing.csv**:  
    1990 Census data on housing in communities in California    
    Data is gathered by block: The US Census Bureau Districts (blocks) are the base units for the Census Bureau's survey process. 
    Features (11):  
	
| Column Name | Description |  
| --------------- | --------------- |   
|longitude|A measure of how far west a house is; a higher value is farther west| 
|latitude	|A measure of how far north a house is; a higher value is farther north|
|housingMedianAge	|Median age of a house within a block; a lower number is a newer building|
|totalRooms	|Total number of rooms within a block|
|totalBedrooms	|Total number of bedrooms within a block|
|population	|Total number of people residing within a block|
|households	|Total number of households, a group of people residing within a home unit, for a block|
|medianIncome	|Median income for households within a block of houses (measured in tens of thousands of US Dollars)|
|medianHouseValue	|Median house value for households within a block (measured in US Dollars)|
|oceanProximity	|Location of the house w.r.t ocean/sea|  
	
	
Observations: 20,641	
	
- **Weather data**:  
    Weather for specific date called through weather API  
    Features (5):  
    - Max Temp  
    - Humidity  
    - Cloudiness  
    - Wind Speed  
    - Description  
    Observations: 20,433 (after merge with cleaned housing dataset)  

- **Population data**:  
    Population information by county and city   
    Features (7):  
    - County  
    - City   
    - Incorporation_date  
    - pop_april_1980  
    - pop_april_1990  
    - pop_april_2000  
    - pop_april_2010  
    Observations: 455
	
## Limitations of the Data Set    
While detailed within the features offered, this dataset has some limitations:  
- the data is gathered by block; however, it varies by unit   
    - 3 features are median values:  
        - age of the houses per block;        
        - income of the population per block; and,       
        - value of the houses per block      
    - 4 are totals of the represented features within a block: 
        - number of rooms     
        - number of bedrooms      
        - number of people    
        - number of households      

Scaling the data brings the input data points closer together. However, understanding the data structure is important for sound interpretation of the results. For example, it is difficut to properly weight total number of rooms on a block as part of individual house values. A better metric would be median or average rooms by household or population.
	
Lastly, the data reflects a single point in time, so the  characteristics relevant to house values cannot be observed over time making it static. With time data, it would be possible to see how impacts change with the changes in the values of the features themselves, and thus get a more accurate undersanding of true trends.  	
 
</p>
</details>

<p>
<details><summary>Production Database Architecture</summary>

## Structuring and Cleaning   
Data preparation began with creating a preliminary data structure usng Pandas to merge and join the individual datasets. Creating common columns to link the datasets was the first step.  The housing file did not include any city names, only the geographic coordinates.  The other datasets were identified by city and county.  The initial transformation added the specific city and county names to the housing dataset by using city.py and the location coordinates to list and append each city name to the housing set. 

 ### Census Data  
 #### Starting URL for Census Data API Call.  
![image](https://user-images.githubusercontent.com/101474477/184716368-41dfe441-b1c7-48cb-b852-4c05a77726e4.png)  
	
**Input Dataset**  
 ![image](https://user-images.githubusercontent.com/101474477/184517692-656ea19d-258b-459f-b8a4-61af6fb7cde9.png)  
#### Cleaning and Manipulation ####  
![image](https://user-images.githubusercontent.com/101474477/184716537-6b0d0bac-c97d-4770-9bbf-b14b7c5f3840.png)
![image](https://user-images.githubusercontent.com/101474477/184994274-f1aee790-c05b-4790-a8f3-1111e45c80f5.png)
![image](https://user-images.githubusercontent.com/101474477/184716726-897741d0-3208-408f-9ed2-161de0304d69.png)
	
**Output Dataset**  
![image](https://user-images.githubusercontent.com/101474477/184517942-b7e7fd2d-e4c3-458a-8407-3788593f9d64.png)

### Population Data   
![image](https://user-images.githubusercontent.com/101474477/184717378-1d510ba0-5a36-4649-b808-fa47842dc609.png) 
	
**Input Dataset**  
![image](https://user-images.githubusercontent.com/101474477/184518484-faac1560-0ac1-417b-9197-56e92bf57d7c.png)
	
#### Cleaning and Manipulation ####  
![image](https://user-images.githubusercontent.com/101474477/184717539-cc872e4a-5d0a-460f-844b-fd511dae511b.png)  
	
![image](https://user-images.githubusercontent.com/101474477/184717654-168a40c1-807a-43bc-86d0-133a0509805f.png)  
	
![image](https://user-images.githubusercontent.com/101474477/184717815-00303379-c15e-427c-afa8-8fa18b38cba4.png)  
	
**Output Dataset**  
![image](https://user-images.githubusercontent.com/101474477/184518591-dcf3d531-b956-4e49-9029-66b6bc6b5a35.png)   

### Weather Data  
#### Read the main datafile to join the weather data to:
![image](https://user-images.githubusercontent.com/101474477/184718023-0a5a2049-00a0-41fd-a460-1e3bd76237b9.png)
	
#### Prepare the location coordinates data for processing. Use citypy to join city name to geographical coordinates: 
![image](https://user-images.githubusercontent.com/101474477/184718425-86e23171-cff3-4e6a-ae84-450f3a2f983a.png)
	
#### Initiate API call  
![image](https://user-images.githubusercontent.com/101474477/184718769-5416282c-3fba-4c2d-bf84-28a11deee29f.png)
	
#### Parse the JSON and retrieve data  
![image](https://user-images.githubusercontent.com/101474477/184720029-7fb3cefc-b9d0-44ef-b7b7-34aac7a54968.png)

**Output Dataset**
	
![image](https://user-images.githubusercontent.com/101474477/184518678-260be8a9-4737-423c-b278-c5f38937b350.png)

### Final Dataset  
**Input Dataset**  
![image](https://user-images.githubusercontent.com/101474477/184720607-2749961a-e565-4a26-8f61-8e4dee7f3517.png)
	
![image](https://user-images.githubusercontent.com/101474477/184518831-d28b4d60-2a12-4dfb-ae52-c579e0013152.png)

#### Cleaning and Manipulation ####  
##### Add City to dataset  
	
![image](https://user-images.githubusercontent.com/101474477/184721399-b83c571e-9c75-4fad-9056-e664eaa19757.png)
	
##### Check for null, duplicate values.  Drop as needed  
![image](https://user-images.githubusercontent.com/101474477/184721758-55f06790-0bca-4b0b-a0ff-eddb279ce156.png)

##### Rename, reorder columns  
![image](https://user-images.githubusercontent.com/101474477/184722110-81be2666-2d95-4848-8235-d19f170a3b53.png)
![image](https://user-images.githubusercontent.com/101474477/184722253-427f4bf7-501e-4c68-acb0-8714b81a716b.png)
	
**Output Dataset**  
![image](https://user-images.githubusercontent.com/101474477/184518858-df74aed6-729e-4131-aa14-46b62006a836.png)
	
</p>
</details>

<details><summary>Database Creation and Integration</summary>
<p>
	
## Creating the Table Structure in pgAdmin
	
The team decided to use AWS as the static data repository and use pgAdmin to create the production database. The tables were created in pgAdmin first following the schema:   

![image](https://user-images.githubusercontent.com/101474477/184518914-16ad6780-6e8e-4954-bbc8-e16e3c47df27.png)  
	
The static datasets were then called into pgAdmin through Spark.
	
![image](https://user-images.githubusercontent.com/101474477/184520065-39833e33-0322-4be6-8203-f0e55a328a42.png)

Weather, population, and census were joined into the main dataset, clean_merged_data.csv.  After being instantiated and joined, the final database was saved to a .csv file and read into Pandas for final data preparation and modelling.

**Output database: clean_merged_data.csv**
	**Observations: 11,454**
</p>
</details>

<details><summary>Technologies</summary>
Technologies, languages, tools, and algorithms used throughout the project

<p>

General  
- API calls
- Python 
- Jupyter Notebook
- R Studio
- Pandas
- numpy
- Mlenv environment

Preprocessing  
- sklearn.preprocessing LabelEncoder
- citypy
	
Database Integration  
- AWS Relational Database System
- pgAdmin
- prosgresSQL  

Statistical and Modeling  
- sklearn.ensemble RandomForestRegressor
- sklearn.datasets make_regression
- sklearn.ensemble HistGradientBoostingRegressor
- GradientBoostingRegressor

- sklearn metrics
- collections Counter
- sklearn.metrics accuracy_score, classification_report

- scipy.stats shapiro, kurtosis, skew
	
- sklearn.preprocessing StandardScaler
- sklearn.model_selection train_test_split
- R

Plotting and Visualization  
- matplotlib.pyplot 
- seaborn 
- dabl (Data Analysis Baseline library)
- Tableau
	
</p>
</details>

<details><summary>Data Exploration</summary>

<p>

Concurrent with data cleaning and structuring, the team conducted preliminary data analysis to get a feel for the data itself.  This took the form of histogram and rough regression on the database elements. The objective is to determine whether the data has a normal distribution, measuring skew and kurtosis.  Regression relies on normal distribution for accuracy; outliers reduce accuracy. 
	
Methods to address this in other code variations included normalizing skew 
	
#### Histograms with Density Plots
	
![image](https://user-images.githubusercontent.com/101474477/184750627-77a41ed2-e676-4a9e-a1ff-f85428c83580.png)

![image](https://user-images.githubusercontent.com/101474477/184750975-e9e7a52b-807d-487b-99a1-0ac550a37f82.png)
	
![image](https://user-images.githubusercontent.com/101474477/184752109-9387659d-3bb9-4652-98af-c17592d23527.png)

![image](https://user-images.githubusercontent.com/101474477/185211555-aab7fdbd-8b66-4974-9381-e25249a6e6b2.png)


All follow a non-normal distribution. Households heavily skew left as do population, total rooms, and total bedrooms. Median house age, median income, and median house value are more symmetrically distributed as are maximum temperature, humidity and wind speed. The Shapito-Wilk test for normality bears this out, as all the p-values for the features below are 0.

#### Descriptive Statistics
	
![image](https://user-images.githubusercontent.com/101474477/185213021-4598bef5-83c3-465e-b287-899eeb9df0d2.png)
	
![image](https://user-images.githubusercontent.com/101474477/184695024-12fcfc7b-20b4-4be3-80a8-1bf0f035c7d0.png)

![image](https://user-images.githubusercontent.com/101474477/184990514-c40b0aa4-6698-4565-933f-b60f0c7d6c7f.png)

![image](https://user-images.githubusercontent.com/101474477/184695555-d9ddc7f2-b783-4952-872b-58afdac373aa.png)

![image](https://user-images.githubusercontent.com/101474477/184695794-c6028139-5cd6-4425-8ec0-c592adb7b68e.png)

![image](https://user-images.githubusercontent.com/101474477/184696301-3b4412ba-cec8-4f2d-bb22-56cbfe2edfd8.png)
	
![image](https://user-images.githubusercontent.com/101474477/184696453-e481a84c-5cc8-41f5-bc6f-20ed74d0aea6.png)

#### Simple Univariate Regression  
Regression plots of these variables against the target variable, median house value, are shown below.  The coefficients are the intercept and the slope for each variable. These become the equation for the value predictions for median house value given the value of the independent variable. The formula for the plot of the regression line is y = a+bx where a is the intercept, b is the slope, and x is the value of the independent variable for that observation. 
	
**Population**  
Y Coefficients:  
|Intercept	|Population	|
|---------------|---------------|
|	225,271.17|	-4,740.26|	

Formula:  
Y = 225,271.17-4,740.26*population  
|Y: House Value|	X: Population|
|--------------|-------------------|
|215,791|	2|
|211,050|	3|
|206,310|	4|  
	
![image](https://user-images.githubusercontent.com/101474477/184520957-234221bc-bc46-4e28-b176-b47810721a78.png)

**Total Rooms**  
Y Coefficients:  
|Intercept	|Rooms	|
|---------------|---------------|	
|225,271.17	|18,450.55	|

Formula:  
Y = 225,271.17+18,450.55*rooms  
|Y: House Value|	X: Rooms|
|--------------|-------------------|
|242,291	|2|
|250,802	|3|
|259,312	|4|  

![image](https://user-images.githubusercontent.com/101474477/184521132-0868cf70-2f1d-4f0d-909b-9b26b89372d8.png)

**Median Income**   
Y Coefficients:  
|Intercept	|ncome	|
|--------------|-------------------|	
|225,271.17	|9,658.27	|

Formula:  
Y = 225,271.17+79,658.27*income  
| House Value	| Income|
|--------------|-------------------|
|75,271	|50,000|
|300,271	|75,000|
|384,588	|100,000|

![image](https://user-images.githubusercontent.com/101474477/184521288-ebbefed6-9b62-4778-aaf9-4679397e519d.png)

**Median House Age**     
 Y Coefficients:  
|Intercept	|Age	|
|--------------|-------------------|
|225,271.17	|8,510.13|

Formula:  
Y = 225,271.17+8,510.13*age  
|Y: House Value	|X: House Age|
|--------------|-------------------|
|242,291	|2|
|250,802	|3|
|259,312	|4|

![image](https://user-images.githubusercontent.com/101474477/184521439-82fa77c2-6417-494c-9d2a-f5b5e56c4704.png)

**Total Households**
Y Coefficients:  
|ntercept	|Households	|
|--------------|-------------------|
|225,271.17	|8,010.86|

Formula:  
Y = 225,271.17+8,010.86*households  
|Y: House Value	|X: Households|
|--------------|-------------------|
|241,293	|2|
|249,304	|3|
|257,315	|4|

![image](https://user-images.githubusercontent.com/101474477/184521521-90dc48ce-e477-4b8d-8dab-49cf39a8720b.png)

**Total Bedrooms**  
Y Coefficients:    
|Intercept	|Households	|
|--------------|-------------------|
|225,271.17	|6,593.80|

Formula:  
Y = 225,271.17+6,593.80*bedrooms  
|Y: House Value	|X: Bedrooms|
|--------------|-------------------|
|238,459	|2|
|245,053	|3|
|251,646	|4|

![image](https://user-images.githubusercontent.com/101474477/184521601-27988feb-a9ff-4e40-ab11-d422763f5693.png)

Although rough, these plots help guide feature selection.

#### Outliers within the dataset 
One of the challenges of the original dataset lies with the variables which represent totals of individual features, such as total number of bedrooms, total number of households, and total number of bedrooms.  This last in particular presents issues with extreme outliers.  The range of this variable is 6 to slightly over 32,000.  Given that the base unit stipulated is a block, a density of 32,000 rooms would argue for large apartment complexes rather than individual houses.  

![image](https://user-images.githubusercontent.com/101474477/185263905-079dabb9-8919-4e8b-839a-b765c45f8799.png)

Total rooms numbering more than 10,000 comprise only about 1.2% of the total of all rooms. Ninety-five percent lie with the range of 6 to 6,000.  As a result, the data for this feature is very heavily skewed left, and the rough regression against median home values has an R-squared factor infinitely close to zero, and a very high MSE. Feature engineering using various approaches could assist in model accuracy. 
 
 </p>
</details>

<details><summary>Choosing the Model</summary>
<p>
The team agreed that a supervised machine learning model would be best suited for the data and objectives of the project.  We were using labelled data and were working with a relatively large dataset. For that reason, the team early in the process (concurrent with the data selection and topic selection discussions), determined that either the Random Forest Regressor or the Hist Gradient Boosting Regressor would be good candidates for the final model, since both have a relatively high degree of accuracy while being resistant to overfitting.
	
We ran both the Random Forest Regressor and the Hist Gradient Boosting Regressor.  In addition, the team decided to explore other models for comparison, so a Linear Regression model was added. As part of the comparison, the team wanted to examine the accuracy scores of the models, but also the feature importances.  Hist Gradient Boosting Regressor does not have a features importances function at this time, but Gradient Boosting Regressor does, so that was substituted for the Hist Gradient Boosting model.  The Linear Regression, Random Forest Regressor, and the Gradient Boosting Regressor are detailed below. 
	
 </p>
</details>

<details><summary>Analysis</summary>

### Final Production Preprocessing  
#### Load the finished production database:  
![image](https://user-images.githubusercontent.com/101474477/184995497-3e7763dc-1c92-4336-9098-cd312ce6d4bd.png)

####  Drop Unecessary Features 
##### Drop low value variables City, County, Longitude, latitude. Encode categorical variables using get.dummies
![image](https://user-images.githubusercontent.com/101474477/184996523-d1b12537-3853-4078-b8eb-f101f991e65c.png)
**Prepared Database**  
![image](https://user-images.githubusercontent.com/101474477/184996812-8a550527-459a-4a05-855b-d7f3cda93784.png)
	
##### Split into the features and target arrays:  
![image](https://user-images.githubusercontent.com/101474477/184996968-54028dda-fbd1-4302-908d-e9c84f18c80b.png)

##### Split the database to create and training and testing datasets:
![image](https://user-images.githubusercontent.com/101474477/184997177-b5f6f564-5a47-4b02-987f-8f8a57eacbd6.png)

The 70/30 split was in line with recommended practice.

#### Scale the Data  
![image](https://user-images.githubusercontent.com/101474477/184997484-659309a2-c226-4c42-bb16-06236578113a.png)

 </p>
</details>	
	
<details><summary>Linear Regression</summary>

<p>
	
#### Define and Fit the Model  

![image](https://user-images.githubusercontent.com/101474477/185000179-cd1a852a-1367-47ff-9fcd-5431916cc71a.png)  

##### Print the Model Intercept and Coefficient Values  
	
![image](https://user-images.githubusercontent.com/101474477/185000560-d1449412-12ca-4bb0-af12-ddbce6e126f3.png)

![image](https://user-images.githubusercontent.com/101474477/185000437-14b5b930-7eb9-4de8-82a4-3563e494a3ab.png)
	
![image](https://user-images.githubusercontent.com/101474477/185000767-42748512-5bf7-4afc-bbaa-f762f5a63e54.png)

![image](https://user-images.githubusercontent.com/101474477/185000856-c40bfefe-f857-403d-9d5a-c8d31c533138.png)

The mean squared error measures the prediction accuracy of a model, and is always 0 or positive. When the MSE is larger, this is an indication that the linear regression model is not accurately predicting the outcome. An important piece to note is that the MSE is sensitive to outliers, so the greater the number and magnitude of the outliers, the greater the deviation from the mean and the less accurate the model. Testing and training scores that are close together indicate minimal or no overfitting; however, a low accuracy score indicates weakness in the data.
	
</p>
</details>


<details><summary>Random Forest Regressor</summary>

<p>
	
#### Define and Fit the Model    
	
![image](https://user-images.githubusercontent.com/101474477/185002626-565c40ae-fbe1-48fa-9993-4a8acc927fea.png)

#### Print the Results  	 
![image](https://user-images.githubusercontent.com/101474477/185002754-46d792cd-83f8-4c32-acd3-f4c30ee55b70.png)

</p>
</details>


<details><summary>Gradient Boosting Regression</summary>

<p>
	
## Pre-Optimization      
#### Define and Fit the Model     
![image](https://user-images.githubusercontent.com/101474477/185002867-8c45f18e-d075-4b4e-8656-4df4dbbf64bd.png)

#### Print the Result  
![image](https://user-images.githubusercontent.com/101474477/185002957-7283e8ca-d054-40cb-9ff3-3e5057162cd0.png)

## Optimize the Model  
### n_estimators  
#### Define Parameters Function, Run the Optimization  
	
![image](https://user-images.githubusercontent.com/101474477/185018374-88388746-5308-44f5-9a79-8673095a84ba.png)

#### Extract Best Fit and Plot  	
![image](https://user-images.githubusercontent.com/101474477/185003594-608509d1-94e3-4e59-beb7-4fe87915822c.png)
![image](https://user-images.githubusercontent.com/101474477/185003845-d0ee3c40-d17d-4cfb-9cdc-195d8b57ae3c.png)
	
##### Summary  
The hyperparameter n_estimators indicates the total number of trees used in the model to arrive at the final result.  A higher number of trees provides better performance but increases the amount of the code processing time. 
	
The accuracy score over the training set increases continuously with the increase in the n_estimators up to a certain point. In this graph, the performance over the test set increases initially as n_estimators increases. After the n_estimators reaches 600, however, the accuracy score becomes stagnant. Which means that even if the value of n_estimators increases over 600, there is no more gain in the accuracy level of test set.  Subsequent increase in the n_estimators value will not add any value to the model; it will slow the model and likely result in overfitting.  

Optimized parameter n_estimators value: **600**

### max_depth  
#### Redefine the Model with Optimized n_estimator Value   

#### Define Parameters Function, Run the Optimization  
With the defined n_estimators of 600, the second optimization model is run for max_depth:  

![image](https://user-images.githubusercontent.com/101474477/185005105-7bbaf492-afdf-4726-99d0-3f1687c147cb.png)  

![image](https://user-images.githubusercontent.com/101474477/185005605-a5649a4f-df90-4182-b013-262ad582d8b8.png)

#### Extract Best Fit and Plot  
	
![image](https://user-images.githubusercontent.com/101474477/185005857-d8be51c1-5537-4ede-9799-723007a9d3b3.png)  

##### Summary    
max_depth indicates the maximum depth of trees in the model. It is defined as longest path between the root node and the leaf node. Using max_depth, we can control the depth we want every tree to grow. In the above graph, as the value of max depth increases, the performance of the model over the training set increases continuously and eventually achieves the 100 % accuracy score. 
	
However, as max_depth value increases, the performance over the test set increases up to a certain point after which it no longer increases model performance.  In this model, after max_depth value of 4, performance begins to decrease rapidly. At this stage, the tree starts to overfit the training set and therefore is not able to generalize over the unseen points in the test set.
	
Optimized parameter max_depth value: **4**  
	
### max_features  
#### Redefine the Model with Optimized n_estimator, max_depth Values  

#### Define Parameters Function, Run the Optimization  
With the defined n_estmators of 600 and max_depth of 4, the third optimization model is run for max_features:  

![image](https://user-images.githubusercontent.com/101474477/185007318-1c0bb3d4-69ae-4158-9092-db44adebd077.png)

#### Extract Best Fit and Plot  
![image](https://user-images.githubusercontent.com/101474477/185007692-45888353-3341-4597-aa8f-eb5e758126c4.png)  

![image](https://user-images.githubusercontent.com/101474477/185007727-5d4b3559-80db-4085-921a-c09d165e6daf.png)


##### Summary    
max_features simulates the number of maximum features provided to each tree in a model. The model chooses some random samples from the features to find the best split. In the above graph, as the value of max_features increases, the performance of the model over training set increases continuously up to the point where max_features is greater than 6.  At this stage, max_features is past optimal value and performance ceases to improve. The tree starts to overfit the training set and hence is not able to generalize over the unseen points in the test set.
	
Optimized parameter max_features value: **6**  
	
## Final Optimized Model      
#### Redefine the Model with Optimized n_estimators = 600, max_depth = 4, max_features = 6 

#### Define Parameters Function, Run the Optimization  
![image](https://user-images.githubusercontent.com/101474477/185017389-ea2f2700-27f7-48a8-b91f-3ba016788d35.png)

#### Specify, Plot Feature Importances  
	
![image](https://user-images.githubusercontent.com/101474477/185017585-7a36ffcc-3475-45cc-9358-acdf795c44cc.png)
![image](https://user-images.githubusercontent.com/101474477/185017893-d388510a-c7e7-45e0-9f47-3ec5f2e0176b.png)

</p>
</details>

<details><summary>Analysis Results</summary>

<p>
After running the three main models, Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor, Gradiebt Boosting Regressor emerged as the technique best suited for the model.  After optimization, Gadient Boosting had the highest accuracy score and the lowest mean errors of the three. 

![image](https://user-images.githubusercontent.com/101474477/185266907-13a7aa81-3edd-48c3-a315-8953e72a6d11.png)


![image](https://user-images.githubusercontent.com/101474477/185269230-59222b32-4a45-43f5-ad91-6be02ac15a75.png)
	
Gradient Boosting, as well as the other two, ranked median income as the top influencer with a weighted score of 40%.  In the Gradient Boosting model feature importances,  median income was followed by max temperature and the number of business establishments. Humidity, total rooms, population, and inland/ocean proximity also had a small impact on the housing prices. The number of employed people, wind speed, total bedrooms, amount of households, age and near ocean had minimal impact. All other features had a little to none weighted score on housing prices. 

![image](https://user-images.githubusercontent.com/101474477/185267168-af1408d1-a5fe-40a4-a34e-4624663205b4.png)

Linear regression is considered one of the base techniques for describing the relationshipAnother issue with regression trees is the number of significant variables and the number of nonsignificant variables in your data set. It is known that when you have few interesting input variables and a large number of noise variables the regression forests does not behave well. Boosting procedures does not have this behavior. There is a good reason for that. Regression forests produce more uninteresting trees which have the potential to move the learned structure away from the true underlying structure. For boosting this does not happen since at each iteration only the region of interests have large weight, so the already learned regions are affected less. The remedy would be to play with the number of variables selected on learning time.

	


</p>
</details>
<details><summary>Conclusion and Recommendation</summary>

<p>
As an initial recommendation, the team is in agreement that upleveling the number and quality of data sources would be an essential priority. .  The team had researched different sources for data related to climate data, crime data, and more economic indicators.  Such data would contribute to robust, portable predictive modelling for housing prices applicable 
	
	After More feature engineering would be beneficial to our model. 


</p>
</details>

####  Data Sources:

[Kaggle Dataset #1](https://www.kaggle.com/datasets/camnugent/california-housing-prices)

[Kaggle Dataset #2](https://www.kaggle.com/datasets/camnugent/california-housing-feature-engineering?select=cal_populations_city.csv)

[Census.gov](https://api.census.gov/data/1990/cbp?get=GEO_TTL,EMP,ESTAB&for=county:*&in=state:06&key=)

[OpenWeatherMap.org](http://api.openweathermap.org/data/2.5/weather?units=Imperial&APPID=)

#### Visualizations: 

[Dashboard](https://public.tableau.com/views/Housing_Estimator/Housing_Estimator?:language=en-US&:display_count=n&:origin=viz_share_link)

[Google Slide Presentation](https://docs.google.com/presentation/d/1T7_yxJK3ywl04BYXVCxGlF-N4pR6hri29zj-ifyfONc/edit#slide=id.p)
