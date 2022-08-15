
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
The topic was selected was housing value trends. Given most recent trends, the team felt that the topic was interesting and relevant, providing a rich opportunity in terms of available data and the broad array of features which can be modeled.  

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
    1999 Census data on housing in communities in California    
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
<details><summary>Architect Production Database </summary>

## Structuring and Cleaning   
Data preparation began with creating a preliminary data structure usng Pandas to merge and join the individual datasets. Creating common columns to link the datasets was the first step.  The housing file did not include any city names, only the geographic coordinates.  The other datasets were identified by city and county.  The initial transformation added the specific city and county names to the housing dataset by using city.py and the location coordinates to list and append each city name to the housing set. 

 ### Census Data  
 #### Starting URL for Census Data API Call.  
![image](https://user-images.githubusercontent.com/101474477/184716368-41dfe441-b1c7-48cb-b852-4c05a77726e4.png)

**Input Dataset**  

 ![image](https://user-images.githubusercontent.com/101474477/184517692-656ea19d-258b-459f-b8a4-61af6fb7cde9.png)

#### Cleaning and Manipulation ####  
![image](https://user-images.githubusercontent.com/101474477/184716537-6b0d0bac-c97d-4770-9bbf-b14b7c5f3840.png)
![image](https://user-images.githubusercontent.com/101474477/184716638-50affa15-2c87-431a-bde6-f3069e419f21.png)
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
	

#### Prepare the location coordinates data for processing. Use citypy to join city name to geographical coordinates 
	
![image](https://user-images.githubusercontent.com/101474477/184718425-86e23171-cff3-4e6a-ae84-450f3a2f983a.png)
	

#### Initiate API call
![image](https://user-images.githubusercontent.com/101474477/184718769-5416282c-3fba-4c2d-bf84-28a11deee29f.png)
	


#### Parse the JSON and retrieve data.  
	
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

## Creating the Table Structure in pgAdmin
	
The team decided to use AWS as the static data repository and use pgAdmin to create the production database. The tables were created in pgAdmin first following the schema:   

![image](https://user-images.githubusercontent.com/101474477/184518914-16ad6780-6e8e-4954-bbc8-e16e3c47df27.png)  
	
The static datasets were then called into pgAdmin through Spark.
	
![image](https://user-images.githubusercontent.com/101474477/184520065-39833e33-0322-4be6-8203-f0e55a328a42.png)

Weather, population, and census were joined into the main dataset, clean_merged_data.csv.  After being instantiated and joined, the final database was saved to a .csv file and read into Pandas for final data preparation and modelling.

**Output database: clean_merged_data.csv**
	**Observations: 11,454**

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
	
- sklearn.preprocessing StandardScaler
- sklearn.model_selection train_test_split
- R

Plotting and Visualization  
- matplotlib.pyplot 
- seaborn 
- dabl
	
</p>
</details>

<details><summary>Data Exploration</summary>

<p>

Concurrent with data cleaning and structuring, the team conducted preliminary data analysis to get a feel for the data itself.  This took the form of histogram and rough regression on the database elements. The objective is to determine whether the data has a normal distribution, measuring skew and kurtosis.  Rgression relies on normal distribution for accuracy; outliers reduce accuracy. 
	
Methods to address this in other code variations included normalizing skew 
	
#### Histograms  
![image](https://user-images.githubusercontent.com/101474477/184519340-7aeb165e-fb85-45ae-a81f-c69508b42a65.png)

![image](https://user-images.githubusercontent.com/101474477/184519385-9baa560a-2d00-4646-98e3-dbda83034041.png)

Few of the variables plotted had normal distributions. Households heavily skew left as do population, total rooms, and total bedrooms. Median house age, median income, and median house value are more symmetrically distributed as are maximum temperature, humidity and wind speed. 
	
![image](https://user-images.githubusercontent.com/101474477/184695024-12fcfc7b-20b4-4be3-80a8-1bf0f035c7d0.png)

![image](https://user-images.githubusercontent.com/101474477/184695309-995e8ff9-9908-4da3-8394-119a6da1cff4.png)
	
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

 </p>
</details>

<details><summary>Choosing the Model</summary>
<p>
The team agreed that a supervised machine learning model would be best suited for the data and objectives of the project.  We were using labelled data and were working with a relatively large dataset. For that reason, the team early in the process (concurrent with the data selection and topic selection discussions), determined that either the Random Forest Regressor or the Hist Gradient Boosting Regressor would be good candidates for the final model, since both have a relatively high degree of accuracy while being resistant to overfitting.
	
We ran both the Random Forest Regressor and the Hist Gradient Boosting Regressor.  In addition, the team decided to explore other models for comparison, so a Linear Regression model was added. As part of the comparison, the team wanted to examine the accuracy scores of the models, but also the feature importances.  Hist Gradient Boosting Regressor does not have a features importances function at this time, but Gradient Boosting Regressor does, so that was substituted for the Hist Gradient Boosting model.  The Linear Regression, Random Forest Regressor, and the Gradient Boosting Regressor are detailed below. 
	
 </p>
</details>

<details><summary>Analysis</summary>

### Production Preprocessing  
After loading and reading the database into Pandas for the actual modelling and analysis, the final preprocessing took place.  
The low value or noisy variables City, County, Longitude, latitude were dropped, and categorical variables, such as Ocean Proximity and weather description were converted to numeric values using get.dummies.
	
Then, the preprocessed data was split into the features and target arrays:
X = housing_df.drop(columns = ["median_house_value"])
y = housing_df['median_house_value']
	
and the training and testing datasets were created:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=1/3)

The 70/30 split was in line with recommended practice.
	
<details><summary>Random Forest Regressor</summary>

<p>



</p>
</details>




<details><summary>Linear Regression</summary>

<p>
The mean squared error is a common way to measure the prediction accuracy of a model.
The mean squared error is always 0 or positive. When a MSE is larger, this is an indication that the linear regression model doesn’t accurately predict the model.

An important piece to note is that the MSE is sensitive to outliers. 	
No overfitting as training and testing scores are very close to each other, though accuracy is poor
	
</p>
</details>


<details><summary>Gradient Boosting Regression</summary>

<p>



</p>
</details>

<details><summary>Results and Recommendations</summary>

<p>

</p>
</details>

<details><summary>Conclusion</summary>

<p>
	
![image](https://user-images.githubusercontent.com/98067116/183781913-c398ffbe-97f8-47a7-910e-74ae0a09246c.png)

</p>
</details>

<details><summary>Results</summary>
Result of analysis  
Recommendation for future analysis  
Anything the team would have done differently  
<p>


</p>
</details>



<details><summary>Recommendations</summary>

<p>




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
