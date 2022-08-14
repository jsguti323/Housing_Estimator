
## Housing Estimator




## Overview
Our final project creates a model to predict median house values in California based on the US government's Census data from the year 1990. The objective of this study is to identify which, if any, of the factors from our data features impact housing values, and to measure their influence. The ultimate determination of the model's effectiveness would be to replicate our result with similar data outside of California.

Following the sequence of data preparation, database creation, feature engineering and selection, the team focused analysis on three machine learning models: Linear Regression, Random Forest Regressor, and Gradient Boosting Regression.  After optimization and comparison, the team concluded the optimized Gradient Boosting Regression to be the most successful model for the study. 

This summary presents the process the team undertook from topic and data selection, through database creation and modeling, to results and conclusion. In it, we detail the technologies we used, our decision-making process, the various iterations of models we considered, and finally lay out our recommendations for the data and the process.  The tabbed outline below follows this sequence.


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

We selected the California Housing Prices database from Kaggle (https://www.kaggle.com/datasets/camnugent/california-housing-prices, details below), median house prices for California districts derived from the 1990 census. This dataset originated S&P Letters Data abnd We collected information on the variables using all the block groups in California from the 1990 Cens us. In this sample a block group on average includes 1425.5 individuals living in a geographically co mpact area. Naturally, the geographical area included varies inversely with the population density. W e computed distances among the centroids of each block group as measured in latitude and longitude. W e excluded all the block groups reporting zero entries for the independent and dependent variables. T he final data contained 20,640 observations on 9 variables. The dependent variable is ln(median house value).

                               Bols    tols
INTERCEPT		       11.4939 275.7518
MEDIAN INCOME	       0.4790  45.7768
MEDIAN INCOME2	       -0.0166 -9.4841
MEDIAN INCOME3	       -0.0002 -1.9157
ln(MEDIAN AGE)	       0.1570  33.6123
ln(TOTAL ROOMS/ POPULATION)    -0.8582 -56.1280
ln(BEDROOMS/ POPULATION)       0.8043  38.0685
ln(POPULATION/ HOUSEHOLDS)     -0.4077 -20.8762
ln(HOUSEHOLDS)	       0.0477  13.0792The Census data  data includes house features (age of the house, number of rooms, number of bedrooms), and community characteristics (median income, number of households, and geographic location).  We added additional context features through weather API calls and county employment rates to expand the scope of the Census data and enlarge the pool of potential influential factors.  In addition to the features above, the team researched availability of community crime statistics and economic indicators.  The deciding factor of whether to include more variables was based on accessibility of the data and on ease with which external data could be merged into the larger dataset. In the end, the team opted for a weather API call and census data on business establishments and number of employees.  

## The Database    

### Description  
Because this project is based on a machine learning model, one of the most important characteristics was the size of the dataset.  Both overfitting and underfitting are primary concerns in machine learning modelling.  Having a large enough data set helps control for both those conditions. The team set the minimal standard of 10,000 rows to meet this requirement.  This size specification narrowed the number of sources suitable for analysis.   After some searching, the team identified the US Census 1990 dataset from Kaggle ( as the main data source, augmented by other data sources. The California Housing Prices database is comprehensive, wide-ranging, saturated in geographic area, and includes geographical location coordinates which can link to a wide range of other data sources.  The external data for county employment figures were derived from census data (Census.gov) and weather from openweathermap.org, both called using APIs. The population information is the Kaggle California cities dataset. After cleaning, restructuring, refining and merging the individual datasets,  these four datasets became the production database and subsequently housed in AWS and connected in pgAdmin.

#### Component datasets: details
- **Census.csv**: 
    1990 Census data on communities   
    Selected features (3):   
    - counties 
    - Employees 
    - Establishments 
    Observations: 60

 - **Housing.csv**:  
    1999 Census data on housing in communities in California    
    Data is gathered by block  
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
  While detailed within the features offered, this dataset had some limitations:  
  
    - the data is gathered by block; however, the data varies by unit   
        - 3 features are median values:  
	    - age of the houses per block;        
	    - income of the population per block; and,       
	    - value of the houses per block      
	- 4 are totals of the represented features within a block: 
	    - number of rooms     
	    - number of bedrooms      
	    - number of people    
            - number of households      

Scaling the data brings the input data points closer together; however, it is more difficult to get a good understanding of what the data is actually saying in interpreting results. For example, it is difficut to properly weight total number of rooms on a block as part of individual house values. If the data set had included the number of houses or dwellings being counted in the per-block reference frame, we could have created calculated features to include in the analysis, such as median number of rooms per house or median number of residents.  
	
Lastly, the data reflects a single point in time, so the  characteristics relevant to house values cannot be observed over time rendering it relatively static. With time data, it would be possible to see how impacts change with the changes in the values of the features themselves, and thus get a more accurate undersanding of true trends.  	
 
</p>
</details>

<p>
<details><summary>Data Preprocessing</summary>

## Structuring and Cleaning   
Data preparation began with creating a preliminary data structure usng Pandas to merge and join the individual datasets. Creating common columns to link the datasets was the first step.  The housing file did not include any city names, only the geographic coordinates.  The other datasets were identified by city and county.  The initial transformation added the specific city and county names to the housing dataset by using city.py and the location coordinates to list and append each city name to the housing set. 

 ### Census Data  
 #### Starting URL for Census Data API Call.  
url = "https://api.census.gov/data/1990/cbp?get=GEO_TTL,EMP,ESTAB&for=county:*&in=state:06&key=" + census_api_key   
census = requests.get(url).json()  
df = pd.DataFrame(census)  

**Input Dataset**  

 ![image](https://user-images.githubusercontent.com/101474477/184517692-656ea19d-258b-459f-b8a4-61af6fb7cde9.png)

Cleaning and Manipulation:  
new_columns = ['County', 'Employees', 'Establishments', 'State', 'County Code']
df['County'] = df['County'].map(lambda x: x.rstrip(" County, CA")
df = df.drop(columns=['State', 'County Code'])  

**Output Dataset**  

![image](https://user-images.githubusercontent.com/101474477/184517942-b7e7fd2d-e4c3-458a-8407-3788593f9d64.png)


### Population Data   

file = '../Data/cal_populations_city.csv'  
	
**Input Dataset** 
	
![image](https://user-images.githubusercontent.com/101474477/184518484-faac1560-0ac1-417b-9197-56e92bf57d7c.png)

 
**Output Dataset** 
	
df = df.drop(columns=['Incorportation_date', 'pop_april_1980', 'pop_april_2000', 'pop_april_2010'])  

![image](https://user-images.githubusercontent.com/101474477/184518591-dcf3d531-b956-4e49-9029-66b6bc6b5a35.png)

### Weather Data  	

-Use citypy to join city name to geographical coordinates    
	city = citipy.nearest_city(coordinate[0], coordinate[1]).city_name  

#### Parse the JSON and retrieve data.  
        city_weather = requests.get(city_url).json()  
        #### Parse out the needed data.     
        city_max_temp = city_weather["main"]["temp_max"]  
        city_humidity = city_weather["main"]["humidity"]  
        city_clouds = city_weather["clouds"]["all"]  
        city_wind = city_weather["wind"]["speed"]  
        city_description = city_weather["weather"][0]["description"]  
        #### Append the city information into city_data list.    
        city_data.append({"City": city.title(),    
                          "Max Temp": city_max_temp,  
                          "Humidity": city_humidity,  
                          "Cloudiness": city_clouds,  
                          "Wind Speed": city_wind,  
                          "Description": city_description})  

**Output Dataset**
	
![image](https://user-images.githubusercontent.com/101474477/184518678-260be8a9-4737-423c-b278-c5f38937b350.png)

### Final Dataset

**Input Dataset**
	
![image](https://user-images.githubusercontent.com/101474477/184518831-d28b4d60-2a12-4dfb-ae52-c579e0013152.png)

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
API calls
matplotlib.pyplot 
collections Counter
seaborn 
Python 
Jupyter Notebook
R and R Studio
Pandas, numpy, citypy
Mlenv environment
Linear Regression
GradientBoostingRegressor
Random Forest Regressor
Database Integration:
AWS Relational Database System
pgAdmin
prosgresSQL

sklearn.ensemble RandomForestRegressor
sklearn.datasets make_regression
sklearn.ensemble HistGradientBoostingRegressor
sklearn.preprocessing LabelEncoder
sklearn metrics

sklearn.ensemble RandomForestRegressor
sklearn.preprocessing StandardScaler
sklearn.model_selection train_test_split
sklearn.metrics accuracy_score, classification_report
</p>
</details>

<details><summary>Data Exploration</summary>

<p>

Concurrent with data cleaning and structuring, the team conducted preliminary data analysis to get a feel for the data itself.  This took the form of histogram and rough regression on the database elements.

#### Histograms  
![image](https://user-images.githubusercontent.com/101474477/184519340-7aeb165e-fb85-45ae-a81f-c69508b42a65.png)

![image](https://user-images.githubusercontent.com/101474477/184519385-9baa560a-2d00-4646-98e3-dbda83034041.png)

Few of the variables plotted had normal distributions. Households heavily skew left as do population, total rooms, and total bedrooms. Median house age, median income, and median house value are more symmetrically distributed as are maximum temperature, humidity and wind speed. 

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
	
We ran both the Random Forest Regressor and the Hist Gradient Boosting Regressor.  In addition, the team decided to explore other models for comparison, so a Linear Regression model was added. As part of the comparison, the team wanted to examine the accuracy scores of the models, but also the feature importances.  Hist Gradient Boosting Regressor does not have a features importances function at this time, but Gradient Boosting Regressor does, so that was substituted for the Hist Gradient Boosting model.  Theses models, the processes, and the results are presented separately below.
	
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




<details><summary>Hist Gradient Boosting Regressor</summary>

<p>

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
