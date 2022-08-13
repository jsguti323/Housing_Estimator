
## Housing Estimator




## Overview
Our final project creates a model to predict median house values in California based on the US government's Census data from the year 1990. This data includes house features (age of the house, number of rooms, number of bedrooms), and community characteristics (median income, number of households, and geographic location).  We added additional context features through weather API calls and county employment rates to expand the scope of the Census data and thus enlarge the pool of potential influential factors. The objective of this study is to identify which, if any, of the factors from our data features impact housing values, and to measure their influence. The ultimate determination of the model's success would be to apply it successfully to similar data outside of California.

Following the sequence of data preparation, database creation, feature engineering and selection, the team focused analysis on three machine learning models: Linear Regression, Random Forest Regressor, and Gradient Boosting Regression.  After optimization and comparison, the team concluded the optimized Gradient Boosting Regression to be the most successful model for the study. 

This summary presents the process the team undertook from topic and data selection, through database creation and modeling, to results and conclusion. In it, we detail the technologies we used, our decision-making process, the various iterations of models we considered, and finally lay out our recommendations for the data and the process.  The tabbed outline below follows this sequence.


<details><summary>Concept Development</summary>

<p>

 Selected topic
Reason the topic was selected
Description of the source of data
Questions the team hopes to answer with the data
Description of the data exploration phase of the project
Description of the analysis phase of the project
Technologies, languages, tools, and algorithms used throughout the project
Result of analysis
Recommendation for future analysis
Anything the team would have done differently
 

 ## Design Study    
The design study followed 5 main steps:  

- Identify the topic  
- Identify our data sources    
- Identify the question to be answered     
- Specify the target variable     
- Determine the model  
  

## Topic and Data Selection  
The topic was selected was housing value trends. Given most recent trends, the team felt that the topic was interesting and relevant, providing a rich opportunity in terms of available data and the broad array of features which can be modeled. 
 
 ### Criteria for Data Selection  
Because this project is based on a machine learning model, one of the most important characteristics was the size of the dataset.  Both overfitting and underfitting are primary concerns in machine learning modelling.  Having a large enough data set helps control for both those conditions. The team set the minimal standard of 10,000 rows to meet this requirement.   
	
In addition to the features above, the team researched availability of community crime statistics and economic indicators.  The deciding factor of whether to include more variables was based on accessibility of the data and on ease with which external data could be merged into the larger dataset. In the end, the team opted for a weather API call and census data on business establishments and number of employees.

## The Database  

### Description  
This size specification narrowed the number of sources suitable for analysis.   After some searching, the team identified the US Census 1990 dataset from Kaggle ((https://www.kaggle.com/datasets/camnugent/california-housing-prices,, (details below) as the main data source, augmented by other data sources. The California Housing Prices database is comprehensive, wide-ranging, saturated in geographic area, and includes geographical location coordinates which can link to a wide range of other data sources.  The external data for county employment figures were derived from census data (Census.gov) and weather from openweathermap.org, both called using APIs. The population information is the Kaggle California cities dataset. After cleaning, restructuring, refining and merging the individual datasets,  these four datasets became the production database and subsequently housed in AWS and connected in pgAdmin.

Component datasets: details
- Census.csv: 
    1990 Census data on communities 
    Selected features (3): 
        * counties 
        * Employees 
        * Establishments 
    Observations: 60

 - Housing.csv: 
    1999 Census data on housing in communities in California
    Data is gathered by block
    Features (11):
longitude	A measure of how far west a house is; a higher value is farther west
latitude	A measure of how far north a house is; a higher value is farther north
housingMedianAge	Median age of a house within a block; a lower number is a newer building
totalRooms	Total number of rooms within a block
totalBedrooms	Total number of bedrooms within a block
population	Total number of people residing within a block
households	Total number of households, a group of people residing within a home unit, for a block
medianIncome	Median income for households within a block of houses (measured in tens of thousands of US Dollars)
medianHouseValue	Median house value for households within a block (measured in US Dollars)
oceanProximity	Location of the house w.r.t ocean/sea
    Observations: 20,641

- Weather data:
    Weather for specific date called through weather API
    Features (5):
        * Max Temp
        * Humidity
        * Cloudiness
        * Wind Speed
        * Description
    Observations: 20,433 (after merge with cleaned housing dataset)

- Population data:
    Population information by county and city 
    Features (7):
        * County
        * City
        * Incorporation_date
        * pop_april_1980
        * pop_april_1990
        * pop_april_2000
        * pop_april_2010
    Observations: 455
	
	
## Limitations of the Data Set  
  the data reflects a single point in time, so the  characteristics relevant to house values cannot be observed over time within the data set
 
</p>
</details>


<details><summary>Data Preprocessing</summary>
 
 Stores static data for use during the project
Interfaces with the project in some format (e.g., scraping updates the database, or database connects to the model)
Includes at least two tables (or collections if using MongoDB)
Includes at least one join using the database language (not including any joins in Pandas)
Includes at least one connection string (using SQLAlchemy or PyMongo)
<p>

</p>
</details>

<details><summary>Database Creation and Integration</summary>
 
 Stores static data for use during the project
Interfaces with the project in some format (e.g., scraping updates the database, or database connects to the model)
Includes at least two tables (or collections if using MongoDB)
Includes at least one join using the database language (not including any joins in Pandas)
Includes at least one connection string (using SQLAlchemy or PyMongo)
<p>

</p>
</details>

 the data reflects a single point in time, so the  characteristics relevant to house values cannot be observed over time within the data set
 

 Description of data preprocessing
Description of feature engineering and the feature selection, including the team's decision-making process
Description of how data was split into training and testing sets
Explanation of model choice, including limitations and benefits
Explanation of changes in model choice (if changes occurred between the Segment 2 and Segment 3 deliverables)
Description of how the model was trained (or retrained if the team used an existing model)
Description and explanation of model's confusion matrix, including final accuracy score
Additionally, the model obviously addresses the question or problem the team is solving.


<details><summary>Data Exploration</summary>

<p>

 </p>
</details>

<details><summary>Choosing the Model</summary>

<p>
For that same reason, the team early in the process (concurrent with the data selection and topic selection discussions), determined that either the Random Forest Regressor a
or the Hist Gradient Boosting Regressor would be good candidates for the final model, since both have a relatively high degree of accuracy while being resistant to overfitting.
 </p>
</details>

<details><summary>Preprocessing</summary>

<p>

 </p>
</details>

<details><summary>Tehnologies</summary>

<p>

</p>
</details>

	
Structuring and Cleaning 
Creating common columns to link the datasets was the first step.  The housing file did not include any city names, only the geo coordinates.  The other datasets were identified by city and county.  The initial transformation added the specific city and county names to the housing dataset by using city.py and the location coordinates to list and append each city name to the housing set. 

 
Original dataset

 
Modified dataset

Create Table Structure pgAdmin
The main database was structured according to this ERD:

 


Weather, population, and census were joined into the main dataset, clean_merged_data.csv.

Output database: clean_merged_data.csv
	Observations: 11,454

</p>
</details>


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

Enter infor here

</p>
</details>

<details><summary>Visualizations</summary>

<p>

![image](https://user-images.githubusercontent.com/98067116/183781913-c398ffbe-97f8-47a7-910e-74ae0a09246c.png)

</p>
</details>

<details><summary>Results</summary>

<p>

If editing, insert text here

</p>
</details>



<details><summary>Recommendations</summary>

<p>

If editing, insert text here


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
