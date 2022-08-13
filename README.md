
## Housing Estimator




## Overview
Our final project creates a model to predict median house values in California based on the 1990 Census data. This data includes house features (age of the house, number of rooms, number of bedrooms, location), and community characteristics (median income, latitude/longitude, and proximity to ocean).  We added additional context features through weather API calls and county employment rates to augment the scope of the Census data. 

After preliminary review and testing, the team focused analysis on three machine learning models: Linear Regression, Random Forest Regressor, and Gradient Boosting Regression.  After optimization and comparison, the team concluded the optimized Gradient Boosting Regression to be the most successful model for the study.

This summary 


<details><summary>Concept Development</summary>

<p>

 ## Design Study    
The design study followed 5 main steps:  

- Identify topic
    * Housing value trends
- Identify data sources  
    * Census data
    * Kaggle
- Identify question to be answered   
    * what factors influence home values
- Identify target variable  
    * median home values
- Identify model  
    * Gradient Boosting Regressor 

## Topic and Data Selection 
Given most recent houing price trends, the team felt that the topic was interesting and relevant, providing a rich opportunity in terms of available data and the broad array of features which can be modeled.  In The team considered multiple factors to include in the study. In addition to the features above, the team researched availability of community crime statistics and economic indicators.  The deciding factor of whether to include more variables was based on the ease with which external data could be merged into the larger dataset.  To expedite the model development, the team decided to streamline the study process and focus on fewer variables.  The initial geographical range was decided to be California.
 
We will store our database on AWS. Our communication protocols include meeting twice a week via Google Meet on Monday and Wednesday before class, as well as additional meetings later in the week if needed. 


</p>
</details>

<details><summary>Tehnologies</summary>

<p>

</p>
</details>



 

<details><summary>Linear Regression</summary>

<p>

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
