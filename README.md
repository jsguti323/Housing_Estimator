
## Housing Estimator




## Overview
Our final project is a study of factors infuencing median house values in California based on the 1990 Census. The Census data comprised house features (age of the house, number of rooms, number of bedrooms, location), and community characteristics (median income, latitude/longitude, and proximity to ocean).  Additional context features were added through weather API calls and county employment rates to establish a more robust result. 

After preliminary review and testing, the team focused analysis on three machine learning models: Linear Regression, Random Forest Regressor, and Gradient Boosting Regression.  After optimization and comparison, the team concluded the optimized Gradient Boosting Regression is the most successful model for the study.

We will store our database on AWS. Our communication protocols include meeting twice a week via Google Meet on Monday and Wednesday before class, as well as additional meetings later in the week if needed. 

<details><summary>Concept Development</summary>

<p>

![image](https://user-images.githubusercontent.com/98067116/183781387-3861e7ee-cdc1-43a3-90e9-358335a26505.png)

</p>
</details>


Design Study
Identify question to be answered: which factors influence home values
Identify target variable: home values
Identify model: RandomForestRegressor, HistGradientBoostingRegressor (secondary)

Data Selection Process
The team considered multiple factors to include in the study. In addition to the features above, the team researched availability of community crime statistics and economic indicators.  The deciding factor of whether to include more variables was based on the ease with which external data could be merged into the larger dataset.  To expedite the model development, the team decided to streamline the study process and focus on fewer variables.  The initial geographical range was decided to be California.


 

<details><summary>Linear Regression</summary>

<p>

![image](https://user-images.githubusercontent.com/98067116/183781387-3861e7ee-cdc1-43a3-90e9-358335a26505.png)

</p>
</details>


<details><summary>Random Forest Regressor</summary>

<p>

![image](https://user-images.githubusercontent.com/98067116/183781529-f9f2a22e-3dfc-4e6b-b475-868d1e2ed469.png)

</p>
</details>




<details><summary>Hist Gradient Boosting Regressor</summary>

<p>

![image](https://user-images.githubusercontent.com/98067116/183781742-f6ec4f46-e6c0-4943-988e-6b7c263a8cd1.png)



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
