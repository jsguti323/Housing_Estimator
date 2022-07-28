SELECT h."City", h."Longitude", h."Lattitude" as Latitude,h."Population",
h."Median Age" As Median_Age ,
h."Median Income" As Median_Income,
h."Median House Value" As Median_House_Value,
h."Total Rooms" As Total_Rooms,
h."Bedrooms",h."Households",
h."Ocean Proximity" As Ocean_Proximity,
w."Max Temp" As Max_Temp,
w."Humidity",w."Cloudiness",
w."Wind Speed" As Wind_Speed,
w."Description"
INTO Housing_Weather
From Housing as h
INNER JOIN Weather as w
ON h."City" = w."City"




SELECT c."County",c."Employees",c."Establishments",
p."City",p."Population"
INTO Population_Census
FROM Census AS c
INNER JOIN Population As p
on c."County" = p."County"


SELECT h."City", h."Longitude", h."latitude",h."Population",
h."median_age" ,
h."median_income",
h."median_house_value",
h."total_rooms",
h."Bedrooms",h."Households",
h."ocean_proximity",
h."max_temp",
h."Humidity",h."Cloudiness",
h."wind_speed",
h."Description", c."County",c."Employees",c."Establishments" 
INTO Merged_data
FROM Housing_Weather as h
LEFT JOIN Population_Census as c
on h."City" = c."City"

SELECT h."City", h."Longitude", h."latitude",h."Population",
h."median_age" ,
h."median_income",
h."median_house_value",
h."total_rooms",
h."Bedrooms",h."Households",
h."ocean_proximity",
h."max_temp",
h."Humidity",h."Cloudiness",
h."wind_speed",
h."Description", c."County",c."Employees",c."Establishments" 
INTO Merged_data
FROM Housing_Weather as h
LEFT JOIN Population_Census as c
on h."City" = c."City"