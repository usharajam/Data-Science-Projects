#!/usr/bin/env python
# coding: utf-8

# <div style="border:solid green 2px; padding: 20px">
# <b>Hello Sharmi!</b>
# 
# My name is Evgeniy D. I'm going to review your project!
# 
# My main goal is not to show that any mistakes have been made somewhere, but to share my experience that will help you in your further work. Further in the file you can see my comments, try to take them into account when performing the next tasks.
# 
# 
# You can find my comments in <font color='green'>green</font>, <font color='blue'>blue</font> or <font color='orange'>orange</font> boxes like this:
# 
# <div class="alert alert-success">
# <b>Success:</b> if everything is done succesfully
# </div>
# 
# <div class="alert alert-warning">
# <b>Remarks: </b> if I can give some recommendations
# </div>
# 
# <div class="alert alert-danger">
# <b>Needs fixing:</b> if the block requires some corrections. Work can't be accepted with the red comments.
# </div>
# 
# 
# If you have 3 orange comments, we will need to adjust the project.
#     
# Let's work on the project in dialogue: if you change something in the project or respond to my comments, write about it. It will be easier for me to track the changes if you highlight your comments:
#     
# <div class="alert alert-info"> <b>Student comment:</b> For example like this.</div>

# # Research on car sales ads
# 
# You're an analyst at Crankshaft List. Hundreds of free advertisements for vehicles are published on your site every day.
# 
# You need to study data collected over the last few years and determine which factors influence the price of a vehicle. 

# ## Open the data file and study the general information. 
# 
# Read the given dataset 'vehicles_us.csv' into pandas dataframe using `read_csv()` method in pandas. Let us use `try` and `except` block to prevent any file I/O errors. `info()` and `head()` methods of the dataframe helps us to explore the dataset.

# <div class="alert alert-success"> <b>Reviewer comment:</b> 
#     
# Getting started is very important. So you explain what it is dedicated to. 
#     
# </div>

# In[313]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import warnings

try:
    car_ads_data = pd.read_csv('vehicles_us.csv')
except:
     car_ads_data =pd.read_csv('/datasets/vehicles_us.csv')
display(car_ads_data.head())
display(car_ads_data.info())


# <div class="alert alert-success"> <b>Reviewer comment:</b> 
#     
# Great and safety way for read our datasets.
#     
# </div>

# ### Conclusion

# There are 51525 records in this dataset. A First glimpse at the columns through `info()`, the columns model_year, cylinders, odometer, paint_color, is_4wd have missing values. Also, when we look at the sample records, year and odometer reading are in floating point values which in real world are discrete nature. Also, if we look at the column `is_wd` it looks like a flag to represent if the vehicle is 4 wheel drive but in the table it is stored as continuous value. So, it there is a  need to change this data type. Thus, at the first glimpse of data we found that the data is not clean and we need to preprocess them before analysis.

# ## Data preprocessing
# 
# We will preprocess the dataset, by first finding and replacing missing values and then analyzing the data types of columns and changing them if required.

# ### Replacing missing values
# 
# First let us find out  columns that has missing values. For this, we use `isnull()` to identify null entries and then `sum()` them to get the number of missing entries in each column in our dataframe `car_ads_data`

# In[314]:


car_ads_data.isnull().sum()


# Thus we have model_year with 3619, cylinders with 5260, odometer with 7892, paint_color with 9267, is4wd with 25953 missing entries. Next let us see the share of them against the total number of car ads.

# In[315]:


missing_entries = car_ads_data.isnull().sum().sort_values(ascending=False)
print("Percentage of Missing entries")
display(missing_entries/len(car_ads_data) * 100)


# <div class="alert alert-success"> <b>Reviewer comment:</b> 
#     
# It's great that we also look at the fraction of missing values. I left an example below, how it can be calculated yet.
#     
# </div>

# In[316]:


car_ads_data.isna().mean().reset_index()


# <div class="alert alert-info"> <b>Student comment:</b> This looks cool. Thanks for sharing, Such a helpful tip! </div>

# <div class="alert alert-success"> <b>Reviewer comment (2):</b> 
#     
# Thanks 😊
#     
# </div>

# From the above results, we can observe that the column is_4wd has more than 50% missing entries - This means that we have this data for only half of the entires. Planning to drop the records that have missing entries would be a bad idea as this will end up in losing half of the dataset. So, filling the missing values in this column becomes our utmost priority.
# 
# Now, let us dive deeper into the data is_wd holds. At the initial glimpse at this column, we observed this column looked like flag to mention if the car has 4 wheel drive but with continuous values. Let us see all possible values for this column using `value_counts()`.

# In[317]:


car_ads_data['is_4wd'].value_counts()


# The column is_4wd has only one value, 1.0. Rest of the records are missing entries. Thus we infer that for the vehicles that are four wheel drive, is_4wd has 1.0. Rest of the entries which are not four wheel drive has no values.  Instead of null values for those entries, we can fill them with 0 to represent they are not four wheel drive.

# In[318]:


car_ads_data['is_4wd'] = car_ads_data['is_4wd'].fillna(0)
display(car_ads_data)


# <div class="alert alert-warning"> <b>Reviewer comment:</b> 
#     
# <s>I recommend don't use attribute `inplace`, because it will be deleted at next upd in the packet `pandas`.
# </div>

# <div class="alert alert-info"> <b>Student comment:</b> Removed the inplace from the above code as recommended. I heavily use `inplace`. Thanks for pointing out. </div>

# Thus we replaced 50% of missing values in the entire dataset. The second column that has highest missing values is the paint_color. This column represents color of the vehicle. Let us see what unique values these column holds. 

# In[319]:


print("List of unique paint_color values:")
display(car_ads_data['paint_color'].unique())
print("Number of vehicles for each unique paint_color:")
display(car_ads_data['paint_color'].value_counts(normalize=True, dropna=False))


# As NaN tops the list except white, significant number of vehicle entries have missing values for paint_color. Dropping the records with the missing entries in this column will affect our analysis, so it is better replace NaN with a meaningful value. Since the data type is string object we can indicate the missing values with a meaningful text as 'unknown'. Using `fillna` we can achieve this as below,

# In[320]:


car_ads_data['paint_color'].fillna('unknown', inplace=True)
display(car_ads_data)


# <div class="alert alert-success"> <b>Reviewer comment:</b> 
#     
# Including the `fillna` and` astype` methods can be applied to multiple columns at the same time.
#     
# If there are more than 2 columns, then I recommend using the `for` loop.
#     
# Example:
#     
# `df.astype ({" col1 ":" float64 "," col2 ":" int64 "})`
#     
# </div>

# Next highest column with missing values is the 'odometer' column. This column holds the value of odometer reading which describes total miles/kms the vehicle is driven with. 

# In[321]:


car_ads_data[car_ads_data['odometer'].isnull()]


# Now, let us see how can we fill these missing odometer values. Using `corr()` let us find how odometer is related to other entries in the ads table.

# In[322]:


car_ads_data.corr()


# Thus, odometer has significant negative correlation to model_year and price with -0.47 and -0.42 respectively. Higher  the model_year in other words most recent the vehicle is, lower is the odometer. This matches the real world scenario. Also, higher the price, lower the odometer which mimics the real world car sales. Cars that are driven less are priced higher.  
# 
# We also note that the odometer reading is independent of other variables like model, cylinders, paint, days listed etc., which is reflected in the correlation matrix listed above. 
# 
# As odometer is correlated with price and year(model_year), let us see if we could fill the missing values based on these columns. Odometer readings are solely based on driven miles, whether it is expensive luxuary vehicle or a vehicle with basic features. Thus, for calculating missing odometer readings, how long the car is driven makes more sense than how much priced the car is. We can do this by grouping cars by model_year and find the their median odometer readings. There is high chance that the mean of odometer readings, is skewed towards upper extreme or lower extreme and so we will take the median of those values.
# 
# Before we choose to fill missing values of odometer using model_year, we need to find the missing model_year entries and fill them first. As we can see in the correlation matrix above, apart from odometer, model_year is also correlated to price. Car prices depriciate as it gets older, which is why the data shows the correlation between model_year and price. 
# 
# As the model_year is positively correlated the price, to fill the row of car ad with missing model_year we can find model year of the similar model whose price is closest to the price in this missing entry row. As price is continuous variable, we can group them to make it easier for comparison. To group the prices first we need to analyze the price distribution. We will use `value_counts()` to get an idea about price distribution.

# <div class="alert alert-success"> <b>Reviewer comment:</b> 
#     
# By the way we can using the `heatmap` method from the` seaborn` library, for visualization our matrix of correlation.
#     
# </div>
# 

# <div class="alert alert-info"> <b>Student comment:</b> Thanks for sharing your thoughts.  Since this is our first project on EDA and  our tutor suggested using matplotlib - to go from basics and try what we learnt in the study material. I will definitely use seaborn in the coming projects.</div>

# <div class="alert alert-success"> <b>Reviewer comment (2):</b> 
#     
# Agreed 😊
#     
# </div>

# In[323]:


# price distribution
display(car_ads_data['price'].value_counts(bins=10))


# As the most prices fall in one bin, let us filter rows with prices < 40000 and plot them in a histogram to visualize the price distribution. 

# In[324]:


# As most prices fall in one bin, we can filter rows with price < 40000 and see the distribution 
display(car_ads_data.query('price < 40000').hist('price', bins=10, range=(0,40000)))


# <div class="alert alert-success"> <b>Reviewer comment:</b> 
#     
# 
# It's great when the graph has title 😊👍
#     
# 
#     
# </div>
# 

# Based on the above information, it is known that, 98% of car prices falls within 37500. Thus to group the prices we need to divide the bin as follows.
# ```
# 0 - 3000 -  1  cheap
# 3001 - 4500 - 2 very low priced
# 4501 - 6000 - 3 low priced
# 6001 - 8000 - 4  average
# 8001 - 10000 - 5 above average
# 10001 - 13000 -6 high
# 13001 - 17000 - 7 very high
# 17001 - 24000 - 8 Expensive
# 24000 and more - 9 Highly Expensive
# ```
# 
# Now, to assign each price to a group let us write a function `assign_price_group()`. This function takes a price returns the group it belongs, based on the above price range criteria. Using `apply()` method of the price series, we can determine the price group of all the prices in the dataset and assign them to a new 'price_group' column in the dataset.

# In[325]:



# function to determine and return the price group of the given price
def assign_price_group(price):

    if price <= 3000:
        return 1
    if price <= 4500:
        return 2
    if price <= 6000:
        return 3
    if price <= 8000:
        return 4
    if price <= 10000:
        return 5
    if price <= 13000:
        return 6
    if price <= 17000:
        return 7
    if price <= 24000:
        return 8
    if price > 24000:
        return 9
    return 0
        
car_ads_data['price_group'] = car_ads_data['price'].apply(assign_price_group)
# view vehicle ads based on price groups 
display(car_ads_data.groupby(['price_group']).count())
# group missing model_year entries by price groups
display(car_ads_data.query('model_year.isnull()').groupby(['price_group']).count())


# <div class="alert alert-warning"> <b>Reviewer comment:</b> 
#     
# <s>It is best to import all libraries at the very beginning of the project, this is a common concept in programming, because it allows you to understand which packages will be used throughout the entire work.
#     
# </div>

# <div class="alert alert-info"> <b>Student comment:</b> Moved the imports to the beginning of the project as recommended. Thank you for educating me on this. </div>

# <div class="alert alert-success"> <b>Reviewer comment (2):</b> 
#     
# It's great that we apply the knowledge gained immediately, so they will be assimilated faster 😊
#     
# </div>

# With the price group in hand now, to fill a missing model_year entry we do the following.
# 
# For each missing model_year in the advertisment dataset,
# 
# 1. Get the advertisment(row in the car_ads_data) with missing model_year as the input
# 2. Find all the ads with non-null model_year ads and with vehicle model matches the model in the input advertisment
# 3. If all the matching models have null entries, then the median of model_year in the entire dataset is filled as the model_year of the input and the row is returned.
# 4. If matching model rows are found in those rows,
#     a. extract  the entries with price_group matching input ad's price group
#     b. If there are ads with matching price_group, get model_year those ads and goto step c. 
#     If they are not found, 
#         get the price_group of ad which has the closest price as input.
#         get the model_year of all the rows in that price_group
#     c. Find the median of all the model_year entries. 
#    
# 5. Fill the input model_year by median model_year from 4.c. and the row is returned.
# 
# We can write a function `assign_model_year` that accepts a row from the car_ads_data and fill its model_year if that is null. Using `apply()` method with axis =1 we will call this function for each row in the dataset and fill all the missing model_year.

# In[326]:



model_group = car_ads_data.groupby('model')
model_price_group = car_ads_data.groupby(['model', 'price_group'])

def assign_model_year(row):
    # find if the model_year entry is missing
    if math.isnan(row['model_year']):
        # get all the ads with non-null model_year entries with the model that matches the row's model
        model_cars = model_group.get_group(row.model).query('model_year.notnull()')
        
        # if there is no matching model then assign the median model year in the entire dataset
        if model_cars.size == 0:
            row['model_year'] = car_ads_data['model_year'].median().round()          
            return row
        
        # Get subset of the model non-null model year entries and with the price group that matches the row
        model_and_price_cars = model_price_group.get_group((row.model, row.price_group)).query('model_year.notnull()')
        
        # If there is no price group with non-null model year, get the closest price group
        if model_and_price_cars.size == 0:           
            closest_idx = model_cars['price'].sub(row.price).abs().idxmin()
            closest_row = model_cars.loc[closest_idx] 
            
            # get the price group of the closest price
            model_and_price_cars = model_price_group.get_group((row.model, closest_row.price_group))
       
        row['model_year'] = model_and_price_cars['model_year'].median().round()
    return row

# use apply to fill the missing values
new_data = car_ads_data.apply(assign_model_year, axis=1)
print("After replacement: ", new_data.isnull().sum())
car_ads_data = new_data


# After grouping by the model and price groups we filled the missing values by taking their median. Next fill the odometer column missing values in way almost similar to `assign_model_year()` we did previously. 
# 
# For this, we calculate the median odometer value for each model_year in a pivot_table, `cars_by_year`. Using `apply()` method we call function `assign_odometer()` for each row in the car_ads_data dataframe. This function fills the missing odometer value in a row by taking its  model year and picking the median odometer value in the cars_by_year pivot table. If the model_year is not available in the pivot table, then a model_year that is closest to that of input row is chosen using `idxmin()`. 

# In[327]:


print("Total missing entries in odometer column: ", car_ads_data['odometer'].isnull().sum())

# pivot table that has median odometer readings for each model_year
cars_by_year = car_ads_data.pivot_table(index='model_year', values='odometer', aggfunc='median')

#  Function assigns odometer based on the median of odometer of entries in the car ads table whose model year is the same as or nearest to the
#  model year in the given row.
def assign_odometer(row):  
    year = row['model_year']
    # if non-null odometer reading then return the row as it is
    if(not math.isnan(row['odometer'])): 
        return row
    
    # if the year has an entry in the pivot table, then get thte median odometer reading
    # Otherwise, get the median odometer reading of the cars with closest year from the pivot table.
    if year in cars_by_year.index:
        closest = year
    else:
        closest = (cars_by_year.index.to_series() - year).abs().idxmin()
    row['odometer'] = cars_by_year.loc[closest, 'odometer']
    return row
car_ads_data = car_ads_data.apply(assign_odometer, axis=1)
display(car_ads_data.head())
print("After filling, missing entries in odometer: ", car_ads_data['odometer'].isnull().sum())


# In[328]:


car_ads_data.isnull().sum()


# Thus we have all missing entries columns filled except for cylinders. Let us find the median number of cylinders in the vehicle ads of models that matcheds the input. Using `fillna()` and `transform()` we can assign the median cylinder count of the model in missing cylinder entry row.

# In[329]:


# get the count of each cylinder type
car_ads_data['cylinders'].fillna(car_ads_data.groupby(['model'])['cylinders'].transform('median'), inplace=True)
print("After filling missing entries, NaN values remaining ", car_ads_data['cylinders'].isnull().sum())


# In[330]:


car_ads_data.isnull().sum()


# #### conclusion
# 
# We found and replaced more than 50% missing entries in the given dataset. Major missing entries were in is_4wd and a simple substituion to 0 for the missing entries saved all those rows. We replaced missing paint_color with 'unknown' instead of leaving it as missing. For the model_year and odometer values, we studied the relationship between them and the price and it really helped us to design the algorithm for filling missing values in those columns. The median model_year is calculated for each model and for each of its  price_groups with which we filled the missing entries. Then we simply used the median odometer values per model_year to fill the missing entries in the odometer column. The missing number of cylinders values are easily replaced using median of cylinders based on model. 
# 
# Our dataset is now complete with no missing entries.

# ### changing datatypes
# 
# After replacing missing values, we need to look at the table if there is a need to change datatype. Remember, at the first glimpse itself, we noted that model_year, odometer and cylinders are continuous which is not representing the real world. Let us revisit the table structure using `info()` again to see what else needs to be modified.

# In[331]:


car_ads_data.info()


# The price column is integer. In the car market prices are mentioned as whole numbers and hence no need to change the data type of price. But, the model_year which represents the year can not be a float value. Hence, this column needs to be changed to integer data type.  Similarly cylinders and odometer need to be changed to integer data type. All these data types are converted using `astype('int64')`
# 
# We already noticed while filling missing values is_4wd holds 1.0 or 0.0 values only based on whether the vehicle is 4 wheel drive or not. This column can better be changed into boolean as it represents a binary value. We can use `astype('bool')` to convert to boolean data type.
# 
# The remaining columns model, fuel, transmission, type, paint_color hold the text to describe vehicle model, fuel used, transmission type, vehicle type like sedan, suv etc., and color respectively. No need to change these fields.
# 
# The date_posted column is represented as string object which we can convert to datetime using `to_datetime()` in pandas.

# In[332]:


# dictionary to hold columns and data types
col_types = {
        'model_year': 'int64', 
        'cylinders': 'int64', 
        'odometer': 'int64', 
        'is_4wd': 'bool'}

for column in col_types:
    car_ads_data[column].astype(col_types[column])
    
car_ads_data['date_posted'] = pd.to_datetime(car_ads_data['date_posted'], format='%Y-%m-%d', errors='coerce')                                           
car_ads_data.info()


# <div class="alert alert-success"> <b>Reviewer comment:</b> 
#     
# 
# Here the `for` loop for automatic type replacement would just help us.
#     
# </div>
# 

# <div class="alert alert-info"> <b>Student comment:</b> Great suggestion. Replaced the code with for loop above.</div>

# <div class="alert alert-success"> <b>Reviewer comment (2):</b> 
#     
# Excellent 😊
#     
# </div>

# #### conclusion
# 
# We explored further on results of  `info()` and replaced data types of model_year, cylinders and odometer to integer data types as we see in the real world. is_4wd is converted to boolean as it was having a binary value. The date_posted column is now made in to pandas friendly datetime object instead of string that will be helpful in doing analysis.

# 
# 
# ## Make calculations and add them to the table
# 
# For further analysis, we need to include few more columns 
# 
#     posted_day_of_the_month -  A number in 1-7 to represent the weekday the ad posted
#     posted_month - A number in 1 - 12, to represent the month ad was posted
#     posted_year - To represent the year ad was posted
#     vehicle_age - To represent age of vehicle as of the day ad was posted
# 
# We calculate vehicle age based on the difference from ad posted_year to the model_year column. This gives an approximate age as we have only model_year not the year the when the vehicle was bought. As we only know the model_year we just subtract year values. Vehicles which have ad posted year and model_year as the same will have vehicle_age as 0, meaning they are only few months old.
#     

# In[333]:


car_ads_data['posted_day_of_the_week'] = car_ads_data['date_posted'].dt.weekday
car_ads_data['posted_month'] = car_ads_data['date_posted'].dt.month
car_ads_data['posted_year'] = car_ads_data['date_posted'].dt.year
car_ads_data['vehicle_age'] = (car_ads_data['posted_year'] - car_ads_data['model_year'])


# Next, let us calculate the mileage per year. For calculating mileages per year we  add an year to vehicle's age, so that the odometer miles driven for posted year also taken into account. For example consider an entry whose model_year is 2018 and ad posted year is also 2018. Though the vehicle's age is less than a year, the miles driven data is critical for analysis and should not be left out. So, for mileage_per_year calculation, we increment vehicle's age by one.

# In[334]:


# mileage/year = odometer/ (vehicle_age + 1)
mileage_in_year = car_ads_data['odometer'] / (car_ads_data['vehicle_age'] + 1)
car_ads_data['mileage'] = mileage_in_year.astype('int64')
car_cols = ['model_year', 'odometer', 'posted_year', 'posted_month','vehicle_age', 'mileage']
car_ads_data[car_cols].head()


# Next requirement is in the condition column, we need to replace string values with a numeric scale:
# 
# ```
#     0 - salvage
#     1 - fair
#     2 - good
#     3 - excellent
#     4 - like new
#     5 - new
# ```
# We can modify the column values using `groupby()` followed by `transform()` methods on the car ads dataframe. The `transform()` function calls `assign_code` which returns the code for the input condition. After the transform operation, the condition column conditon strings are transformed in condition code.

# In[335]:


# assign_code function gets the index of given condition from the condition list
condition_list = ['salvage', 'fair', 'good', 'excellent', 'like new', 'new']
def assign_condn(condn_group):
    # condn_group get the index used for grouping
    condition = condn_group.name
    # condition_list has list of conditions and their indexes represent the respective values
    if condition in condition_list:
        return condition_list.index(condition)
    else:
        return condition
car_ads_data['condition'] = car_ads_data.groupby('condition')['condition'].transform(assign_condn)
car_ads_data.head()


# ### conclusion
# 
# In this step we created four new columns by applying pandas datetime features. posted_day_of_the_week carries the values 1-7 to represent the weekday. Posted_month represents the date in the month which is in the range 1-31, posted_year to represent the year of the posted advertisment, vehicle_age to get age of vehicle from model year to posted year. Mileage to present miles driven per year. Finally, we modified condtion string values into numeric status codes. Thses columns improves readability and helps us to efficiently do our exploratory data analysis which we will see in the coming sections.

# <div class="alert alert-success"> <b>Reviewer comment:</b> 
#     
# 
# We have processed our data in sufficient detail and added all the necessary values ​​to our sample. Let's do the analysis now.
#     
# </div>
# 

# ## Carry out exploratory data analysis
# 
# Now the dataset is preprocessed and ready for analysis. Let us begin our exploratory data analysis by plotting histograms of the following columns:
# 
#     price
#     vehicle_age
#     mileage 
#     cylinders
#     condition
# 
# Histograms present the frequency distribution of the values in a column. This helps us to identify the outliers in dataset. Let us see distribution of values in above columns one by one. 
# <div class="alert alert-info"> <b>Student comment:</b> Added functions as recommended to automate plotting and to avoid repeation </div>
# 
# To automate plotting data, we define two functions - plot_data and plot_hist_range. 
# 
# plot_data will generate plot for any kind of graph - hist, box etc., It has the following arguments,
# 
#     kind - type of graph to be plotted
#     dataset - dataframe to be plotted
#     column - column in the dataset to be plotted
#     title - title for the plot
# 
# plot_hist_range will generate plot for histogram with bins (optional) and range (optional). It has the following arguments,
# 
#     dataset - dataframe to be plotted
#     column - column in the dataset to be plotted
#     title - title for the plot
#     data_range (optional) - range of data to be taken for plotting 
#     bins (optional) - default 10
#   
#    

# In[336]:


# plotting function to plot chart for column given dataset with title
def plot_data(kind, dataset, column, title):
    dataset[column].plot(kind=kind, title=title, figsize = (10,7))
    plt.show()

# plotting function to plot histogram for column given dataset with given range, bins and title.

def plot_hist_range(dataset, column, title, data_range=None, bins=10):    
    if (data_range==None):
        plt.figure (figsize = (10,7))
        plt.title(title)
        plt.hist(dataset[column], bins)
       
    else:        
        plt.figure (figsize = (10,7))
        plt.title(title)
        plt.hist(dataset[column], bins, range=data_range)
    plt.show()

# function to find the upper fence Q3 + (1.5 * IQR) of given dataset column values
def assign_upper_limit(dataset, column):
    
    Q3 = dataset[column].describe()['75%'] 
    Q1 =  dataset[column].describe()['25%'] 
    upper_limit = Q3 + (1.5 * (Q3 - Q1))
    return upper_limit

# function to find the lower fence Q1 - (1.5 * IQR) of given dataset column values
def assign_lower_limit(dataset, column):
    # Q1 - (1.5 * IQR)
    Q3 = dataset[column].describe()['75%'] 
    Q1 = dataset[column].describe()['25%'] 
    lower_limit = Q1 - (1.5 * (Q3 - Q1))
    return lower_limit


# <div class="alert alert-success"> <b>Reviewer comment (2):</b> 
#     
# It's great that we created functions, thereby optimizing our code.
#     
# </div>

# ### Price Analysis
# 
# Remember, we already plotted histogram for price while we categorized the prices for price_group column. Using `value_counts()` learnt that most of the entries have prices less than 40000 and plotted the histogram filtering those entries. Now, let us revisit price column using histogram and boxplots.

# In[337]:


kind = ['hist', 'box']
for i in range(0,2):
    plot_data(kind[i], car_ads_data, 'price', 'Picture {} : price'.format(i+1))


# In Picture1, out of 51,525 records we have 50,000 that is 98% of records falling in the first bin which holds price values from 0 to around 40000. In Picture 2, the box plot shows the values plotted beyond Q4.

# <div class="alert alert-warning"> <b>Reviewer comment:</b> 
#     
# <s>Don't forget to write the title of the chart.
#     
# This is an important point of the whole project, so it is faster to understand what is being discussed on the graph.
#     
# </div>

# <div class="alert alert-info"> <b>Student comment:</b> Done! Thanks for the tip. Encapsulated this in the plot_data function.</div>

# We can see all the outliers lying beyond the upper whishker. Let us filter some of the extreme outliers to get closer to the distribution in the boxplot by setting the ylim. 

# In[338]:


plt.ylim(-100, 50000)
plot_data('box', car_ads_data, 'price', 'Picture 3: price with adjusted limits')


# <div class="alert alert-success"> <b>Reviewer comment:</b> 
#     
# 
# <s>By the way, in order to sign the chart, we can use the `plt.title ()` method and at the end of the code use `plt.show ()`
#     
# </div>
# 
# 

# <div class="alert alert-info"> <b>Student comment:</b> Done! Thanks for the tip</div>

# Now, we can see the median price value at 10000. That is almost 50% of car ads have prices below 10000. 75% have prices approximately below 18000. The upper whishker lies (approx) around 35000 beyond which are all outliers. These are the 2% records that we saw before in the histogram. `describe()` and `value_counts()` method gives the exact numeric values for the distribution we observed in histogram and boxplot. We normalize the value_counts method so that we can find the distribution of price values

# In[339]:


display(car_ads_data['price'].describe())
display(car_ads_data['price'].value_counts(normalize=True, bins=15))


# Based on the above distribution of prices, we infer that as prices go up the number of cars decreases. There are very few highly priced cars. Let us plot histograms with car prices that fall within upper limit of Interquartile Range and car prices that fall above the upper limit of Interquartile Range. 
# 
# Similarly, we plot histograms to see car prices that fall below the lower limit of Interquartile Range and car prices that fall between lower limit and  Interquartile Range. 
# 
# The upper limit and lower limit for the Interquartile Range are calculated by calling the functions `assign_upper_limit()`, `assign_lower_limit()`. Use them as in the ranges that we pass in  `plot_hist_range()` function to get the histograms plotted.

# In[340]:


upper_limit = assign_upper_limit(car_ads_data, 'price')
lower_limit = assign_lower_limit(car_ads_data, 'price')

# list of ranges for different histograms for outlier detection
plot_params = [(upper_limit, car_ads_data['price'].max()), 
               (car_ads_data['price'].median(), upper_limit),
               (lower_limit, car_ads_data['price'].min()), 
               (car_ads_data['price'].min(), car_ads_data['price'].describe()['25%']),
               (car_ads_data['price'].min(), 1000)]
# list of titles for histograms
plot_titles = ['beyond upper limit', 
               'median to upper limit',
               'lower limit to minimum', 
               'minimum to Q1',
               'minimum to 1000']

# use for loops to plot histograms
for index in range(0, 5):
    plot_hist_range(
        car_ads_data, 
        'price', 
        'Picture {}: Prices {}'.format(4+index, plot_titles[index]),
        plot_params[index])


# <div class="alert alert-success"> <b>Reviewer comment:</b> 
#     
# 
# <s>In order not to manually set the `range` parameter, we can see in the public domain how to calculate the interquartile range for the lower and upper levels of the normal distribution.
#     
# </div>
# 
# 

# <div class="alert alert-info"> <b>Student comment:</b> Done. Implemented functions upper_limit() and lower_limit() to calculate upper and lower limits of IQR. Thanks for the suggestion. </div>

# <div class="alert alert-success"> <b>Reviewer comment (2):</b> 
#     
# 👍
#     
# </div>

# Picture 4 shows the prices beyond upper limit, we see there are around 1600 entries in the first bin 50,000 and very few rows after that. 
# 
# Picture 5 tells us that number of entries are gradually decreasing as the price increases. The least number of records in the last bin with price 35,000 has only 300 entries. 
# 
# From these graphs we understand that beyond upper limit of IQR  there are considerable number of advertisements with prices upto 50,000. Advertisements beyond that price are so minimal and perfect candidate for outliers. So will filter the entries with prices atmost 50,000. Next, we will look at the lower outliers.
# 
# From Picture 6, we observe that the lower fence for IQR is a negative value. Since Price of cars can not be negative, there are no entries below 0 in the histogram. But the last bin being the minimum price value with 1 has 800 entrees. Cars with price as 1 are candidates to be considered as outliers. 
# 
# Picture 7 shows distribution of prices from the minimum to Q1, we see the first bin has higher number of entries. The 800 entries with price has 2/3 portion of this first bin. Picture 8 further looks closer at the values at the lower extreme from 1 to 1000 most entries have price within 100. Vehicle with prices 100 are too far from the median and mean in our dataset and in real world this is truely an artifact. 
# 
# Let us have a quick glimpse at the vehicles with price less than 100 in our dataset.

# In[341]:


car_ads_data.query('price <= 100').head()


# we can be pretty sure that most of these cars are worth more than the price mentioned. The above results have vehicles in all conditions and ages. So, there might be a problem with the data entry or with the source where the data is fetched. This needs to be further investigated with the team that sourced the data. As price is an important feature for our analysis, with the inaccurate prices would not help us to derive meaningful conclusions. So let us remove this outliers in our dataset.
# 
# We found the lower and upper limit for outliers, and now we will filter vehicles whose prices greater than 1 and prices less than 50000 and store them in a separate dataset.

# In[342]:



car_ads_data_filtered = car_ads_data.query('price >=100 and price <= 50000')
# histogram with sliced data in car_ads_data_filtered
plot_hist_range(car_ads_data_filtered, 'price', 'Picture 8: Prices after filtering outliers', bins=20)


# <div class="alert alert-success"> <b>Reviewer comment:</b> 
#     
# 
# <s>And there we have the title 😊👏
#     
# ---
#     
# We can also enlarge the graph itself to make it easier to study using the method `plt.figure (figsize = (10,7))`
#     
# </div>
# 
# 

# <div class="alert alert-info"> <b>Student comment:</b> Thanks for the suggestion. I included figsize setting in the plotting function.</div>

# <div class="alert alert-danger"> <b>Reviewer comment:</b> 
#     
# 
# <s>Let's try to automate the plotting through the `for` loop.
#     
# Why is it important?
#     
# - we save our time
# - optimizing the code
# - we make the work more presentable
#     
# Choose any part of the code where it is most appropriate to do it.    
# </div>
# 
# 

# <div class="alert alert-info"> <b>Student comment:</b> Done. It is a meaningful suggestion. It really optimized code and improved readability. </div>

# <div class="alert alert-success"> <b>Reviewer comment (2):</b> 
#     
# Very good 😊👍🎉
#     
# </div>

# Now we can compare and analyse the raw dataset and the filtered dataset by plotting them in one graph. We will  plot the vertical lines for raw and filtered mean.

# In[343]:


car_ads_data_filtered = car_ads_data_filtered.query('price > 100')
ax = car_ads_data.plot(kind='hist', y='price', histtype='step', bins=25,
                              linewidth=5, alpha=0.7, label='raw')
car_ads_data_filtered.plot(kind='hist', y='price', histtype='step', bins=25, 
                       linewidth=5, alpha=0.7, label='filtered', ax=ax, grid=False, legend=True)
ax.set_xlabel('Vehicle Prices')
ax.axvline(x = car_ads_data['price'].mean(), color ="blue", linestyle ="dotted", label="Raw Mean")
ax.axvline(x = car_ads_data_filtered['price'].mean(), color ="orange", linestyle ="dotted", label="Filtered Mean")
plt.title("Price Analysis")
ax.legend()
plt.show()


# #### conclusion
# Thus the outlier removal gave a total different perspective about the price distribution. In the first picture we could not derive much information. We see most of the ads have prices between 0-50,000 and not much information otherwise. But once the dataset is filtered after the outliers removed, we can peak come at the second bin when the prices are less than 5000. We can see how many rows are distributed along each bins. The mean and median are not much changed after removal of outliers.

# <div class="alert alert-success"> <b>Reviewer comment:</b> 
#     
# 
# It's great that we are comparing our data in one graph.
#     
# </div>

# ### Analysis on vehicle's age 
# 
# Next, we will look at the distribution of age of vehicles in the ads. We can view the distribution of ages using `describe()` and then plot the histogram using `hist()` with bin size as 10.

# In[344]:


upper_limit_age = assign_upper_limit(car_ads_data, 'vehicle_age')
print("Upper Limit for IQR: ", upper_limit_age)
print(car_ads_data['vehicle_age'].describe())
[plot_data(kind[i], car_ads_data,'vehicle_age', 'Vehicle Age') for i in range(2)]


# <div class="alert alert-success"> <b>Reviewer comment:</b> 
#     
# 
# <s>Please note that our actions are repeated in terms of plotting.
#     
# This suggests that we can create a function that will do all the actions for us.
#     
# Thus, we will be able to optimize our code and project as a whole.
#     
# </div>

# <div class="alert alert-info"> <b>Student comment:</b> Implemented plotting functions </div>

# Thus both the histograms and boxplot clearly shows the presence outliers  in the upper extreme values for vehicle ages. The average and median ages are almost around 8 years while the maximum is 100 years. A 100 year old car is clearly an outlier. This value is too far from the mean and median values of vehicle ages. From the box plot above, the cars with ages above 24 lying beyond Q4 are potential outliers which may affect the visualization or mean/median values of the dataset. Let us filter out those outliers

# In[345]:


car_ads_data_filtered = car_ads_data.query('vehicle_age <= @upper_limit_age')
plot_data('hist', car_ads_data_filtered,'vehicle_age', 'Vehicle Age less than Upper limit ( <= {})'.format(upper_limit_age))


# After filtering the unrealistic data, now the histogram highly readable and gives a better insight on the age of vehicles and their frequency. The histogram is right skewed telling us that there are huge number of cars with recent model years than the old one. As the ages go above and beyond 15, there is steady fall in their frequency.
# 
# Next, let us find the mean of the raw and filtered datasets. Draw histograms of these dataset with vertical lines to represent the mean values and draw conclusions after comparing them.

# In[346]:


ax = car_ads_data.plot(kind='hist', y='vehicle_age', histtype='step', bins=25,
                              linewidth=5, alpha=0.7, label='raw')
car_ads_data_filtered.plot(kind='hist', y='vehicle_age', histtype='step', bins=25, 
                       linewidth=5, alpha=0.7, label='filtered', ax=ax, grid=False, legend=True)
ax.set_xlabel('Vehicle Ages')
ax.axvline(x = car_ads_data['vehicle_age'].mean(), color ="blue", linestyle ="dotted", label="Raw Mean")
ax.axvline(x = car_ads_data_filtered['vehicle_age'].mean(), color ="orange", linestyle ="dotted", label="Filtered Mean")
plt.title("Vehicle Age Analysis")
ax.legend()
plt.show()

# calculate the size of data removed for analysis
print("Percentage of outlier data that are removed from original dataset: {:.2%}".format(
      round(len(car_ads_data)- len(car_ads_data_filtered))/len(car_ads_data)))


# ## conclusion
# With the outlier removed Vehicle ages are distributed evenly among the bins. But in the raw data set gives no clue about how the cars with vehicle ages between 0 - 20 are spread between the bins. They are piled together in the same bin. With the filtered dataset in orange, there is a great clarity in the distribution of vehicles based on their ages. Removing outliers did not deviate mean values much from the raw data set, but improved the readability and form of the histogram. Thus without affecting the dataset much, we removed outliers and reduced the vehicle ads dataset size to some extent.
# 
# 

# ###  Analyzing Vehicle Mileage
# 
# Like the price and age features analysed before, let us analyze the vehicle mileages. We can view the distribution of mileages using describe() and then plot the histogram using hist() with bin size as 10.

# In[347]:


upper_limit_mileage = round(assign_upper_limit(car_ads_data, 'mileage'))
print("Upper Limit for IQR: ", upper_limit_mileage)
display(car_ads_data['mileage'].describe().apply("{0:.5f}".format))
plot_data('hist', car_ads_data, 'mileage', 'Mileage')
plt.show()


# The results of  `describe()` above, we can see a huge gap between the 75% percentile value and the maximum value. The maximum mileage is so high. In the histogram we could not locate this maximum value directly but with the widened x axis to hold mileage values beyond 350K clearly shows the presence of high mileage values. This maximum value in the dataset, totally shrinked the quartile region in the box plot. Let us see the details of the car which holds the maximum value to understand  why this value is so high.

# In[348]:


car_ads_data.query('mileage == mileage.max()')


# The vehicle with maximum mileage is only few months old and has high the odometer value. This looks like an artifact - A few months old car, have to be driven 1000+ miles each day to have such high odometer reading. This unrealistic and we need to discuss with the team that sourced the data. As per Federal Highway Administration, Americans drive an average of 14,300 miles per year, which is almost close to our mean mileage of our dataset. Now to decide mileage limit beyond which are outliers, we have to look at the histogram. In 10 bins between 0-50000, 8 bins hold most of the car mileages. Also, we already saw that the upper limit of InterQuartile Range for mileage as 29162.

# In[349]:


car_ads_data_filtered = car_ads_data.query('mileage <= @upper_limit_mileage')
plot_data('hist', car_ads_data_filtered, 'mileage', 'Filtered Mileages')


# Now with the filtered dataset the readability of the histogram greatly improved. We have got a balanced histogram  with the peak occuring at the fourth bin. As the form of histogram is balanced we see gradual increase in the frequency as the mileages increases and  after reaching the peak frequency starts to descend. 
# 
# Next, we will compare the raw and filtered mileages of the car ads dataset. We will find and compare mean values in both datasets.

# In[350]:


ax = car_ads_data.plot(kind='hist', y='mileage', histtype='step', bins=25,
                              linewidth=5, alpha=0.7, label='raw')
car_ads_data_filtered.plot(kind='hist', y='mileage', histtype='step', bins=25, 
                       linewidth=5, alpha=0.7, label='filtered', ax=ax, grid=True, legend=True)
ax.set_xlabel('Vehicle Mileages')
plt.title("Vehicle Mileage Analysis - Raw & Filtered Comparison")
ax.axvline(x = car_ads_data['mileage'].mean(), color ="blue", linestyle ="--", label="Raw Mean")
ax.axvline(x = car_ads_data_filtered['mileage'].mean(), color ="orange", linestyle ="--", label="Filtered Mean")
ax.legend()
plt.show()


# #### conclusion
# 
# 
# 

# From the above comparision of dataset containing raw and filtered mileages, we infer that after filtering, the dataset distribution is balanced and gives a clarity about how the mileages are distributed. The mean values did not change much before and after applying the filter. Thus filtering outliers dramatically improvised the form and readability of the histograms.

# ### Number of Cylinders analysis
# 
# Next, let us analyse the distribution of number of cylinders in our dataset. First let us see what `describe()` and `hist()` tells about the data.

# In[351]:


display(car_ads_data['cylinders'].describe())
plot_data('hist', car_ads_data, 'cylinders', 'Cylinders')


# The number of cylinders is ranging from 3 to 12. We can infer that this column has categorial values from 3-12. Most of the vehicles have 3-9 cylinders. Very few vehicles have cylinders less than 3 or more than 9. As these are like categorical values, let us use value_counts() to get a clear distribution of number of cylinders in the car ads dataset.

# In[352]:


car_ads_data['cylinders'].value_counts(normalize=True)


# 98% of cars are 4-, 6- or 8-cylinders categories. 1% of cars have 10 cylinders. 0.6% of cars have 5 cylinders. Cars with 12 cylinders is the rarest one in this dataset. Next less popular cylinder category is 3 -cylinder and 10-cylinder cars. Cars having less than 4 cylinders and more than 8 cyinders are thus rare and so we can filter out those cars.

# In[353]:


car_ads_data_filtered = car_ads_data.query('cylinders <=8 and cylinders > 3')
plot_hist_range(car_ads_data_filtered, 'cylinders', 'Filtered Cylinders')


# Now with the filtered dataset, we will compare the original one and draw our conclusions.

# In[354]:


ax = car_ads_data.plot(kind='hist', y='cylinders', bins=10,
                              linewidth=5, alpha=0.7, label='raw')
car_ads_data_filtered.plot(kind='hist', y='cylinders', bins=10, 
                       linewidth=5, alpha=0.7, label='filtered', ax=ax, grid=True, legend=True)
ax.set_xlabel('Number of Cylinders')
plt.title("Number of cylinders  Analysis - Raw & Filtered Comparison")
ax.axvline(x = car_ads_data['cylinders'].median(), color ="red", linestyle ="-", label="Raw Mean")
ax.axvline(x = car_ads_data_filtered['cylinders'].median(), color ="green", linestyle ="-", label="Filtered Mean")
ax.legend()
plt.show()


# #### conclusion
# 
# The histogram is balanced before and after removing filters. Since the size of outlier data we filtered are small, we could not see much changes in the mean. Thus removing outliers in number of cylinders has little impact on the distribution.

# <div class="alert alert-success"> <b>Reviewer comment:</b> 
#     
# 
# If you like the visualization theme, I recommend checking out the `seaborn` library.
#     
# </div>

# #### Vehicle Condition Analysis
# 
# Like number of cylinders, vehicle condition is also categorical data. So we will use `value_counts()` and `hist()` to see and plot the frequency of vehicles in each condition.
# 

# In[355]:


display(car_ads_data['condition'].value_counts(normalize=True))
plot_hist_range(car_ads_data,'condition', 'Condition', bins=20)


# From the above histogram we can sense that cars with salvage condition and vehicles with new condition are very few when compared to other conditions. This mimics the real world where most new car owners wont sell their just bought cars. Similarly as salvage condition cars are not usable condition and sellers with this car condition are also uncommon. But in the histogram there is peak in 3, which is for excellent car condition. Thus, Most cars advertised for resale are in excellent condtion. We will remove the 0 and 5th condtion car ads and see how the histograms appear after filtering.

# In[356]:


car_ads_data_filtered = car_ads_data.query('condition > 0 and condition < 5')
plot_hist_range(car_ads_data_filtered, 'condition', "Filtered Data: Condition", bins=20)


# The removal of 0 and 5 values from condition histogram makes us to focus only on the cars with the most common conditions which are excellent, like new, good and fair in the market.
# 
# Now, let us compare raw and filtered datasets. We will derive the median values in both the datasets. Based on the results let us draw our conclusions. 

# In[357]:


ax = car_ads_data.plot(kind='hist', y='condition', bins=10,
                              linewidth=5, alpha=0.7, label='raw')
car_ads_data_filtered.plot(kind='hist', y='condition', bins=10, 
                       linewidth=5, alpha=0.7, label='filtered', ax=ax, grid=True, legend=True)
ax.set_xlabel('Car Conditions')
plt.title("Car Condtions  Analysis - Raw & Filtered Comparison")
ax.axvline(x = car_ads_data['condition'].median(), color ="red", linestyle ="-", label="Raw Median")
ax.axvline(x = car_ads_data_filtered['condition'].median(), color ="green", linestyle ="-", label="Filtered Median")
ax.legend()
plt.show()


# #### conclusion
# 
# The histogram is balanced before and after removing filters. The median in both datasets are the same pointing to 3. Since the size of outlier data we filtered are small, we could not see much changes in the median. Thus removing outliers in car conditions enhances the readability of the histogram without affecting the mean, median and frequency of the distribution.

# ### Analyzing  lifetime of advertisment
# 
# An advertisment's lifetime begins when an ad is posted for sale and it will continue to be there till the seller removes the ad from the site. A seller removes the ad when the vehicle is sold or when he changes his mind not to sell his vehicle. If any of those events happen, the advertisment's life comes to an end. To represent the number of days the advertisement is up, we have a column in the car ads dataset as  "days_listed". We will analyze the distribution of this variable by using `describe()` and `value_counts()` to represent numerically and `hist()` to represent visually as a histogram.

# In[358]:


ax = car_ads_data.plot(kind='hist', y='days_listed', bins=100, grid=False, alpha=0.6)
ax.axvline(x = car_ads_data['days_listed'].median(), color ="red", linestyle ="--", label="Median - days listed")
ax.axvline(x = car_ads_data['days_listed'].mean(), color ="black", linestyle ="--", label="Mean - days listed")
ax.set_xlabel('days listed')
plt.title("Days listed - Analysis")

plt.legend()
display(car_ads_data['days_listed'].describe())
display(car_ads_data['days_listed'].value_counts().tail())
display(car_ads_data['days_listed'].value_counts().sort_index().head())
display(car_ads_data['days_listed'].value_counts().sort_index().tail())
plt.show()


# #### conclusion
# 
# From the results of `describe()` we infer that the life of an advertisement ranges from 0 to 271 days. With the histogram, we observe that peak frequency occurs when the ads posted have 15-30 days as their values. The histogram is right skewed with most advertisments having the lower number of ad listed days. As the number of days listed increases its frequency decreases. We observe that most advertisements live only for two-four weeks, that is 15-30 after it is posted. The mean falls behind the peak value which is 39.55 days. The median is slightly closer to the peak than the mean with 33 days. As number of days listed goes beyond 170, the frequency dips to the level that it is invisible in the histogram.
# 

# ### Analysing advertisments based on vehicle types 
# 
# Now, let us find the different vehicle types in the dataset and for each of those vehicle types we shall calculate the number of advertisements listed and their mean price. To do this we will create pivot table with the vehicle types as indexes and price as the column values. 

# In[359]:


type_data = car_ads_data.pivot_table(index='type', values='price', aggfunc=['count', 'mean'])
type_data.columns = ['total', 'average_price']
type_data['average_price'] = round(type_data['average_price'], 2)
type_data = type_data.sort_values('total', ascending=False)
display(type_data)


# The vehicle types holds string values that are of nominal types - they are names of each vehicle type. To visualize these kind of values bar charts or pie charts helps us. we can apply the `plot()` to present pie chart for visual treat. 

# In[360]:


# get the y as total and x as index from the type_data pivot table we calculated before
# This to get the label with percentage in the legend box - as few closer labels in the pie are overlapping
y=type_data['total']
x=type_data.index

patches, texts = plt.pie(y, radius=1.9, labels=x, startangle=45, rotatelabels=True, labeldistance=1)
ax.set_ylabel('')

percent = 100.*y/y.sum()
labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, percent)]

# sort the vehicle type percentages descending order
patches, labels, dummy =  zip(*sorted(zip(patches, labels, y),
                                          key=lambda x: x[2],
                                          reverse=True))
# set the location and apperance of legends
plt.legend(patches, labels, bbox_to_anchor=(1.3, 1.),loc='best',
           fontsize=8)

plt.savefig('piechart.png',  bbox_inches='tight')



# #### conclusion
# 
# Thus we aggregated the dataset by vehicle types and found the total ads and mean prices of each type. Next we plotted the pie chart to show percentage of ads in each vehicle type. From the pie chart, it is evident that top two most popular vehicle types as SUV and Truck. When we compare the mean price, the Truck costs more than the SUVs.

# ### Analyzing factors impacting price
# 
# We found the popular vehicle types among the advertisements in the previous section. Next, we need to see how the features such as age, mileage, condition, transmission type, and color affect the price in each these popular vehicle types. We already analysed the relationship of price to odometer and model_year are significant.
# 
# - odometer has the negative correlation, that is high odometer reading has lower prices and vice versa
# - model_year has positive correlation, that is higher the model_year, higher the price. Lower the model_year that is lower the price. 
# 
# We have not yet analyzed the correlation of price to new columns during preprocessing. We did not analyze price correlation in the dataset based on vehicle types before. So, let us revisit the price correlation for the records that belong to SUV and Truck which are the most popular vehicle types among the posted ads.
# 
# To check the dependancy we need to use `corr()` method in dataframe. To plot them visually we need to use scatter plots. For categorical columns we will use boxplots to display price ranges for each category.
# 
# To analyse the features that impact the price, we split the columns into two lists.
# ```
# numerical_cols = ['vehicle_age', 'mileage', 'condition']
# category_cols = ['transmission', 'paint_color']
# ```
# We will write a function `corr_to_price_by_type` that takes a vehicle type and axes as input. The function then retrieve all the advertisements of the given vehicle type. Then we explore numerical_cols in those entries and plot scatter plots against price for each of those columns in the given axes.
# 
# The same function will handle plotting box plots for categorical columns with categories having more than 50 entries.
# 
# With the `type_data` pivot table we constructed before, we will take the top two vehicle types and call the function `corr_to_price_by_type` method to get the scatter and box plots plotted.

# In[361]:


numerical_cols = ['vehicle_age', 'mileage', 'condition']
category_cols = ['transmission', 'paint_color']
figure_one, axes_one = plt.subplots(nrows=1, ncols=5, sharey=True, gridspec_kw={'width_ratios': [4,4,4,10,10]}, figsize=(20,5))
figure_two, axes_two = plt.subplots(nrows=1, ncols=5, sharey=True, gridspec_kw={'width_ratios': [4,4,4,10,10]}, figsize=(20,5))

# ignore depreciated numpy array warnings

warnings.filterwarnings('ignore')

# function to plot scatter/box plot to analyze impact of age, mileage, condition, transmission, paint color 
# to price of vehicle ads based on input vehicle type

def corr_to_price_by_type(vehicle_type, axes):
    # extract the vehicle ads based on type
    popular_type_data = car_ads_data.query('type == @vehicle_type')
    

    print("The age, mileage and condition correlation to price of {} type vehicle ads:".format(vehicle_type))    
    # for quantiative columns - age, mileage and condition - find p value and graph the scatter plot  
    ax_y = 0
    for col in numerical_cols:
        print("{} to price: {:.3f}".format(col,popular_type_data[col].corr(popular_type_data['price'])))
    
        popular_type_data.plot(x=col, y='price', kind='scatter', alpha=0.3, ax=axes[ax_y])
        ax_y += 1
    
    # for categorical columns - transmission, paint_color - plot the box plot for categories appearing in atleast 50 ads
    for cat_column in category_cols:
        # extract the each category and its slice into a dictonary
        category_dict = {}
        for category_name, category_slice in popular_type_data.groupby(cat_column)['price']:
                if len(category_slice) >= 50:
                    category_dict[category_name] = category_slice
               
        axes[ax_y].boxplot(category_dict.values(), labels=category_dict.keys())
        
        axes[ax_y].set_xticklabels(category_dict.keys(), rotation=45, fontsize=12)
        ax_y += 1
    
# Get the first popular vehicle types from the type_data pivot table we constructed before
figure_one.suptitle("Analyzing {} Advertisments".format(type_data.index[0]), size='xx-large')
corr_to_price_by_type(type_data.index[0], axes_one)
# Get the second popular vehicle types from the type_data pivot table we constructed before
corr_to_price_by_type(type_data.index[1], axes_two)
figure_two.suptitle("Analyzing {} Advertisements".format(type_data.index[1].upper()), size='xx-large')
plt.show()


# ax = car_ads_data.plot(kind='hist', y='condition', bins=10,
#                               linewidth=5, alpha=0.7, label='raw')
# car_ads_data_filtered.plot(kind='hist', y='condition', bins=10, 
#                        linewidth=5, alpha=0.7, label='filtered', ax=ax, grid=True, legend=True)
# 

# We found that SUV and Truck as the top vehicle types in the vehicle ads dataset before. The correlation of vehicle age to price in SUV type vehicles and Truck type vehicles are almost the same which is -0.57. This means that there is a significant correlation between age of vehicles and the price in the negative way. Thus the data shows the the same fact which we see in the real world that vehicle prices depriciate as they get older. The negative correlation is perfectly visualized in the scatter plot for vehicle_age. 
# 
# Mileage to price correlation is not as significant as price in both  SUV ads and Truck ads. These ads have their correlation values as 0.19 and 0.21 which are almost the same. When comparing their scatter plots, SUV vehicles have lot of high priced cars with lower mileages than the truck.
# 
# The condition to price correlation is also not much significant. But when comparing the SUV vehicles to the truck, SUVs have less correlation value as 0.26 than trucks which 0.30. The SUVs  have higher prices for excellent conditon vehicles than for like new and new conditions. For truck good, excellent and like new ocnditions has the same level of distribution and highest prices in each conditions are almost the same. There are few trucks with new condition but they also have the highest price same as the good, excellent and like new conditions. The salvage and fair condition SUVs and Trucks are price very low and they are few in numbers. Truck have more new condition vehicles than SUVs.
# 
# Next we will analyze category based columns  - Transmission and Colors affecting the price. We displayed the boxplots for categories that has atleast 50 entries. We 
# 
# Next, let us analyze the transmission types and the price of SUVs and trucks. The lower whiskers are missing in the transmission type with others in both SUVs and trucks. Let us see the value counts of price column with tranmission type as 'other' in SUVs and Trucks separately. 
# 

# <div class="alert alert-success"> <b>Reviewer comment:</b> 
#     
# 
# Wow, interesting chart for our visualization.
#     
# </div>

# In[362]:


display(car_ads_data.query('type == "truck" and transmission == "other"')['price'].value_counts(normalize=True).head())
display(car_ads_data.query('type == "SUV" and transmission == "other"')['price'].value_counts(normalize=True).head())


# From the above results we see there are 56% rows with price as 1 in Trucks and 41% rows with price as 1 in SUVs. The lower limits are unrealistically low and in large number. This pulled the median price of Trucks to 1 that both whisker and Q1 are equal to Q2 which is median. For SUVs, percentage of price with 1 falls below 50% so this becomes Q1 and whisker at the lower limit is invisible. Thus the box plot simply exposed the huge artifact about prices of vehicles with tranmission type as other. This data needs to be investigated further with the team that sourced the data. 
# 
# When exploring further on the automatic transmission for SUVs and Trucks we found that there large number outliers beyond the upper limit of whisker. But for manual and other tranmission such outliers are significantly less. The price ranges of automatic truck are higher than the manual trucks.
# 
# Next, let us look at the box plots of prices based on color. For SUVs most popular color is black with largest price range from Q1-Q4. When comparing median price in each color of SUV types orange colored SUVs top the list. In Trucks yellow is the rarest color and has the lowest price range. Black trucks have large price ranges similar to SUVs. For Trucks, there is a color label called 'Custom'. It shows that there are considerable number of trucks with custom color. 

# #### conclusion
# 
# We find `corr()` values of numerical columns - age, mileage and conditions against price. We found the significant negative correlation of age to price. The scatter plots gave further clarity. The box plots for categorical columns with categories against price undercovered huge artifact that the price values has 1 in half of the entries with transmission as "other". We found black is popular color in SUVs and Trucks in all price ranges. Orange color had the highest median price among other colors in SUVs. Thus, exploratory data analysis gave great insight about the relationship of price to the other features  and uncovered lot of details about the data.

# ## Overall conclusion

# Using Exploratory data analysis we got lot of insights of the dataset. We opened the dataset and using `info()` and `describe()` we got the first glimpse on the columns and datatypes. 
# 
# We found more than 50% of rows have missing entries. We further explored and find that is_4wd column has huge number of missing entries. Then we found that they take binary values and replaced missing entries with 0 values which was changed to boolean datatype later. The paint_color missing values were assigned "unknown". Then we filled model_year by getting median model_year of the matching price group. We filled missing odometer values by grouping model_years. We took the median odometer value of the model_year corresponding to the missing entry. Then we filled cylinders by grouping cylinders by model and taking the median cylinders of the model corresponding to the missing entry.
# 
# We changed datatypes of odometer, cylinders to integer, is_4wd to boolean, date_posted to pandas datetime object. 
# 
# Then we created new columns for Day of the week, month, and year the ad was placed, The vehicle's age (in years), The vehicle's average mileage per year. Then we modified in the condition column, replace string values with numbers.
# 
# We studied the parameters: price, vehicle's age when the ad was placed, mileage, number of cylinders, and condition. We plotted histograms for each of these parameters and found the outliers and how they affected the form and
# readability of the histograms. We determined the upper limits of outliers, removed the outliers and stored the filtered data to plot new histograms. We compared raw and filtered datasets and drew conclusions.
# 
# Then we studied how many days advertisements were displayed and plotted the histogram for that column. We calculated the mean and median and found the typical lifetime of an ad. 
# 
# We analyze the number of ads and the average price for each type of vehicle and plotted a pie chart showing the dependence of the number of ads on the vehicle type. We selected SUVs and trucks as top two popular vehicle types and analysed factors that impacted the price most in those advertisements. We studied dependency of the price to the
# age, mileage, condition, transmission type, and color. For categorical variables, we choose categories with atleast 50 entries and we plotted box-and-whisker charts. For non-categorical columns we created scatterplots. 
# 
# We found the significant negative correlation of age to price.  The box plots for transmission columns with categories against price undercovered huge artifact that the price values has 1 in half of the the entries with transmission as "other". We discovered black is the most common color in SUVs and Trucks in all price ranges.
# 
# Thus, exploratory data analysis gave great insight about the dataset and uncovered lot of artifacts that greatly improved the quality of data.
# 

# <div class="alert alert-success"> <b>Reviewer comment:</b> 
#     
# 
# The conclusions are clear and logical, and most importantly, they are supported by the revealed facts.
#     
# The work done in sufficient detail and this cannot but rejoice 😊
#     
# Let's correct my comments and move on.
#     
# If you suddenly have any questions, I will be happy to answer them 😊
#     
# </div>

# <div class="alert alert-success"> <b>Reviewer comment (2):</b> 
#     
# It's great that we quickly corrected our project and made it even better 😊 A particularly important nuance in all training, apply the knowledge gained immediately and we cope with it with a bang 😊
# 
# In general, the project itself is at a good level! Throughout the entire work, the depth of the task is felt and this is an absolute plus for us. A large number of methods have been used that will help you in future projects.
#     
#     
# Congratulations on the successful completion of the project 😊👍
#     
# And I wish you success in your new work 😊
#     
# **[general comment]** Cells **markdown** including we can use as an additional way to style our project. I left a link below, with examples of styling.
# 
# https://sqlbak.com/blog/jupyter-notebook-markdown-cheatsheet
#     
# </div>

# # Project completion checklist
# 
# Mark the completed tasks with 'x'. Then press Shift+Enter.

# - [x]  file opened
# - [x]  files explored (first rows printed, info() method)
# - [x]  missing values determined
# - [x]  missing values filled in
# - [x]  clarification of the discovered missing values provided
# - [x]  data types converted
# - [x]  explanation of which columns had the data types changed and why
# - [x]  calculated and added to the table: day of the week, month, and year the ad was placed
# - [x]  calculated and added to the table: the vehicle's age (in years) when the ad was placed
# - [x]  calculated and added to the table: the vehicle's average mileage per year
# - [x]  the following parameters investigated: price, vehicle's age when the ad was placed, mileage, number of cylinders, and condition
# - [x]  histograms for each parameter created
# - [x]  task completed: "Determine the upper limits of outliers, remove the outliers and store them in a separate DataFrame, and continue your work with the filtered data."
# - [x]  task completed: "Use the filtered data to plot new histograms. Compare them with the earlier histograms (the ones that included outliers). Draw conclusions for each histogram."
# - [x]  task completed: "Study how many days advertisements were displayed (days_listed). Plot a histogram. Calculate the mean and median. Describe the typical lifetime of an ad. Determine when ads were removed quickly, and when they were listed for an abnormally long time.  "
# - [x]  task completed: "Analyze the number of ads and the average price for each type of vehicle. Plot a graph showing the dependence of the number of ads on the vehicle type. Select the two types with the greatest number of ads. "
# - [x]  task completed: "What factors impact the price most? Take each of the popular types you detected at the previous stage and study whether the price depends on age, mileage, condition, transmission type, and color. For categorical variables (transmission type and color), plot box-and-whisker charts, and create scatterplots for the rest. When analyzing categorical variables, note that the categories must have at least 50 ads; otherwise, their parameters won't be valid for analysis.  "
# - [x]  each stage has a conclusion
# - [x]  overall conclusion drawn
