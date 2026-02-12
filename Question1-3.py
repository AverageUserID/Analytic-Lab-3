# %%
# Import libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For scaling
from io import StringIO  # For reading string data as file
import requests  # For HTTP requests to download data

# %% 
# Step 1
# One problem to be solved with the College Completion Dataset:
# How can we accurately predict 4 year graduation rate?

# One problem to be solved with the Job Placement Dataset:
# How can we accurately predict whether a candidate is placed or not?
# %%
# Step 2
# College data
# Independent bussiness metric: grad_100_rate (the graduation rate within 100% of normal time)
# Importing the data
college = pd.read_csv("https://raw.githubusercontent.com/UVADS/DS-3021/refs/heads/main/data/cc_institution_details.csv")
# %%
college.info()
# %%
# First dealing with null values
# Using print and to_string to see avoid pandas truncating the data
# Sorting the value to see the columns with the most misisng value
print(college.isna().sum().sort_values(ascending=False).to_string())
# Flagship and hbcu seems to be boolean columns with null value as 0
college["hbcu"] = college["hbcu"].apply(lambda x : 1 if x == "X" else 0)
college["flagship"] = college["flagship"].apply(lambda x : 1 if x == "X" else 0)
# %%
# Only keep columns with less than 20% null values 
college_clean = college.dropna(axis=1, thresh=int(0.8 * len(college)))
# %%
# Now dropping all the rows with null values
college_clean = college_clean.dropna()
college_clean.shape
# %%
# Filtering for all the str columns
cols = college_clean.columns[college_clean.dtypes == "str"]
cols
# %%
# Droping all the columns not fit for making a category
college_clean = college_clean.drop(["chronname", "city", "state", "basic", "site", "similar", "counted_pct"], axis = 1)
# %%
# Changing the data type into category
cols = college_clean.columns[college_clean.dtypes == "str"]
college_clean[cols] = college_clean[cols].astype('category')
# %%
# Now moving on to the numeric columns
# Dropping index, unitid, long_x, and lat_y, because they are either unique or nominal in a sense
college_clean = college_clean.drop(["index", "unitid", "long_x", "lat_y"], axis = 1)
numeric_cols = list(college_clean.select_dtypes('number'))
numeric_cols.remove("grad_100_percentile")
# %%
numeric_cols
# %%
college_clean[numeric_cols] = MinMaxScaler().fit_transform(college_clean[numeric_cols])
# %%
# Now apply the One-Hot encoding
category_list = list(college_clean.select_dtypes('category'))
college_encoded = pd.get_dummies(college_clean, columns=category_list)
# %%
# The target variable here will be grad_100_percentile
# First we describe the target variable
print(college_encoded.grad_100_percentile.describe())
# The upper quantile is at 0.74
# %%
college_encoded['grad_100_percentile_cat'] = pd.cut(
    college_encoded['grad_100_percentile'],
    bins=[-0.01, 50, 80, 100],
    labels=[0, 1, 2]
)

college_encoded.info()
# %%
# Calculate the prevalence
prevalence = (
    college_encoded['grad_100_percentile_cat']
        .value_counts(normalize=True)
        .sort_index()
)
print(prevalence)

# %%
college_encoded = college_encoded.drop(["grad_100_value", "grad_100_percentile", "grad_150_value", "grad_150_percentile"], axis = 1)
college_encoded.info()
# %% 
# Now for the train tune test split
# I will do a 70 30 train test split first
train, test = train_test_split(
    college_encoded,
    train_size=0.7,
    stratify=college_encoded.grad_100_percentile_f
)
print(f"Training set shape: {train.shape}")
print(f"Test set shape: {test.shape}")
# %%
# And now a 50 50 tune test split
tune, test = train_test_split(
    test,
    train_size=0.5,
    stratify=test.grad_100_percentile_f
)
print(f"Training set shape: {tune.shape}")
print(f"Test set shape: {test.shape}")


# %%
# Job data
# Independent bussiness metric: status
# Importing the data
job = pd.read_csv("https://raw.githubusercontent.com/DG1606/CMS-R-2020/master/Placement_Data_Full_Class.csv")
# %%
# Explore the structure of the dataset
job.info()
# Check for data types and identify which columns need type conversion
# %%
job["status"].value_counts()
# %%
# Categorical covnersion of str columns and standarisation of numeric columns
# Keeping status but changing it to boolean because that is the independent bussiness
cat_cols = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation"]
needstand_cols = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p", "salary"]
job[cat_cols] = job[cat_cols].astype('category')
job[needstand_cols] = MinMaxScaler().fit_transform(job[needstand_cols])
job["status"] = job["status"].apply(lambda x: 1 if x == "Placed" else 0)
# %%
job["status"].value_counts()
# %%
# One-hot encoding
job_list = list(job.select_dtypes('category'))
job_encoded = pd.get_dummies(job, columns=job_list)
job_encoded.info()

# %%
# Dropping index column
job_clean = job_encoded.drop(['sl_no'], axis=1)
# %%
job_clean.head()

# %%
# Dealing with the null values
job_clean["salary"] = job_clean["salary"].apply(lambda x : 0 if pd.isna(x) else x)
# %%
job_clean.status.value_counts
# %%
# Calculate the prevalence
prevalence = (job_clean.status.value_counts()[1] /
              len(job_clean.status))
print(f"Baseline/Prevalence: {prevalence:.2%}")
# %% 
# Rrain tune test split
train, test = train_test_split(
    job_clean,
    train_size=0.7,
    stratify=job_clean.status
)
print(f"Training set shape: {train.shape}")
print(f"Test set shape: {test.shape}")
# %%
# And now a 50 50 tune test split
tune, test = train_test_split(
    test,
    train_size=0.5,
    stratify=test.status
)
print(f"Training set shape: {tune.shape}")
print(f"Test set shape: {test.shape}")

# %%
# Step 3
# College Data
# I think the data is able to address the problem, but I am worried that the dataset lacks columns with a strong correlation to student performance.

# Job Data
# I think the data is able to address the problem, but I am worried that since the data is small, the model might not perform too well.