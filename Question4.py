# %%
def Library():
    import pandas as pd  # For data manipulation and analysis
    import numpy as np  # For numerical operations
    import matplotlib.pyplot as plt  # For data visualization
    from sklearn.model_selection import train_test_split  # For splitting data
    from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For scaling
    from io import StringIO  # For reading string data as file
    import requests  # For HTTP requests to download data
# %%
def CollegePreprocessing(
        url):
    college = pd.read_csv(url)
    college["hbcu"] = college["hbcu"].apply(lambda x : 1 if x == "X" else 0)
    college["flagship"] = college["flagship"].apply(lambda x : 1 if x == "X" else 0)
    college_clean = college.dropna(axis=1, thresh=int(0.8 * len(college)))
    college_clean = college_clean.dropna()
    college_clean = college_clean.drop(["chronname", "city", "state", "basic", "site", "similar", "counted_pct"], axis = 1)
    cols = college_clean.columns[college_clean.dtypes == "str"]
    college_clean[cols] = college_clean[cols].astype('category')
    college_clean = college_clean.drop(["index", "unitid", "long_x", "lat_y"], axis = 1)
    numeric_cols = list(college_clean.select_dtypes('number'))
    college_clean[numeric_cols] = MinMaxScaler().fit_transform(college_clean[numeric_cols])
    category_list = list(college_clean.select_dtypes('category'))
    college_encoded = pd.get_dummies(college_clean, columns=category_list)
    college_encoded['grad_100_percentile_f'] = pd.cut(college_encoded.grad_100_percentile,
                                                      bins=[-1, 0.43, 1],
                                                      labels=[0, 1])
    prevalence = (college_encoded.grad_100_percentile_f.value_counts()[1] /
                  len(college_encoded.grad_100_percentile_f))
    train, test = train_test_split(
        college_encoded,
        train_size=0.7,
        stratify=college_encoded.grad_100_percentile_f
    )
    tune, test = train_test_split(
        test,
        train_size=0.5,
        stratify=test.grad_100_percentile_f
    )
    return prevalence, train, test, tune
# %%
def JobPreprocessing(
        url):
    job = pd.read_csv("https://raw.githubusercontent.com/DG1606/CMS-R-2020/master/Placement_Data_Full_Class.csv")
    cat_cols = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation"]
    needstand_cols = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p", "salary"]
    job[cat_cols] = job[cat_cols].astype('category')
    job[needstand_cols] = MinMaxScaler().fit_transform(job[needstand_cols])
    job["status"] = job["status"].apply(lambda x: 1 if x == "Placed" else 0)
    job_list = list(job.select_dtypes('category'))
    job_encoded = pd.get_dummies(job, columns=job_list)
    job_clean = job_encoded.drop(['sl_no'], axis=1)
    job_clean["salary"] = job_clean["salary"].apply(lambda x : 0 if pd.isna(x) else x)
    prevalence = (job_clean.status.value_counts()[1] /
              len(job_clean.status))
    train, test = train_test_split(
        job_clean,
        train_size=0.7,
        stratify=job_clean.status
    )
    tune, test = train_test_split(
        test,
        train_size=0.5,
        stratify=test.status
    )
    return prevalence, train, test, tune