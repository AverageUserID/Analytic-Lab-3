"""
Instructions:

Let's build a kNN model using the college completion data. 
The data is messy and you have a degrees of freedom problem, as in, we have too many features.  

You've done most of the hard work already, so you should be ready to move forward with building your model. 

1. Use the question/target variable you submitted and 
build a model to answer the question you created for this dataset (make sure it is a classification problem, convert if necessary). 

2. Build a kNN model to predict your target variable using 3 nearest neighbors. Make sure it is a classification problem, meaning
if needed changed the target variable.

3. Create a dataframe that includes the test target values, test predicted values, 
and test probabilities of the positive class.

4. No code question: If you adjusted the k hyperparameter what do you think would
happen to the threshold function? Would the confusion look the same at the same threshold 
levels or not? Why or why not?

5. Evaluate the results using the confusion matrix. Then "walk" through your question, summarize what 
concerns or positive elements do you have about the model as it relates to your question? 

6. Create two functions: One that cleans the data & splits into training|test and one that 
allows you to train and test the model with different k and threshold values, then use them to 
optimize your model (test your model with several k and threshold combinations). Try not to use variable names 
in the functions, but if you need to that's fine. (If you can't get the k function and threshold function to work in one
function just run them separately.) 

7. How well does the model perform? Did the interaction of the adjusted thresholds and k values help the model? Why or why not? 

8. Choose another variable as the target in the dataset and create another kNN model using the two functions you created in
step 7. 
"""


# %%
# 1.
# Question: How can we accurately predict 4 year graduation rate?
# Independent bussiness matrix: grad_100_rate

# Importing the funtion for data cleaning
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For scaling
from io import StringIO  # For reading string data as file
import requests  # For HTTP requests to download data
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn import metrics
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
    numeric_cols.remove("grad_100_percentile")
    college_clean[numeric_cols] = MinMaxScaler().fit_transform(college_clean[numeric_cols])
    category_list = list(college_clean.select_dtypes('category'))
    college_encoded = pd.get_dummies(college_clean, columns=category_list)
    college_encoded['grad_100_percentile_cat'] = pd.cut(college_encoded['grad_100_percentile'],
                                                        bins=[-0.01, 50, 80, 100],
                                                        labels=[0, 1, 2]
    )
    prevalence = (college_encoded['grad_100_percentile_cat'].value_counts(normalize=True).sort_index())
    college_encoded = college_encoded.drop(["grad_100_value", "grad_100_percentile", "grad_150_value", "grad_150_percentile"], axis = 1)
    train, test = train_test_split(
        college_encoded,
        train_size=0.7,
        stratify=college_encoded.grad_100_percentile_cat
    )
    tune, test = train_test_split(
        test,
        train_size=0.5,
        stratify=test.grad_100_percentile_cat
    )
    return train, test, tune, prevalence

train, test, tune, prevalence = CollegePreprocessing(url = "https://raw.githubusercontent.com/UVADS/DS-3021/refs/heads/main/data/cc_institution_details.csv")

# %%
# 2.
# Training model for K = 3
import random
random.seed(1984)
train.columns
# %%
# Training data
X_train = train[["student_count", "aid_value", "retain_percentile"]]
y_train = train['grad_100_percentile_cat'].values

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

train_accuracy = neigh.score(X_train, y_train)
print(f"Training Accuracy: {train_accuracy}")

# %%
# Tuning Data
X_val = tune[["student_count", "aid_value", "retain_percentile"]]
y_val = tune['grad_100_percentile_cat'].values

print(neigh.score(X_val, y_val))
# %%
# Testing Data
X_test = test[["student_count", "aid_value", "retain_percentile"]]
y_test = test['grad_100_percentile_cat'].values

print(neigh.score(X_test, y_test))
print(f"Prevalence: {prevalence}")
# %%
# 3.
# Since we are not doing any fine tuning yet, I will just use validation set
# And since I have three classes, I will just say that the desired class is 2 out of the possible class 0, 1, and 2
y_val_pred = neigh.predict(X_val)

probs = neigh.predict_proba(X_val)
class_index = list(neigh.classes_).index(2)
positive_probs = probs[:, class_index]

dataframe = pd.DataFrame({
    "test_target_value": y_val,
    "test_predicted_value": y_val_pred,
    "test_probability_class_2": positive_probs
})
dataframe
# %%
# 4.
# Changing the value of K alters the predicted probability estimates because KNN probabilities are computed as the fraction of neighbors belonging to a class. 
# When K changes, the weight of each vote gets smaller and smaller and the probability becomes less rigid.
# Even if the classification threshold remains fixed, some observations may move above or below that threshold. 
# Therefore, the confusion matrix would generally not remain the same even for the same threshold level.
# %%
# 5. 
cm = confusion_matrix(y_val, y_val_pred, labels=neigh.classes_)
disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=neigh.classes_)  
disp.plot()
plt.show()
# The model seems decent at correctly predicting the 0 category, but struggles with accuracy in predicting 1 and 2, the latter being the desired class
# Question: How can we accurately predict 4 year graduation rate?
# Since I just choose three columns that I intuitively think would work well as predictors, the fact that the model is not just plain terrible
# seemed to suggest that the question could be answered with the data at hand, it is just that more optimization could be done.

# %%
# 6.
# I will just modify my target column to be binary for simplicity sake
def preprocess_and_split(url, train_size=0.7, random_state=42, target = "grad_100_percentile"):
    df = pd.read_csv(url)

    df["hbcu"] = df["hbcu"].apply(lambda x: 1 if x == "X" else 0)
    df["flagship"] = df["flagship"].apply(lambda x: 1 if x == "X" else 0)

    df = df.dropna(axis=1, thresh=int(0.8 * len(df)))
    df = df.dropna()

    cutoff_75 = df[target].quantile(0.75)
    df["target_cat"] = (df[target] >= cutoff_75).astype(int)
    prevalence = df["target_cat"].value_counts(normalize=True).sort_index()

    predictors = ["student_count", "aid_value", "retain_percentile"]
    keep_cols = predictors + ["target_cat"]

    df = df[keep_cols].copy()
    scaler = MinMaxScaler()
    df[predictors] = scaler.fit_transform(df[predictors])

    train_df, test_df = train_test_split(
        df,
        train_size=train_size,
        random_state=random_state,
        stratify=df["target_cat"]
    )

    X_train = train_df[predictors]
    y_train = train_df["target_cat"]

    X_test = test_df[predictors]
    y_test = test_df["target_cat"]

    return X_train, X_test, y_train, y_test, prevalence
# %%
from sklearn.metrics import confusion_matrix, accuracy_score

def adjust_thres(pred_probs, threshold, y_true):
    # your logic: 1 if p > threshold else 0
    y_pred = np.array([1 if p > threshold else 0 for p in pred_probs], dtype=int)
    cm = confusion_matrix(y_true, y_pred)
    return y_pred, cm

def knn_k_threshold_search_binary(
    X_train, y_train, X_val, y_val,
    prevalence=None,                 
    k_values=(3,5,7,9,11,15,21),
    thresholds=(0.3,0.4,0.5,0.6,0.7),
    sort_by="accuracy"
):
    y_val_arr = pd.Series(y_val).astype(int).to_numpy()

    prevalence_pos = None
    if prevalence is not None:
        # handle cases where label "1" might not appear
        prevalence_pos = float(prevalence.get(1, np.nan))

    rows = []
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)

        classes = list(model.classes_)
        if 1 not in classes:
            continue

        probs = model.predict_proba(X_val)
        pos_probs = probs[:, classes.index(1)]

        for th in thresholds:
            y_pred, cm = adjust_thres(pos_probs, th, y_val_arr)

            rows.append({
                "k": k,
                "threshold": th,
                "prevalence_pos": prevalence_pos,
                "accuracy": accuracy_score(y_val_arr, y_pred),
            })

    results_df = pd.DataFrame(rows).sort_values(by=sort_by, ascending=False).reset_index(drop=True)
    best_row = results_df.iloc[0].to_dict() if len(results_df) else None
    return best_row, results_df
# %%
X_train, X_test, y_train, y_test, prevalence = preprocess_and_split("https://raw.githubusercontent.com/UVADS/DS-3021/refs/heads/main/data/cc_institution_details.csv", train_size=0.7, random_state=42)
best, results = knn_k_threshold_search_binary(
    X_train, y_train, X_val, y_val,
    prevalence = prevalence,
    k_values=[3,5,7,9,11,15,21],
    thresholds=[0.3,0.4,0.5,0.6,0.7],
    sort_by="accuracy"
)
best
results.head(10)

# %%
# 7.
# The model did alright, the accuracy was significantly higher than prevalence.
# Lookin at the table, iterating over the function with different k and threshold did help the model
# But the improvement does not seem to be that significant, most likely because the predictor columns were not as correlated with the target as they could be
# %%
# 8.
# I will set the new target to be "carnegie_ct"
X_train, X_test, y_train, y_test, prevalence = preprocess_and_split("https://raw.githubusercontent.com/UVADS/DS-3021/refs/heads/main/data/cc_institution_details.csv", train_size=0.7, random_state=42, target = "carnegie_ct")
# %%
best, results = knn_k_threshold_search_binary(
    X_train, y_train, X_val, y_val,
    prevalence = prevalence,
    k_values=[3,5,7,9,11,15,21],
    thresholds=[0.3,0.4,0.5,0.6,0.7],
    sort_by="accuracy"
)
best
results.head(10)
# %%
