---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.10.11
  nbformat: 4
  nbformat_minor: 5
---

::: {#e66d097b .cell .markdown}
# Fictional Telecommunications Data Analysis

The dataset covers data from a fictional telco company. The records in
this dataset pertain to customer demographics information ranging from
the customer\'s attributes, location, personal status, financial gain,
and service usage.

This analysis aims to identify churn patterns, which pertain customers
cancelling their subscription, that may appear and the reason as to why.
This analysis also aims to identify the general interest of customers
based on their demographics.

The insights derived from this analysis may help in understanding how a
customer uses the telecommunication company\'s services. It may also
help in identifying ways in promoting and marketing to improve customer
rate and customer retention.

At the end of this analysis, a Logistic Regression model is applied to
answer these questions.

Additionally, a simple model will also be built to predict the occurence
of rain, trained using the dataset.
:::

::: {#09b0afde .cell .markdown}
## Data Analysis Objectives
:::

::: {#ff78636d .cell .markdown}
This project aims to satisfy the following objectives:

-   Preprocess the dataset to be applicable for a machine learning model
-   Identify factors that are driving customer churn
-   Identify factors that are important in maintaining and further
    improving customer retention.
-   Develop a Logistic Regression model to predict customers that are at
    risk of churning
:::

::: {#1592030c .cell .markdown}
## Data Exploration
:::

::: {#a969886d .cell .markdown}
Before we begin, let us first import the dataset and take a look before
proceeding.
:::

::: {#8a9d6edc .cell .code execution_count="134"}
``` python

from datasets import load_dataset 
ds = load_dataset("aai510-group1/telco-customer-churn")

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
```
:::

::: {#1f07bb8f .cell .markdown}
As seen below, the dataset is already split between a 60-20-20
train-test-validation split. This is particularly common for a dataset
with a relatively small overall size.

This dataset contains a total of 7043 records, hence, a split of
60-20-20 is perfectly reasonable.

Source:
<https://towardsdatascience.com/model-validation-techniques-explained-a-visual-guide-with-code-examples-eb13bbdc8f88/>
:::

::: {#7b3d8785 .cell .code execution_count="25"}
``` python
# Takes the entire row count of the telecommunications dataset.
total = sum(len(dataset) for dataset in ds.values())

# Displays the row count and percentage of each split in the dataset.
for split, dataset in ds.items():
    percentage = len(dataset) / total * 100
    print(f"{split}: {len(dataset):,} rows ({percentage:.2f}%)")
    
print("Total Rows:", total)
```

::: {.output .stream .stdout}
    train: 4,225 rows (59.99%)
    validation: 1,409 rows (20.01%)
    test: 1,409 rows (20.01%)
    Total Rows: 7043
:::
:::

::: {#d25838b5 .cell .markdown}
We can convert the dataset into a Pandas datatype for proper display and
convenience.
:::

::: {#85d62c75 .cell .code execution_count="55"}
``` python
import pandas as pd

train_df = ds["train"].to_pandas()
test_df = ds["test"].to_pandas()
validation_df = ds["validation"].to_pandas()
```
:::

::: {#61af284f .cell .markdown}
Below is a showcase of the training dataset\'s first 5 records. As we
can see from below, the dataset contains 52 columns, mostly numerical,
with categorical data such as \'City\', \'Churn Category\', and
\'Contract\'.
:::

::: {#407f8202 .cell .code execution_count="56"}
``` python
# Pandas for easier visualization in a tabular format.
train_df.head(5)
```

::: {.output .execute_result execution_count="56"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Avg Monthly GB Download</th>
      <th>Avg Monthly Long Distance Charges</th>
      <th>Churn</th>
      <th>Churn Category</th>
      <th>Churn Reason</th>
      <th>Churn Score</th>
      <th>City</th>
      <th>CLTV</th>
      <th>Contract</th>
      <th>...</th>
      <th>Streaming TV</th>
      <th>Tenure in Months</th>
      <th>Total Charges</th>
      <th>Total Extra Data Charges</th>
      <th>Total Long Distance Charges</th>
      <th>Total Refunds</th>
      <th>Total Revenue</th>
      <th>Under 30</th>
      <th>Unlimited Data</th>
      <th>Zip Code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>72</td>
      <td>4</td>
      <td>19.44</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>51</td>
      <td>San Mateo</td>
      <td>4849</td>
      <td>Two Year</td>
      <td>...</td>
      <td>0</td>
      <td>25</td>
      <td>2191.15</td>
      <td>0</td>
      <td>486.00</td>
      <td>0.0</td>
      <td>2677.15</td>
      <td>0</td>
      <td>1</td>
      <td>94403</td>
    </tr>
    <tr>
      <th>1</th>
      <td>27</td>
      <td>59</td>
      <td>45.62</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>27</td>
      <td>Sutter Creek</td>
      <td>3715</td>
      <td>Month-to-Month</td>
      <td>...</td>
      <td>1</td>
      <td>35</td>
      <td>3418.20</td>
      <td>0</td>
      <td>1596.70</td>
      <td>0.0</td>
      <td>5014.90</td>
      <td>1</td>
      <td>1</td>
      <td>95685</td>
    </tr>
    <tr>
      <th>2</th>
      <td>59</td>
      <td>0</td>
      <td>16.07</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>59</td>
      <td>Santa Cruz</td>
      <td>5092</td>
      <td>Month-to-Month</td>
      <td>...</td>
      <td>0</td>
      <td>46</td>
      <td>851.20</td>
      <td>0</td>
      <td>739.22</td>
      <td>0.0</td>
      <td>1590.42</td>
      <td>0</td>
      <td>0</td>
      <td>95064</td>
    </tr>
    <tr>
      <th>3</th>
      <td>25</td>
      <td>27</td>
      <td>0.00</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>49</td>
      <td>Brea</td>
      <td>2068</td>
      <td>One Year</td>
      <td>...</td>
      <td>0</td>
      <td>27</td>
      <td>1246.40</td>
      <td>30</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1276.40</td>
      <td>1</td>
      <td>0</td>
      <td>92823</td>
    </tr>
    <tr>
      <th>4</th>
      <td>31</td>
      <td>21</td>
      <td>17.22</td>
      <td>1</td>
      <td>Dissatisfaction</td>
      <td>Network reliability</td>
      <td>88</td>
      <td>San Jose</td>
      <td>4026</td>
      <td>One Year</td>
      <td>...</td>
      <td>0</td>
      <td>58</td>
      <td>3563.80</td>
      <td>0</td>
      <td>998.76</td>
      <td>0.0</td>
      <td>4562.56</td>
      <td>0</td>
      <td>1</td>
      <td>95117</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 52 columns</p>
</div>
```
:::
:::

::: {#8c8633f6 .cell .markdown}
Below, we can also check out the features or columns of the dataset in a
more compressed manner. The info() methods allows for a summarized
version, displaying the columns, their type, how many non-null values
and null values each column has.
:::

::: {#d0ef8fe2 .cell .code execution_count="57"}
``` python
train_df.info()
```

::: {.output .stream .stdout}
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4225 entries, 0 to 4224
    Data columns (total 52 columns):
     #   Column                             Non-Null Count  Dtype  
    ---  ------                             --------------  -----  
     0   Age                                4225 non-null   int64  
     1   Avg Monthly GB Download            4225 non-null   int64  
     2   Avg Monthly Long Distance Charges  4225 non-null   float64
     3   Churn                              4225 non-null   int64  
     4   Churn Category                     1121 non-null   object 
     5   Churn Reason                       1121 non-null   object 
     6   Churn Score                        4225 non-null   int64  
     7   City                               4225 non-null   object 
     8   CLTV                               4225 non-null   int64  
     9   Contract                           4225 non-null   object 
     10  Country                            4225 non-null   object 
     11  Customer ID                        4225 non-null   object 
     12  Customer Status                    4225 non-null   object 
     13  Dependents                         4225 non-null   int64  
     14  Device Protection Plan             4225 non-null   int64  
     15  Gender                             4225 non-null   object 
     16  Internet Service                   4225 non-null   int64  
     17  Internet Type                      3339 non-null   object 
     18  Lat Long                           4225 non-null   object 
     19  Latitude                           4225 non-null   float64
     20  Longitude                          4225 non-null   float64
     21  Married                            4225 non-null   int64  
     22  Monthly Charge                     4225 non-null   float64
     23  Multiple Lines                     4225 non-null   int64  
     24  Number of Dependents               4225 non-null   int64  
     25  Number of Referrals                4225 non-null   int64  
     26  Offer                              1901 non-null   object 
     27  Online Backup                      4225 non-null   int64  
     28  Online Security                    4225 non-null   int64  
     29  Paperless Billing                  4225 non-null   int64  
     30  Partner                            4225 non-null   int64  
     31  Payment Method                     4225 non-null   object 
     32  Phone Service                      4225 non-null   int64  
     33  Population                         4225 non-null   int64  
     34  Premium Tech Support               4225 non-null   int64  
     35  Quarter                            4225 non-null   object 
     36  Referred a Friend                  4225 non-null   int64  
     37  Satisfaction Score                 4225 non-null   int64  
     38  Senior Citizen                     4225 non-null   int64  
     39  State                              4225 non-null   object 
     40  Streaming Movies                   4225 non-null   int64  
     41  Streaming Music                    4225 non-null   int64  
     42  Streaming TV                       4225 non-null   int64  
     43  Tenure in Months                   4225 non-null   int64  
     44  Total Charges                      4225 non-null   float64
     45  Total Extra Data Charges           4225 non-null   int64  
     46  Total Long Distance Charges        4225 non-null   float64
     47  Total Refunds                      4225 non-null   float64
     48  Total Revenue                      4225 non-null   float64
     49  Under 30                           4225 non-null   int64  
     50  Unlimited Data                     4225 non-null   int64  
     51  Zip Code                           4225 non-null   object 
    dtypes: float64(8), int64(29), object(15)
    memory usage: 1.7+ MB
:::
:::

::: {#a4c8dc66 .cell .markdown}
Below, we can check out important demographic information for the
customers. This gives us important context on the scenario itself.

As we can see, the dataset all surround the state of California, United
States. There is almost an even 50-50 split in gender, with 2124 male
and 2101 female customers. Meanwhile, we have 1121 churned customers in
the training split.
:::

::: {#7f4d0c01 .cell .code execution_count="83"}
``` python
identifier_cols = ['City', 'Country', 'State', 'Gender', 'Customer Status', 'Contract', 'Internet Type']

for column in identifier_cols:
    print(f"\n===== {column} =====")
    print(train_df[column].value_counts(dropna=False))
```

::: {.output .stream .stdout}

    ===== City =====
    City
    Los Angeles      175
    San Diego        168
    San Francisco     61
    San Jose          61
    Sacramento        60
                    ... 
    San Bruno          1
    Salida             1
    Lewiston           1
    Casmalia           1
    Sloughhouse        1
    Name: count, Length: 1085, dtype: int64

    ===== Country =====
    Country
    United States    4225
    Name: count, dtype: int64

    ===== State =====
    State
    California    4225
    Name: count, dtype: int64

    ===== Gender =====
    Gender
    Male      2124
    Female    2101
    Name: count, dtype: int64

    ===== Customer Status =====
    Customer Status
    Stayed     2832
    Churned    1121
    Joined      272
    Name: count, dtype: int64

    ===== Contract =====
    Contract
    Month-to-Month    2193
    Two Year          1128
    One Year           904
    Name: count, dtype: int64

    ===== Internet Type =====
    Internet Type
    Fiber Optic    1829
    DSL            1008
    No Internet     886
    Cable           502
    Name: count, dtype: int64
:::
:::

::: {#b85fe0aa .cell .markdown}
I also would like to check how the dataset is split based on how many
customers Churned. Using the *Customer Status* column, we can identify
that the dataset has an equivalent class distribution across the three
splits.

This means that the three splits are satisfactory. It is slightly
**imbalanced** with a 74-26 ratio of Retention-Churned, but it is not
heavily imabalanced and can still be considered fairly balanced. This is
important since models aim to maximize accuracy. If a model is skewed
towards a particular class, it will aim to mostly guess towards that
class, leading to its prediction being incorrect.

A balanced dataset prevents bias and ensures we are able to use this
model in a real world setting.
:::

::: {#1aa1ef41 .cell .code execution_count="87"}
``` python
print(train_df['Customer Status'].value_counts(dropna=False))
print(test_df['Customer Status'].value_counts(dropna=False))
print(validation_df['Customer Status'].value_counts(dropna=False))
```

::: {.output .stream .stdout}
    Train: 26.53% Churned
    Test: 26.54% Churned
    Validation: 26.54% Churned
:::
:::

::: {#23c1e950 .cell .markdown}
Firstly, I want to look into the *Satisfaction Score*, as that may have
an indication on the *Churn* status of customers. I looked into if the
binary column Churn and compared with the Satisfactory Score to check
the values, and have observed that most customers who churned are those
with a satisfactory score of 1-3, while those that didn\'t were on 3-5.
:::

::: {#57dc267d .cell .code execution_count="146"}
``` python
satisfaction_churn = (
    train_df.groupby('Satisfaction Score')['Churn']
      .value_counts(normalize=True)
      .unstack(fill_value=0)
)

# Rename columns for readability
satisfaction_churn = satisfaction_churn.rename(
    columns={0: 'No Churn', 1: 'Churn'}
)

# Plot both
satisfaction_churn.plot(
    kind='bar',
    figsize=(9, 3)
)

plt.title('Churn Rate by Satisfaction Score')
plt.xlabel('Satisfaction Score')
plt.ylabel('Proportion of Customers')
plt.xticks(rotation=0)
plt.ylim(0, 1)
plt.legend(title='Customer Status')
plt.show()
```

::: {.output .display_data}
![](vertopal_d621a775d8a2441fab5a019668e3b8dd/7ec40c9865a6e07ae7c007f7c2be43ff5eeeead5.png)
:::
:::

::: {#1683a3a0 .cell .code execution_count="140"}
``` python
train_df['Tenure in Months']
```

::: {.output .execute_result execution_count="140"}
    0       25
    1       35
    2       46
    3       27
    4       58
            ..
    4220     1
    4221    22
    4222    18
    4223    57
    4224    17
    Name: Tenure in Months, Length: 4225, dtype: int64
:::
:::

::: {#15efbaa6 .cell .markdown}
Next, I took a look into the Churn rate as the time of service went on,
using *Tenure in Months*. From here, we can observe that those who
haven\'t used the service yet for a year are the most likely ones to
churn.

This trend continues in a downward manner, where the more tenured
customers are more likely to stay and retain their use of the service.
:::

::: {#b704a234 .cell .code execution_count="144"}
``` python
train_copy = train_df.copy()
train_copy['Tenure Group'] = pd.cut(
    train_copy['Tenure in Months'],
    bins=[0, 12, 24, 36, 48, 60, 72],
    labels=['0-12', '13-24', '25-36', '37-48', '49-60', '61-72']
)

tenure_churn = (
    train_copy.groupby('Tenure Group', observed=False)['Churn']
      .apply(lambda x: (x == 1).mean())
)

tenure_churn.plot(
    kind='bar',
    figsize=(5, 3)
)

plt.title('Churn Rate by Tenure')
plt.xlabel('Tenure (Months)')
plt.ylabel('Churn Rate')
plt.xticks(rotation=0)
plt.show()
```

::: {.output .display_data}
![](vertopal_d621a775d8a2441fab5a019668e3b8dd/407f208b7bab3741f058fa85446f21374714608c.png)
:::
:::

::: {#149ae339 .cell .markdown}
Next, we can take a look into how customers churn based on their
contract. There are only 3 types of contracts here, Month-to-Month, One
Year, Two Year.

From the graph below, customers with shorter-term contracts tend to
churn while customers with longer-term contracts are expected to stay.
This is a common event especially for users who want to try using the
product for a month before committing to it and extending their
contract.
:::

::: {#2a195f24 .cell .code execution_count="150"}
``` python
churn_rate = (
    train_copy.groupby('Contract')['Churn']
      .value_counts(normalize=True)
      .unstack()
)

churn_rate[1].sort_values(ascending=False).plot(
    kind='bar',
    figsize=(7, 3)
)

plt.title('Churn Rate by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Churn Rate')
plt.xticks(rotation=0)
plt.show()
```

::: {.output .display_data}
![](vertopal_d621a775d8a2441fab5a019668e3b8dd/7bb845a6c7513bf6eb01fc2af3f0620e8f4abce6.png)
:::
:::

::: {#88c73abd .cell .markdown}
Now that exploration has been implemented, based on the features
observed, I am hypothesizing that the following features will be strong
features:

-   Contract - Customers on shorter contracts have less commitment and
    may be more likely to leave to try other products.
-   Tenure in Months - Newer customers may have a higher likelihood of
    churning, while long-term customers are committed.
-   Satisfaction Score - Unhappy customers are more likely to leave.
-   Monthly Charge - Higher monthly costs may increase the likelihood of
    customers switching to competitors and leaving.
-   Number of Referrals and Number of Dependents - Customers who
    actively refer others may be more engaged with the service.
:::

::: {#55b71a00 .cell .markdown}
## Data Preprocessing
:::

::: {#c8244af9 .cell .markdown}
### Handling Missing Values
:::

::: {#c405dbb7 .cell .markdown}
Next, we can also observe from the .info() method output that there are
columns which are have a non-null value different from the rest. This
indicates that these features have a \'null\' or \'None\' value. Which
we will be handling next.
:::

::: {#a88fe3a4 .cell .markdown}
Let us first begin by trying to check out which columns in our dataset
have these null values across the three splits.
:::

::: {#55f19aa2 .cell .code execution_count="58"}
``` python
def check_null_columns(dataset: pd.DataFrame) -> None:
    """Identify columns containing null values in a DataFrame.

    Args:
        dataset (pd.DataFrame): The DataFrame used for identifying null columns.
    """

    null_columns = dataset.columns[dataset.isnull().any()]

    if len(null_columns) > 0:
        print("Columns with null values:")
        for column in null_columns:
            print(f"- {column}: {dataset[column].isnull().sum()} null values")
    else:
        print("No columns with null values.")


for split, dataset in {
    "Train": train_df,
    "Test": test_df,
    "Validation": validation_df
}.items():
    print(f"\n{split} Split")
    check_null_columns(dataset)
```

::: {.output .stream .stdout}

    Train Split
    Columns with null values:
    - Churn Category: 3104 null values
    - Churn Reason: 3104 null values
    - Internet Type: 886 null values
    - Offer: 2324 null values

    Test Split
    Columns with null values:
    - Churn Category: 1035 null values
    - Churn Reason: 1035 null values
    - Internet Type: 308 null values
    - Offer: 762 null values

    Validation Split
    Columns with null values:
    - Churn Category: 1035 null values
    - Churn Reason: 1035 null values
    - Internet Type: 332 null values
    - Offer: 791 null values
:::
:::

::: {#177b7a23 .cell .markdown}
As we can see, all 3 datasets have a commonality when it came to the
missing values. They all stemmed from the same 3 columns:

-   Churn Category
-   Churn Reason
-   Internet Type
-   Offer

To further contextualize these four columns, let us take a sample look
into them.
:::

::: {#b321097d .cell .code execution_count="59"}
``` python
# Our columns with null values.
columns = ["Churn Category", "Churn Reason", "Internet Type", "Offer"]

df = train_df[columns]

df.head(10)
```

::: {.output .execute_result execution_count="59"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Churn Category</th>
      <th>Churn Reason</th>
      <th>Internet Type</th>
      <th>Offer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>None</td>
      <td>None</td>
      <td>Fiber Optic</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>None</td>
      <td>None</td>
      <td>Fiber Optic</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>None</td>
      <td>None</td>
      <td>DSL</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Dissatisfaction</td>
      <td>Network reliability</td>
      <td>Cable</td>
      <td>Offer B</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Competitor</td>
      <td>Competitor had better devices</td>
      <td>Cable</td>
      <td>Offer E</td>
    </tr>
    <tr>
      <th>6</th>
      <td>None</td>
      <td>None</td>
      <td>Fiber Optic</td>
      <td>None</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Price</td>
      <td>Price too high</td>
      <td>Fiber Optic</td>
      <td>Offer B</td>
    </tr>
    <tr>
      <th>8</th>
      <td>None</td>
      <td>None</td>
      <td>Fiber Optic</td>
      <td>Offer A</td>
    </tr>
    <tr>
      <th>9</th>
      <td>None</td>
      <td>None</td>
      <td>DSL</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#243013b6 .cell .markdown}
These four columns are explainable as to why they are missing. Missing
values aren\'t necessarily indicative of missing data. In this case, it
is easy to understand as to why these are missing.

-   Offer - Offer refers to the marketing offer that the customer
    accepted. If this is null, the customer simply had no offer given to
    them or they didn\'t accept an offer.
-   Internet Type - This is simply the type of internet service availed
    by the customer. A null value means no internet service was recorded
    for the customer.

On the other hand, the former two columns, *Churn Category* and *Churn
Reason* are very important in our dataset. These are closely related to
the *Churn* column, and together, these are the indicators that specify
if a customer has churned/left the service.
:::

::: {#4d5dbd98 .cell .code execution_count="61"}
``` python
# Columns to Replace
columns = ["Churn Category", "Churn Reason", "Internet Type", "Offer"]

# Fill null values across the three datasets with appropriate values.
for df in [train_df, test_df, validation_df]:
    df["Offer"] = df["Offer"].fillna("No Offer")
    df["Internet Type"] = df["Internet Type"].fillna("No Internet")
    df["Churn Category"] = df["Churn Category"].fillna("Did Not Churn")
    df["Churn Reason"] = df["Churn Reason"].fillna("Did Not Churn")
```
:::

::: {#178eb20d .cell .markdown}
And after running, we can use the check_null_columns that we defined to
see if there are any more columns that are null. As we can see, we are
now ready to proceed.
:::

::: {#a8574b4f .cell .code execution_count="62"}
``` python

for split, dataset in {
    "Train": train_df,
    "Test": test_df,
    "Validation": validation_df
}.items():
    print(f"\n{split} Split")
    check_null_columns(dataset)## Outlier Detection
```

::: {.output .stream .stdout}

    Train Split
    No columns with null values.

    Test Split
    No columns with null values.

    Validation Split
    No columns with null values.
:::
:::

::: {#9a0e812b .cell .markdown}
### Handling Outliers
:::

::: {#9666f52f .cell .markdown}
#### Categorical Outliers
:::

::: {#49c90e0c .cell .markdown}
Now is the time to see if there are any columns that go past beyond
their logical point or become outliers. First, we will look into
categorical fields for any outliers.
:::

::: {#109f7663 .cell .markdown}
As seen below, there are no erroneous categories or any categories that
are different from their respective column. The only categories that had
a large number were *Lat Long* and *Customer ID* since they are both of
Object type, but each feather very unique values.

In conclusion, we can move on to numerical values.
:::

::: {#99cd7e97 .cell .code execution_count="80"}
``` python
categorical_cols = train_df.select_dtypes(include="object").columns

for column in categorical_cols:
    print(f"\n===== {column} =====")
    print(train_df[column].value_counts(dropna=False))
```

::: {.output .stream .stdout}

    ===== Churn Category =====
    Churn Category
    Did Not Churn      3104
    Competitor          486
    Attitude            200
    Dissatisfaction     171
    Other               134
    Price               130
    Name: count, dtype: int64

    ===== Churn Reason =====
    Churn Reason
    Did Not Churn                                3104
    Competitor had better devices                 188
    Competitor made better offer                  187
    Attitude of support person                    136
    Don't know                                     87
    Attitude of service provider                   64
    Competitor offered more data                   57
    Competitor offered higher download speeds      54
    Network reliability                            46
    Price too high                                 44
    Product dissatisfaction                        42
    Long distance charges                          39
    Service dissatisfaction                        32
    Moved                                          32
    Extra data charges                             27
    Limited range of services                      25
    Lack of affordable download/upload speed       20
    Lack of self-service on Website                17
    Poor expertise of online support               14
    Deceased                                        6
    Poor expertise of phone support                 4
    Name: count, dtype: int64

    ===== City =====
    City
    Los Angeles      175
    San Diego        168
    San Francisco     61
    San Jose          61
    Sacramento        60
                    ... 
    San Bruno          1
    Salida             1
    Lewiston           1
    Casmalia           1
    Sloughhouse        1
    Name: count, Length: 1085, dtype: int64

    ===== Contract =====
    Contract
    Month-to-Month    2193
    Two Year          1128
    One Year           904
    Name: count, dtype: int64

    ===== Country =====
    Country
    United States    4225
    Name: count, dtype: int64

    ===== Customer ID =====
    Customer ID
    4526-ZJJTM    1
    2920-RNCEZ    1
    6859-QNXIQ    1
    0129-KPTWJ    1
    8390-FESFV    1
                 ..
    1689-MRZQR    1
    8735-IJJEG    1
    5186-EJEGL    1
    0916-QOFDP    1
    7049-GKVZY    1
    Name: count, Length: 4225, dtype: int64

    ===== Customer Status =====
    Customer Status
    Stayed     2832
    Churned    1121
    Joined      272
    Name: count, dtype: int64

    ===== Gender =====
    Gender
    Male      2124
    Female    2101
    Name: count, dtype: int64

    ===== Internet Type =====
    Internet Type
    Fiber Optic    1829
    DSL            1008
    No Internet     886
    Cable           502
    Name: count, dtype: int64

    ===== Lat Long =====
    Lat Long
    33.141265, -116.967221    24
    32.825086, -117.199424    21
    33.362575, -117.299644    21
    32.886925, -117.152162    20
    33.507255, -117.029473    19
                              ..
    34.866032, -120.536546     1
    38.013825, -122.039144     1
    37.791481, -121.903253     1
    33.313828, -116.940501     1
    38.470423, -121.114897     1
    Name: count, Length: 1627, dtype: int64

    ===== Offer =====
    Offer
    No Offer    2324
    Offer B      515
    Offer E      475
    Offer D      341
    Offer A      319
    Offer C      251
    Name: count, dtype: int64

    ===== Payment Method =====
    Payment Method
    Bank Withdrawal    2325
    Credit Card        1680
    Mailed Check        220
    Name: count, dtype: int64

    ===== Quarter =====
    Quarter
    Q3    4225
    Name: count, dtype: int64

    ===== State =====
    State
    California    4225
    Name: count, dtype: int64

    ===== Zip Code =====
    Zip Code
    92122    24
    92027    24
    92028    21
    92117    21
    92126    20
             ..
    95127     1
    95662     1
    95312     1
    95430     1
    95683     1
    Name: count, Length: 1594, dtype: int64
:::
:::

::: {#3368393a .cell .markdown}
#### Numerical Outliers
:::

::: {#88575731 .cell .markdown}
To begin checking the numerical values if there are any outliers, I will
first split the binary columns from the numerical/continuous values.

A binary column is one where its values are 0 or 1, and this is
observable in columns such as *Churn* or *Gender*.

We can identify the columns that aren\'t binary by excluding them and
making a new list.
:::

::: {#ee9263e2 .cell .code execution_count="88"}
``` python

binary_cols = [
    column for column in train_df.columns
    if train_df[column].nunique() == 2
]

numeric_cols = train_df.select_dtypes(include="number").columns

continuous_cols = [
    column for column in numeric_cols
    if column not in binary_cols
]

print(continuous_cols)
```

::: {.output .stream .stdout}
    ['Age', 'Avg Monthly GB Download', 'Avg Monthly Long Distance Charges', 'Churn Score', 'CLTV', 'Latitude', 'Longitude', 'Monthly Charge', 'Number of Dependents', 'Number of Referrals', 'Population', 'Satisfaction Score', 'Tenure in Months', 'Total Charges', 'Total Extra Data Charges', 'Total Long Distance Charges', 'Total Refunds', 'Total Revenue']
:::
:::

::: {#4b583034 .cell .markdown}
Next, outliers usually persist if they exceed any interquartile range,
but this isn\'t an indication of outliers.

The code below takes the columns that exceed the interquartile range,
and we will be investigating those that violate that range.
:::

::: {#427b45ad .cell .markdown}
Now that we have identified that these data are likely MAR (Missing at
Random), we would still need to manage these missing records for our
analysis. We can go ahead and manage the null values across all three
datasets with the following:

-   Offer - No Offer
-   Internet - No Internet
-   Churn Category - Did Not Churn
-   Churn Reason - Did Not Churn
:::

::: {#1b2c19cf .cell .code execution_count="90"}
``` python
outlier_columns =[]

# Loop through all continuous columns to identify outliers using the IQR method.
# The columns were identified in the previous code block.
for column in continuous_cols:
    Q1 = train_df[column].quantile(0.25)
    Q3 = train_df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outlier_count = (
        (train_df[column] < lower) |
        (train_df[column] > upper)
    ).sum()

    if outlier_count > 0:
        print(f"{column}: {outlier_count:,} outliers")
        outlier_columns.append(column)
    
```

::: {.output .stream .stdout}
    Avg Monthly GB Download: 211 outliers
    Number of Dependents: 985 outliers
    Number of Referrals: 413 outliers
    Population: 33 outliers
    Satisfaction Score: 549 outliers
    Total Extra Data Charges: 450 outliers
    Total Long Distance Charges: 108 outliers
    Total Refunds: 320 outliers
    Total Revenue: 14 outliers
:::
:::

::: {#ff77cbcd .cell .markdown}
We can already knock off some columns that were tagged as outliers, such
as *Satisfaction Score*, *Population*.

Population was identified as having 33 statistical outliers using
IQRmethod. However, these values should not be considered erroneous
because Population represents the population associated with a
customer\'s ZIP code rather than the customer\'s individual
characteristics. Multiple customers living within the same ZIP code can
therefore legitimately have the same population value.
:::

::: {#4b81bfa9 .cell .markdown}
Similarly, *Satisfaction Score* produced 549 records as outliers.
However, this variable is a discrete rating on a scale of 1 to 5, rather
than a continuous measurement.

This extends to *Number of Dependents* and *Number of Dependents*.
Despite not having a scale, are discrete values that range from 0 to 8
and 0 to 11 respectively as seen below.
:::

::: {#3c44265e .cell .code execution_count="107"}
``` python
print(train_df["Satisfaction Score"].value_counts().sort_index())

print(train_df["Number of Dependents"].value_counts().sort_index())

print(train_df["Number of Referrals"].value_counts().sort_index())
```

::: {.output .stream .stdout}
    Satisfaction Score
    1     549
    2     314
    3    1622
    4    1044
    5     696
    Name: count, dtype: int64
    Number of Dependents
    0    3240
    1     337
    2     315
    3     316
    4       7
    5       8
    6       1
    8       1
    Name: count, dtype: int64
    Number of Referrals
    0     2282
    1      642
    2      136
    3      158
    4      137
    5      160
    6      130
    7      167
    8      116
    9      149
    10     146
    11       2
    Name: count, dtype: int64
:::
:::

::: {#fbb4e8fb .cell .markdown}
Next, *Avg Monthly GB Download* had 211 observations classified as
outliers. However, these observations have a realistic range of
approximately 0--85 GB per month and can represent a legitimate
customers\' data usage.
:::

::: {#407466c3 .cell .code execution_count="118"}
``` python
# Prints the minimum and maximum of Avg Monthly GB Download
print(f"Avg Monthly GB Download: min = {train_df['Avg Monthly GB Download'].min()}, max = {train_df['Avg Monthly GB Download'].max()}")
```

::: {.output .stream .stdout}
    Avg Monthly GB Download: min = 0, max = 85
:::
:::

::: {#d6a05054 .cell .markdown}
Next, we can observe the remaining continuous variables which are the
following:

-   Total Extra Data Charges
-   Total Long Distance Charges
-   Total Refunds
-   Total Revenue

To observe these, I graphed a histogram plot of the values shown below.
The bar graph is binned to group data values into 10 groups.
:::

::: {#ff1915f0 .cell .code execution_count="132"}
``` python
import matplotlib.pyplot as plt

columns = [
    "Total Extra Data Charges",
    "Total Long Distance Charges",
    "Total Refunds",
    "Total Revenue"
]

for column in columns:
    plt.figure(figsize=(4,4))

    train_df[column].plot(
        kind="hist",
        bins=10
    )

    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
```

::: {.output .display_data}
![](vertopal_d621a775d8a2441fab5a019668e3b8dd/768a03f993390bddbde21e6cb62f9067cc9b068c.png)
:::

::: {.output .display_data}
![](vertopal_d621a775d8a2441fab5a019668e3b8dd/4dd977cbaf63065f6254e5747827cf8bebe6bd66.png)
:::

::: {.output .display_data}
![](vertopal_d621a775d8a2441fab5a019668e3b8dd/54282c85824ef578a98c9e9a6437f4dfd97430b5.png)
:::

::: {.output .display_data}
![](vertopal_d621a775d8a2441fab5a019668e3b8dd/14065de7674a5dd660c240e87693e50848cb8790.png)
:::
:::

::: {#02c333b7 .cell .markdown}
As we can observe, there is a right skew distribution within these 4
datasets, though it is less obvious in the Total Extra Data Charges and
Total Refunds.

Still, we can observe that the right skew distribution is consistent and
there isn\'t a particular data value that heavily manipulates the
histogram. Meaning we can safely say that there are also no outliers
within these columns.
:::

::: {#77d3d6ae .cell .markdown}
### Feature Engineering
:::

::: {#80f5f01e .cell .markdown}
Before we begin, let us first lighten our workload a bit by splitting
the X and y\'s of our dataset.

This involves taking our target *Churn*, and creating the output columns
for the 3 datasets.
:::

::: {#377a4974 .cell .code execution_count="164"}
``` python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

target = 'Churn'

X_train = train_df.drop(columns=[target])
y_train = train_df[target]

X_test = test_df.drop(columns=[target])
y_test = test_df[target]

X_validation = validation_df.drop(columns=[target])
y_validation = validation_df[target]
```
:::

::: {#6ab16ca9 .cell .markdown}
Next, I believe it is time to remove the columns we don\'t need.

-   Churn Category, Churn Reason, Churn Score - This column is only
    useful after churn, it doesn\'t help contextualize why the customer
    churned
-   Customer Status - This will cause the model to have a bias since
    Customer Status describes if a customer has churned or stayed.
-   Customer ID - Unique Identifier
-   Lat Long - We already have a latitude and longtitude column for
    geographic information
-   Country, State - Since we are in California, we can remove these two
    columns.
:::

::: {#af2cd4af .cell .code execution_count="166"}
``` python
drop_columns = [
    'Churn Category',
    'Churn Reason',
    'Churn Score',
    'Customer Status',
    'Customer ID',
    'Lat Long',
    'Country',
    'State'
]

X_train = X_train.drop(columns=drop_columns, errors='ignore')
X_test = X_test.drop(columns=drop_columns, errors='ignore')
X_validation = X_validation.drop(columns=drop_columns, errors='ignore')
```
:::

::: {#a95f05b1 .cell .markdown}
#### Data Encoding
:::

::: {#57bf143f .cell .markdown}
For this dataset, I used the One-Hot Encoding method for categorical
variables such as Contract, Internet Type, and Payment Method, as they
do not have any ordinal scaling.
:::

::: {#e601a069 .cell .code execution_count="171"}
``` python
numerical_features = X_train.select_dtypes(
    include=['int64', 'float64']
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=['object', 'category']
).columns.tolist()

print("Numerical features:", numerical_features)
print("Categorical features:", categorical_features)
```

::: {.output .stream .stdout}
    Numerical features: ['Age', 'Avg Monthly GB Download', 'Avg Monthly Long Distance Charges', 'CLTV', 'Dependents', 'Device Protection Plan', 'Internet Service', 'Latitude', 'Longitude', 'Married', 'Monthly Charge', 'Multiple Lines', 'Number of Dependents', 'Number of Referrals', 'Online Backup', 'Online Security', 'Paperless Billing', 'Partner', 'Phone Service', 'Population', 'Premium Tech Support', 'Referred a Friend', 'Satisfaction Score', 'Senior Citizen', 'Streaming Movies', 'Streaming Music', 'Streaming TV', 'Tenure in Months', 'Total Charges', 'Total Extra Data Charges', 'Total Long Distance Charges', 'Total Refunds', 'Total Revenue', 'Under 30', 'Unlimited Data']
    Categorical features: ['City', 'Contract', 'Gender', 'Internet Type', 'Offer', 'Payment Method', 'Quarter', 'Zip Code']
:::
:::

::: {#fac07d15 .cell .markdown}
#### Data Normalization
:::

::: {#7fc1fde0 .cell .markdown}
We will then create the preprocessor for scaling. Scaling is important
in models as it allows for a standardized dataset. Meaning, all data
will be treated as if they are among an equal scale. This allows for
values like currency that is typically large to not heavily the
influence the model, and for lesser values like categorical values to
also be considered and have an impact on the model.

For this, we will apply **StandardScaler**.
:::

::: {#cd671eb0 .cell .code execution_count="180"}
``` python
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(
            handle_unknown='ignore',
            drop='first'
        ), categorical_features)
    ]
)
```
:::

::: {#fcb9d7c7 .cell .markdown}
Let us apply this transformation across our train, test, and validation
sets.
:::

::: {#49d34aff .cell .code execution_count="181"}
``` python
X_train_processed = preprocessor.fit_transform(X_train)



X_test_processed = preprocessor.transform(X_test)


X_validation_processed = preprocessor.transform(X_validation)

```

::: {.output .stream .stderr}
    c:\Users\user\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\preprocessing\_encoders.py:246: UserWarning: Found unknown categories in columns [0, 7] during transform. These unknown categories will be encoded as all zeros
      warnings.warn(
    c:\Users\user\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\preprocessing\_encoders.py:246: UserWarning: Found unknown categories in columns [0, 7] during transform. These unknown categories will be encoded as all zeros
      warnings.warn(
:::
:::

::: {#b9843e6d .cell .markdown}
Since we used **OneHotEncoding**, this creates multiple features that
expand our dataset. This is because one-hot encoding converts categories
into binary values. For example, the column **Internet Type** have
multiple categories (Fiber Optic, No Internet, DSL). This will create
multiple columns that can be referred to as \"Has Fiber Optic\", \"Has
DSL\", \"Has Cable\". Resulting in the final feature count of 2725.
:::

::: {#305f5957 .cell .code execution_count="183"}
``` python
print("Train Final feature count:", X_train_processed.shape[1])
print("Test Final feature count:", X_test_processed.shape[1])
print("Validation Final feature count:", X_validation_processed.shape[1])
```

::: {.output .stream .stdout}
    Train Final feature count: 2725
    Test Final feature count: 2725
    Validation Final feature count: 2725
:::
:::

::: {#b214a6d7 .cell .markdown}
The data was cleaned by removing identifiers and variables that directly
describe the churn outcome, which could give the model clues on what to
predict.

Categorical information was then converted into separate indicators so
that categories such as contract type and payment method could be
compared without having any ranks between them.

Numerical variables were then standardized so that features with large
values, such as total charges, do not affect the logistic regression
model compared with smaller-scale variables such as age.

After preprocessing, the dataset contained 2725 predictive features.
:::

::: {#5b95144a .cell .markdown}
## Model Training
:::

::: {#21d1997c .cell .markdown}
### Training Logistic Regression
:::

::: {#ab9ebc66 .cell .markdown}
We can begin with creating the baseline model with sklearn\'s
LogisticRegression with the following hyperparameters:

-   C = 1.0 - The default regularization value. It usually helps the
    model to stay simple.
-   solver = \'liblinear\' - Works good for binary logistic regression
    and can handle high-dimensional datasets.
-   max_iter = 1000 - Setting to only 1000 so that the model doesn\'t
    run for way too long and can converge after 1000 iterations.
:::

::: {#7f526e44 .cell .code execution_count="202"}
``` python
# Import LogisticRegression from sklearn.linear_model
from sklearn.linear_model import LogisticRegression

# Create a Logistic Regression model with specified hyperparameters
logreg = LogisticRegression(
    C=1.0,
    solver='liblinear',
    max_iter=1000,
)

logreg.fit(X_train_processed, y_train)

# Creating a model that is balanced
logreg_balanced = LogisticRegression(
    C=1.0,
    solver='liblinear',
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

logreg_balanced.fit(X_train_processed, y_train)
```

::: {.output .execute_result execution_count="202"}
```{=html}
<style>#sk-container-id-3 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-3 {
  color: var(--sklearn-color-text);
}

#sk-container-id-3 pre {
  padding: 0;
}

#sk-container-id-3 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-3 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-3 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-3 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-3 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-3 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-3 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-3 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-3 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-3 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-3 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-3 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-3 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-3 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-3 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-3 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-3 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-3 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-3 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-3 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-3 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-3 div.sk-label label.sk-toggleable__label,
#sk-container-id-3 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-3 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-3 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-3 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-3 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-3 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-3 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-3 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-3 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-3 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-3 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-3 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression(class_weight=&#x27;balanced&#x27;, max_iter=1000, random_state=42,
                   solver=&#x27;liblinear&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" checked><label for="sk-estimator-id-3" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>LogisticRegression</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.LogisticRegression.html">?<span>Documentation for LogisticRegression</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>LogisticRegression(class_weight=&#x27;balanced&#x27;, max_iter=1000, random_state=42,
                   solver=&#x27;liblinear&#x27;)</pre></div> </div></div></div></div>
```
:::
:::

::: {#d26c3062 .cell .markdown}
Next, we can proceed with testing the model with the processed training
and test set as is.

Accuracy simply means how many records match from the dataset that were
predicted. Meaning, how many customers that churned and didn\'t churn
were properly identified.
:::

::: {#73af4f78 .cell .code execution_count="209"}
``` python
from sklearn.metrics import accuracy_score
import pandas as pd

# The predict method is used to generate predictions for the training, validation, and test datasets.
train_pred = logreg.predict(X_train_processed)
val_pred = logreg.predict(X_validation_processed)
test_pred = logreg.predict(X_test_processed)

# We will be scoring the model's performance using accuracy as the evaluation metric. 
# The accuracy_score function from sklearn.metrics calculates the accuracy for each dataset.
accuracy_results = pd.DataFrame({
    'Dataset': ['Training', 'Validation', 'Test'],
    'Accuracy': [
        accuracy_score(y_train, train_pred),
        accuracy_score(y_validation, val_pred),
        accuracy_score(y_test, test_pred)
    ]
})

accuracy_results
```

::: {.output .execute_result execution_count="209"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dataset</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Training</td>
      <td>0.984615</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Validation</td>
      <td>0.967353</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Test</td>
      <td>0.962385</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#8eb5556d .cell .markdown}
After comparing the training, validation, and test accuracies, we can
see that they are extremely high. The relatively small difference
between the training and validation/test scores indicates that the model
does not show any overfitting nor underfitting for that matter. This
indicates that the **baseline model** generalizes reasonably well.
:::

::: {#d48db6cb .cell .markdown}
## Model Evaluation
:::

::: {#b940ea45 .cell .markdown}
We can proceed with evaluating the model. The most common metrics used
for a model like Logistic Regression are the Precision and Recall.

We can see that the Precision of 97% and 94% for No Churn and Churn
respectively means that the model can correctly predict those who leave
and those who don\'t. Of all the customers that were predicted to have
churned, 94% of those predictions were correct.

For Recall of 92% on Churn, of all the customers who churned, the model
guessed 92%. This is the reason **Recall is more important**, since the
primary basis for a high recall is if the model can guess those who
churned.
:::

::: {#4e719f57 .cell .code execution_count="210"}
``` python
from sklearn.metrics import classification_report

print(classification_report(
    y_test,
    test_pred,
    target_names=['No Churn', 'Churn']
))
```

::: {.output .stream .stdout}
                  precision    recall  f1-score   support

        No Churn       0.97      0.98      0.97      1035
           Churn       0.94      0.92      0.93       374

        accuracy                           0.96      1409
       macro avg       0.95      0.95      0.95      1409
    weighted avg       0.96      0.96      0.96      1409
:::
:::

::: {#14072aee .cell .markdown}
Next, we can display the confusion matrix, which essentially tells us
how many were identified:

-   True Positive (TP)/Top Left: A customer churned and was correctly
    identified.
-   False Negative (FN)/Lower Left: A customer churned but the model
    predicted that they would stay.
-   False Positive (FP)/Top Right: The model predicted churn, but the
    customer stayed.
-   True Negative (TN)/Lower Right: The model predicted no churn, and
    the customer did stay.
:::

::: {#f0c4d9a8 .cell .code execution_count="196"}
``` python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, test_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['No Churn', 'Churn']
)

tn, fp, fn, tp = cm.ravel()

print("True Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives :", tp)

disp.plot()
plt.title('Confusion Matrix - Logistic Regression')
plt.show()
```

::: {.output .stream .stdout}
    True Negatives : 1012
    False Positives: 23
    False Negatives: 30
    True Positives : 344
:::

::: {.output .display_data}
![](vertopal_d621a775d8a2441fab5a019668e3b8dd/de0ea3c1a8920f3b6858c2ae01bb3b6cf8dc3567.png)
:::
:::

::: {#f16a93f6 .cell .markdown}
False negatives usually have a higher business cost because these are
the customers who churned without being identified for possible
retention efforts.

These could be retained customers that would\'ve been beneficial for the
telecommunications company, but since they were retained, profits were
lost. And if these aren\'t identified, then there would be an increase
in losses in the future.
:::

::: {#42c8718a .cell .markdown}
Next, we can use the ***ROC-AUC*** curve, which simple measures if the
model can accurately predict, rather than just guess.

With an ROC-AUC value of 0.99 or 99.31%, we can say that the model
performs better than just randomly guessing. The closer the AUC reaches
0.50 or 50%, we are essentially guessing the values, but a 1.00 means we
can clearly identify rather than guess.
:::

::: {#7eee0422 .cell .code execution_count="197"}
``` python
from sklearn.metrics import roc_auc_score, roc_curve

test_prob = logreg.predict_proba(X_test_processed)[:, 1]

roc_auc = roc_auc_score(y_test, test_prob)

print("ROC-AUC:", roc_auc)

fpr, tpr, thresholds = roc_curve(y_test, test_prob)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f'Logistic Regression (AUC = {roc_auc:.3f})'
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle='--',
    label='Random Guessing'
)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend()
plt.show()
```

::: {.output .stream .stdout}
    ROC-AUC: 0.9931437133483169
:::

::: {.output .display_data}
![](vertopal_d621a775d8a2441fab5a019668e3b8dd/3d4eac36209d1adf84b88e56d244f7e1a9ed66e5.png)
:::
:::

::: {#2dddb238 .cell .markdown}
The logistic regression model achieved an accuracy of **96.24%** on the
**test dataset**. This performance indicates significant generalization.
The model achieved a **ROC-AUC of 99.31%**, meaning it can easily
distinguish between customers who churn and those who remain.

Following that, the confusion matrix shows that the model correctly
identified **344 churners**, while **30** churners were missed.

For the improvement of the business, improving **recall** could help
identify customers, targeting them for retention campaigns.
:::

::: {#452650fb .cell .markdown}
### Balanced Comparison
:::

::: {#b81290f1 .cell .code execution_count="203"}
``` python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Baseline
baseline_pred = logreg.predict(X_test_processed)
baseline_prob = logreg.predict_proba(X_test_processed)[:, 1]

# Balanced
balanced_pred = logreg_balanced.predict(X_test_processed)
balanced_prob = logreg_balanced.predict_proba(X_test_processed)[:, 1]

comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
    'Baseline': [
        accuracy_score(y_test, baseline_pred),
        precision_score(y_test, baseline_pred),
        recall_score(y_test, baseline_pred),
        f1_score(y_test, baseline_pred),
        roc_auc_score(y_test, baseline_prob)
    ],
    'Balanced': [
        accuracy_score(y_test, balanced_pred),
        precision_score(y_test, balanced_pred),
        recall_score(y_test, balanced_pred),
        f1_score(y_test, balanced_pred),
        roc_auc_score(y_test, balanced_prob)
    ]
})

comparison
```

::: {.output .execute_result execution_count="203"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>Baseline</th>
      <th>Balanced</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Accuracy</td>
      <td>0.962385</td>
      <td>0.958126</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Precision</td>
      <td>0.937330</td>
      <td>0.886978</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Recall</td>
      <td>0.919786</td>
      <td>0.965241</td>
    </tr>
    <tr>
      <th>3</th>
      <td>F1-Score</td>
      <td>0.928475</td>
      <td>0.924456</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ROC-AUC</td>
      <td>0.993144</td>
      <td>0.993084</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#57fa7e19 .cell .markdown}
After evaluation, I decided to try evaluate the \'balanced\' model since
there is a 76-24 ratio class imbalance of No Churn to Churn.

The updated model\'s performance was compared against the baseline using
accuracy, precision, recall, F1-score, and ROC-AUC. Although as we can
observe, the original baseline model has better resulting metrics, and
the **performance didn\'t improve**.

This may be due to the fact that the model places greater emphasis on
correctly identifying churners, also known as the minority class. Since
we \'balanced\' the dataset, we emphasized more on the Churn, which was
26% lesser than the No Churn.
:::

::: {#bbb0d4fa .cell .markdown}
### Threshold Comparison
:::

::: {#9e80b73d .cell .markdown}
Since our threshold is at a 0.5 probability, this essentially treats the
two events of Churn and No Churn to occur equally. We can adjust this
threshold to try and get closer to reality, identifying more churners,
but may cause more incorrect predictions.
:::

::: {#db98d588 .cell .code}
``` python
import numpy as np

# Get predicted probability of churn
test_prob = logreg.predict_proba(X_test_processed)[:, 1]

thresholds = np.arange(0.10, 0.91, 0.05)

results = []

# Test the thresholds in the range of 0.10 to 0.90 with a step size of 0.05
# Adds to the result list.
for threshold in thresholds:
    predictions = (test_prob >= threshold).astype(int)

    results.append({
        'Threshold': threshold,
        'Precision': precision_score(
            y_test,
            predictions,
            zero_division=0
        ),
        'Recall': recall_score(
            y_test,
            predictions,
            zero_division=0
        ),
        'F1-Score': f1_score(
            y_test,
            predictions,
            zero_division=0
        )
    })

threshold_results = pd.DataFrame(results)

# Graphs the precision and recall scores against the classification thresholds.
plt.figure(figsize=(9, 6))

plt.plot(
    threshold_results['Threshold'],
    threshold_results['Precision'],
    marker='o',
    label='Precision'
)

plt.plot(
    threshold_results['Threshold'],
    threshold_results['Recall'],
    marker='o',
    label='Recall'
)

plt.xlabel('Classification Threshold')
plt.ylabel('Score')
plt.title('Precision vs Recall at Different Classification Thresholds')
plt.legend()
plt.grid(True)

plt.show()
```

::: {.output .display_data}
![](vertopal_d621a775d8a2441fab5a019668e3b8dd/d99d3adfea79d9b1bd2e6fe1f0c33c8fc300d774.png)
:::
:::

::: {#340b8f1b .cell .markdown}
As we can see, as we increase the threshold, the precision-recall graph
converges, but diverges at around 0.45 to 0.5, and slowly becomes
opposites of each other.

I took the highest F1-score to identify the best threshold result, and
it resulted with **0.45**. Lowering the threshold by a bit from 0.5
increased **recall and f1-score** because more customers were classified
as potential churners, but this also increased the **false positives**
and **reduced precision**.

Based on the observed precision-recall tradeoff, a threshold of **0.45**
provides an appropriate balance for that tradeoff.
:::

::: {#9a25948d .cell .code execution_count="206"}
``` python
best_row = threshold_results.loc[
    threshold_results['F1-Score'].idxmax()
]

print(best_row)
```

::: {.output .stream .stdout}
    Threshold    0.450000
    Precision    0.928382
    Recall       0.935829
    F1-Score     0.932091
    Name: 7, dtype: float64
:::
:::

::: {#fd24eb45 .cell .markdown}
## Feature Importance
:::

::: {#ff21a83d .cell .code execution_count="198"}
``` python
feature_names = preprocessor.get_feature_names_out()

coefficients = logreg.coef_[0]

coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Absolute Coefficient': abs(coefficients)
})

coef_df = coef_df.sort_values(
    'Absolute Coefficient',
    ascending=False
)

coef_df.head(20)
```

::: {.output .execute_result execution_count="198"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature</th>
      <th>Coefficient</th>
      <th>Absolute Coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22</th>
      <td>num__Satisfaction Score</td>
      <td>-6.286594</td>
      <td>6.286594</td>
    </tr>
    <tr>
      <th>13</th>
      <td>num__Number of Referrals</td>
      <td>-1.895110</td>
      <td>1.895110</td>
    </tr>
    <tr>
      <th>1120</th>
      <td>cat__Contract_Two Year</td>
      <td>-1.764045</td>
      <td>1.764045</td>
    </tr>
    <tr>
      <th>868</th>
      <td>cat__City_San Diego</td>
      <td>1.312820</td>
      <td>1.312820</td>
    </tr>
    <tr>
      <th>15</th>
      <td>num__Online Security</td>
      <td>-1.163437</td>
      <td>1.163437</td>
    </tr>
    <tr>
      <th>534</th>
      <td>cat__City_Lakewood</td>
      <td>0.950850</td>
      <td>0.950850</td>
    </tr>
    <tr>
      <th>10</th>
      <td>num__Monthly Charge</td>
      <td>0.914430</td>
      <td>0.914430</td>
    </tr>
    <tr>
      <th>1151</th>
      <td>cat__Zip Code_90024</td>
      <td>0.902527</td>
      <td>0.902527</td>
    </tr>
    <tr>
      <th>717</th>
      <td>cat__City_Olivehurst</td>
      <td>0.899968</td>
      <td>0.899968</td>
    </tr>
    <tr>
      <th>2596</th>
      <td>cat__Zip Code_95961</td>
      <td>0.899968</td>
      <td>0.899968</td>
    </tr>
    <tr>
      <th>1393</th>
      <td>cat__Zip Code_91762</td>
      <td>0.869545</td>
      <td>0.869545</td>
    </tr>
    <tr>
      <th>719</th>
      <td>cat__City_Ontario</td>
      <td>0.857762</td>
      <td>0.857762</td>
    </tr>
    <tr>
      <th>1780</th>
      <td>cat__Zip Code_93245</td>
      <td>0.836471</td>
      <td>0.836471</td>
    </tr>
    <tr>
      <th>549</th>
      <td>cat__City_Lemoore</td>
      <td>0.836471</td>
      <td>0.836471</td>
    </tr>
    <tr>
      <th>1493</th>
      <td>cat__Zip Code_92122</td>
      <td>0.824639</td>
      <td>0.824639</td>
    </tr>
    <tr>
      <th>1492</th>
      <td>cat__Zip Code_92121</td>
      <td>0.821427</td>
      <td>0.821427</td>
    </tr>
    <tr>
      <th>1284</th>
      <td>cat__Zip Code_90808</td>
      <td>0.808203</td>
      <td>0.808203</td>
    </tr>
    <tr>
      <th>1129</th>
      <td>cat__Offer_Offer E</td>
      <td>0.797294</td>
      <td>0.797294</td>
    </tr>
    <tr>
      <th>2112</th>
      <td>cat__Zip Code_94609</td>
      <td>0.772341</td>
      <td>0.772341</td>
    </tr>
    <tr>
      <th>214</th>
      <td>cat__City_Cerritos</td>
      <td>0.766166</td>
      <td>0.766166</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#fde59a40 .cell .code execution_count="199"}
``` python
top_features = coef_df.head(15).sort_values('Coefficient')

plt.figure(figsize=(10, 7))

plt.barh(
    top_features['Feature'],
    top_features['Coefficient']
)

plt.axvline(0, linestyle='--')

plt.xlabel('Logistic Regression Coefficient')
plt.ylabel('Feature')
plt.title('Top 15 Features by Absolute Coefficient')

plt.tight_layout()
plt.show()
```

::: {.output .display_data}
![](vertopal_d621a775d8a2441fab5a019668e3b8dd/b04c93a7665f866302dca4a1b6323b0d32ba12bc.png)
:::
:::

::: {#61d713fb .cell .markdown}
The coefficient graph above indicates that **Satisfaction Score** is the
strongest predictor of the Churn, higher satisfaction means a lower
likelihood of Churn, followed by **Number of Referrals** and having a
**Two-Year Contract**. Several location-based features, particularly
cities such as **San Diego and Lakewood**, show positive associations.
:::

::: {#b2fae97f .cell .markdown}
### Business Insights
:::

::: {#933ca821 .cell .code execution_count="200"}
``` python
top_increase = (
    coef_df[coef_df['Coefficient'] > 0]
    .sort_values('Coefficient', ascending=False)
    .head(5)
)

top_increase
```

::: {.output .execute_result execution_count="200"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature</th>
      <th>Coefficient</th>
      <th>Absolute Coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>868</th>
      <td>cat__City_San Diego</td>
      <td>1.312820</td>
      <td>1.312820</td>
    </tr>
    <tr>
      <th>534</th>
      <td>cat__City_Lakewood</td>
      <td>0.950850</td>
      <td>0.950850</td>
    </tr>
    <tr>
      <th>10</th>
      <td>num__Monthly Charge</td>
      <td>0.914430</td>
      <td>0.914430</td>
    </tr>
    <tr>
      <th>1151</th>
      <td>cat__Zip Code_90024</td>
      <td>0.902527</td>
      <td>0.902527</td>
    </tr>
    <tr>
      <th>717</th>
      <td>cat__City_Olivehurst</td>
      <td>0.899968</td>
      <td>0.899968</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#05a229b8 .cell .markdown}
Increase Churn Risk:

-   City: San Diego: +1.313 - Customers in San Diego have the higheest
    predicted likelihood of churning.
-   City: Lakewood: +0.951 - Customers here have the second highest
    predicted likelihood of churning.
-   Monthly Charge: +0.914 - Higher monthly charges are associated with
    a higher likelihood of churn.
-   Zip Code 90024: +0.903 - Customers in this ZIP code have a higher
    predicted likelihood of churning.
-   City: Olivehurst: +0.900 - Customers here also have a higher
    predicted likelihood of churning.
:::

::: {#148ff7e5 .cell .code execution_count="201"}
``` python
top_decrease = (
    coef_df[coef_df['Coefficient'] < 0]
    .sort_values('Coefficient', ascending=True)
    .head(5)
)

top_decrease
```

::: {.output .execute_result execution_count="201"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature</th>
      <th>Coefficient</th>
      <th>Absolute Coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22</th>
      <td>num__Satisfaction Score</td>
      <td>-6.286594</td>
      <td>6.286594</td>
    </tr>
    <tr>
      <th>13</th>
      <td>num__Number of Referrals</td>
      <td>-1.895110</td>
      <td>1.895110</td>
    </tr>
    <tr>
      <th>1120</th>
      <td>cat__Contract_Two Year</td>
      <td>-1.764045</td>
      <td>1.764045</td>
    </tr>
    <tr>
      <th>15</th>
      <td>num__Online Security</td>
      <td>-1.163437</td>
      <td>1.163437</td>
    </tr>
    <tr>
      <th>1476</th>
      <td>cat__Zip Code_92102</td>
      <td>-0.744023</td>
      <td>0.744023</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#bcb4115e .cell .markdown}
Decrease Churn Risk:

-   Satisfaction Score: -6.287 - Higher satisfaction score means higher
    chance of retention and lower likelihood of churn.
-   Number of Referrals: -1.895 - Customers with more referrals tend to
    have a lower likelihood of churning. These customers may be really
    sold on the service.
-   Contract: Two Year: -1.764 - Customers on two-year contracts have a
    lower likelihood of churn, especially since they are in a committed
    long-term contract.
-   Online Security: -1.163- Customers with online security are
    associated with a lower likelihood of churn.
-   Zip Code: 92102: -0.744 - Customers in this ZIP code have a lower
    likelihood of churn.
:::

::: {#dcedcb10 .cell .markdown}
### Business Validation
:::

::: {#0469389d .cell .markdown}
Some of these findings do make sense. As hypothesized, some features
like Satisfaction Score and Cotnract Type had an effect on the
likelihood of Churn. It makes sense that Satisfaction Score that is high
have a negative coefficient, since customers typically would stay with a
product they are happy with.

On another note, customers on a long-term contract would be more willing
to honor the contract and continue using the service, in this case, for
two years. This goes similarly for referrals. Customers who really trust
a product will refer it to more people, friends or family.

Lastly, on the other hand, locations do matter. San Diego is a very
populated city in California. Meaning, more people will generally tend
to churn as opposed to lesser populated cities. The monhtly charges also
greatly affect a customer\'s reasoning to stay with a service. They may
want to move since they do not have the budget, or rather go for
something cheaper.
:::

::: {#198b7795 .cell .markdown}
<https://www.sciencedirect.com/science/article/pii/S2666720726001384>

Fig. 3. Comparison of the most influential features identified by SHAP
for IBM Telco and Cell2Cell.
:::

::: {#b51602f8 .cell .markdown}
This study above, MetaLight-ChurnNet: A leakage-safe embedding-enhanced
ensemble framework for telecom customer churn prediction, gathered an
ensemble framework in customer churn prediction from IBM Telco and
Cell2Cell with Random Forest Classification.

In terms of feature importance, there are similarities along with IBM
Telco, such as Month-to-month Contracts and Monthly Charges.
Unfortunately, this fictional dataset doesn\'t have other features
featured in these two Telcos, which have a high importance in customer
churn prediction.

A discrepancy visible is that IBM Telco had a high score on tenure, but
this dataset did not, which was one of the hypothesized important
features.
:::

::: {#1ae6737f .cell .markdown}
### Actionable Insights
:::

::: {#6513d9b3 .cell .markdown}
Finally, here are some of the actionable insights I can provide given
the output of the model itself.

1.  **Promotion of longer-term contracts with a reasonable price** - The
    company should encourage a monthly basis of extending the contracts
    of customers in a monthly contract to a **longer-term contract**.
    They can also provide discounts and incentives that may sway the
    customer to consider, given the **monhtly charges** having a large
    impact on customer churn.

2.  **Prioritize customers reporting low satisfaction** - To improve
    satisfaction, the company should follow-up on customers, providing
    surveys, customer-service interventions, and open feedback when
    satisfaction scores decline. Knowing the feedback is key to
    improving customer satisfaction.

3.  **Consider promotion towards highly populated cities** - The company
    can consider a city like San Diego, with the 2nd largest California
    population, to expand efforts when it comes to promoting and
    marketing. They could also focus on customers here to improve
    customer retention as it is the most impactful one causing churn.
:::
