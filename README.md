Toh Meng Seng

# 20-toh-meng-seng-224E

Batch 20 Technical Assessment

Project objective:

1. Perform a preliminary data exploration
2. Perform Exploratory Data Analysis (EDA) 
3. Execute Machine Learning (ML) pipeline, for at least 3 models
   
#### Project Structure

    ├── data/
    │   ├── agri_eda_completed.csv
    │   ├── agri.csv
    │   └── agri.db
    ├── src/
    │   ├── classification_models.py
    │   ├── config.yaml
    │   ├── data_preparation.py
    │   └── regression_models.py
    ├── eda.ipynb
    ├── LICENSE
    ├── main.py
    ├── README.md
    ├── requirements.txt
    └── run.sh

#### Perform a preliminary data exploration

Aim: get a sense of the data, it's structure, quality, and key characteristics before further analysis.

The EDA Steps:
- Check for Duplicates: Identify and remove identical rows.
- Handle Missing Values: Address NaNs using imputation or removal.
- Data Type Validation: Convert data types where necessary.
- Detect Outliers: Use statistical methods or visualizations.
- Feature Engineering: Create new meaningful features.
- Data Normalization & Transformation: Standardization, scaling, or encoding.
- Exploratory Visualizations: Histograms, boxplots, correlation heatmaps.
- Statistical Tests: Checking relationships between variables.


 |#   |Column                |Non-Null Count  |Dtype| 
 |--- |---                   |---             |---  |
 |0 |  Client ID             |41188 non-null  |int64|
 |1 |  Age                   |41188 non-null  |object|
 |2 |  Occupation            |41188 non-null  |object|
 |3 |  Marital Status        |41188 non-null  |object|
 |4 |  Education Level       |41188 non-null  |object|
 |5 |  Credit Default        |41188 non-null  |object|
 |6 |  Housing Loan          |16399 non-null  |object|
 |7 |  Personal Loan         |37042 non-null  |object|
 |8 |  Contact Method        |41188 non-null  |object|
 |9 |  Campaign Calls        |41188 non-null  |int64 |
 |10|  Previous Contact Days |41188 non-null  |int64 |
 |11|  Subscription Status   |41188 non-null  |object|

 Dataset:s 41188  rows, 11 columns
* mix of object and numerical (int64) data types
* features have null entries/missing values

#### <u>Problem statement</u>

Predicting the likelihood that a client, from AI-Vive-Banking, will subscribe to a term deposit, based on client information and also data from direct marketing campaigns.

Objective: To perform **Classification ("yes", "no") of the Subscription Status**.

The dataset consists of the following features:

1. Age
2. Occupation
3. Marital Status
4. Education Level
5. Credit Default
6. Housing Loan
7. Personal Loan
8. Contact Method
9. Campaign Calls
10. Previous Contact Days
11. Subscription Status (Target)

This dataset will be used to develop the classification models to predict the Subscription Status of the client, enabling the bank to enhance their **resource allocation** for customer engagement.

The outcome aims to streamline the bank operations to improve its **financial performance**.

After understanding the data types with respect to the problem statements, the preparation of data can be performed before the actual EDA.

#### main.py
- Requires Python 3.11.0
- Library imports
- Import libraries required for classification models

1. Define the path to the config.yaml file
2. Load the data from the database
3. Data Cleaning
4  Step 1 - Call Classification_setup
* Display Classification_setup_results
1. Step 2- Call Classification_preprocessor
* Display Preprocessor_results
1. Step 3 - Perform classification modeling - Baseline
2. Step 4 - Perform classification modeling - with hyperparameters tuning

#### data_preparation.py
**Findings:**
1. Dataset:s 41188  rows, 11 columns
    * mix of object and numerical (int64) data types
    * features have null entries/missing values
2. There are duplicates of > 10% by the removal of the column Client ID. 
    * Converting to lowercase
    * Duplicates are removed.

#### Perform Exploratory Data Analysis (EDA) 
**Findings:**
3. High missing values for columns Housing Loan and Personal Loan
4. Instead of removing values in Age column that is > 100, imputting with mean, within the < 100 range, will keep the data size intact for useful analysis.
5. At this stage, the size of duplicates increases from 12% to 20%
6. For Credit Default, the percentage of 'unknown' values is about 20%.
7. For Credit Default column, addressing how to handle values of 'unknown'
   * Instead of removal, these value can be better addressed by renaming as "undisclosed".
   * This is to provide better understanding of the data.
8. For column Campaign Calls, there are negative values which are not valid.
9. From the boxplot of 'Campaign Calls' column, there are:
   * outliers
   * negative values
10. Campaign Calls seems to be ineffective from persistent callings. This feature can be considered to be dropped off.
11. Boxplot of 'Previous Contact' column showed the data type change from numerical to object, with 2 distinct records.
    * 'no prior contact'
    * 'prior contact'
12. There are high percentage of missing values in both Housing Loan and Personal Loan
    1. Column: Housing Loan
       - Number of missing values: 20768
       - Percentage of missing values: 60.38%
    2. Column: Personal Loan
       - Number of missing values: 3515
       - Percentage of missing values: 10.22%
13. From the bar chart, both the Housing Loan and Personal Loan features can be combined to form a new feature that indicates that the client has existing loans. 
    * This will make the analysis more effective.
14. Results from bivariate analysis of categorical columns

| Categorical Column | Feature Engineering options | 
| ------------- | ------------- | 
|Occupation | To **re-map** the row values as blue-collar, white-collar & grey-collar. This can assist the bank to target their actions effectively.| 
|Marital Status | **Renamed** 'divorced' as 'single', to remove bias| 
|Education Level | To **re-map** the row values as primary, secondary & tertiary.This can assist the bank to target their actions effectively.| 
|Credit Default | To **remove row values with 'yes'** in order to improve performance and lower the risk to the bank.| 
|Housing Loan | High count of value 'no'. To **transform** with Personal Loan into a new column 'On Loan'.| 
|Personal Loan | High count of value 'no'. To **transform** with Housing Loan into a new column 'On Loan'.| 
|Contact Method | To be **removed** since the method is voice call, whether via cell or telephone. | 
|Previous Contact | To **renamed** row value '999' and '0' as 'no prior contact' while the remainder as 'prior contact' |

15. Results from bivariate analysis of numerical columns

| Numerical Column | Feature Engineering options  |
| ------------- | ------------- | 
|Age | Distributions for both subscription value have right skewness with tail to the right. **Transforming** 'Age' into age range into a new column 'Age Group' will allow the analysis to be more effective. | 
|Campaign Calls | From the plot, there seems to be some positive subscription for age > 50, even though the number of calls are high for age < 50. However, the effectiveness reduces after > 10 calls. To **remove outliers** using IQR method. |
16. From the Campaign Calls, there are few calls that are over 10. The outliers should be removed.
17. For Marital Status, rename 'divorced' and 'single' as 'not married'
    * This will reduce the dimension of column 'Marital Status' to indicate a binary quality:
      * married
      * not married
    * This will remove any bias for divorced status, making the machine learning model more **explainable**.
18. High counts of missing values for both Housing Loan & Personal Loan  
19. From the plot of 'On Loan', the transfomation reduces the worst case of Missing values (as seen in 'Housing Loan'), from 56.92 % to 49.84% (reclassify as 'unknown' in the new feature), But this number is still significant.
    * Similar to 'Credit Default', the 'unknown' can be renamed as 'undisclosed' for better data analysis, instead of imputting with a 'yes' or 'no' or considering row removal which may impact the dataset performance. 
20. Key Observations from Spearman correlation:

| Features | Observation |
| --- | --- |
|Marital Status & Age Group (0.1266)|Slight positive correlation - older individuals may have different marital patterns.|
|Marital Status & Credit Default (-0.1092)| Weak negative correlation - marital status may have some effect on credit reliability.|
|Previous Contact & Age Group (0.0814)| Slight positive correlation - older individuals might react differently.|
|Occupation Group & Previous Contact (0.0981)| Mild positive correlation - certain occupations may receive more targeted outreach.|
|Education Group & Credit Default (0.0737)| Weak positive correlation - education might slightly impact credit behavior.|
|On Loan Status & Other Variables| Relatively weak correlations - with other categorical variables.|

21. Key Observations from Chi-Square test:

Dependent variables (significant association with Subscription Status):
 
|  Dependent variable  |  Target  |
| --- | --- |
|Marital Status| Dependent  |
|Credit Default| Dependent: strong association or significantly influences   |
|Previous Contact| Dependent: very strong association   |
|Age Group| Dependent: strong association or strong relationship  |
|Education Group| Dependent  |
|Occupation Group| Dependent: strongly affects  |
|On Loan status| Dependent   |

#### model_training.py
Classification models, which can be selected using config.ymal, are:
 - LogisticRegression
 - KNeighborsClassifier
 - RandomForestClassifier
 - SVC
 - GradientBoostingClassifier

Hyperparamter tuning is stored in config.yaml and can be modified.

Key: classification_tuning_performance_set_optimum: 1
Within the config.yaml, this key, range 1 or 2, can be updated to run various depth of hyperparameters, from 1=fast or 2=slow.

Note: For submission, the key is set to 1 for faster response during the hyperparameters tuning.

22. At this junction, the best results:
    1. Baseline:
    Best model based on F1 Score: LogisticRegression , with a F1 Score of:  0.667
    2. With hyperparameter tuning:
    Best model based on F1 Score: SVC , with a F1 Score of:  0.8745

Proceed at this point to fine tune ine the dataset and use Logistic Regression to determine the best feature selection before attmepting the hyperparameter testing with the best model.

Set the classification_models to list only Logistic Regression in order to run through the following combinations:
* Check for improvement in the F1 Score when:
    1. Dropping column one-by-one
    2. Dropping column two-by-two
    3. Dropping column three-by-three
* F1 Score for all cases is 0.667

#### Summary of the Final Cleaned Dataset

|Data type: object 	|Number of unique values	|Unique values|
|  	----------- 	|  		-----------	        |-----------|
|Subscription Status|2			        	    |['no' 'yes']|
|Marital Status		|2			               	|['married' 'not married']|
|Credit Default		|2			             	|['no' 'undisclosed']|
|Previous Contact Days	|2	    	    		|['no prior contact' 'prior contact']|
|Age Group		    |5			             	|['Twenties' 'Thirties' 'Forties' 'Fifties' 'Above Sixties']|
|On Loan		    |2			            	|['no' 'undisclosed']|
|Education Group	|3		            		|['secondary' 'tertiary' 'primary']|
|Occupation Group	|3		            		|['blue collar' 'white collar' 'grey collar']|		
|**Data type: int64**	|**Number of unique values**	|**Unique values**|
|Campaign Calls		|6			            	|[1 4 2 3 5 6]|

#### The EDA and ML pipeline completed, the following are observed.

1. Feature `Age` is a range of numnber that has little performance impac to positive subscription and this feature is best encoded by binning into age range or `Age Group`
2. Similar for categorical features of `Occupation`, `Education Level`, which are encoded to group them into similar characteristics, to new features `Occupation Group` and `Education Group`
3. For `Marital Status`, the value 'divorced' is renamed as 'single'
4. For `Credit Default`, the value 'yes' is dropped while the 'unknown' is renamed as 'undisclosed'.
5. Both `Housing Loan` and `Personal Loan`  have large count of missing values. The transformation of these 2 features into a new `On loan` features reduces the missing values by taking into account the 'yes' in either both features, for example.
6. `Contact Method` is dropped as the values of 'Cell' 'telephone' 'cellular' and 'Telephone' mean the same communication method. A call is a call.
7. `Campaign Calls`, after removal of negative numbers, has its outliers removed using 1.5xIQR. The config.ymal hosts this configuration numnber that can assist future performance check by fine tuning this number easily using config.yaml.
8. `Previous Contact Days` has high count of '999' which refer to 'no prior contact'. This feature is thus transformed into a new feature `Previous Contact`
9. Lastly the target `Subscription Status` has imbalance distribution, which is expected as the bank strives to increase subscriptions.
10. The tuning of hyperparameters improves performance.
11. The cleaned dataset has significant duplicates due to the few categorial characteristics within each feature and the duplicates has to be dropped.

#### Some improvements
- Handle class imbalance (e.g., using SMOTE).
- Other ways of feature engineering may boost performance.

#### Summary of the Final Evaluation

#### Classification:

* **Best Model:** LogisticRegression
* **Evaluation Metrics:**
  * Accuracy: 0.901797
  * Precision: 0.887818
  * Recall: 0.901797
  * F1 Score: 0.875377 

- Decent but not perfect.
- A F1 score of 0.87 suggests that the model is doing a reasonable job at baseline but has room for improvement.
- It is better than random guessing, indicating it is a moderately effective model.

#### Conclusion

Hence, AI-Vive-Banking is able to effectively identify, using the features of the dataset and the establised machine learning pipeline, to categorise the `Subscription Status` of the clients, thus enhancing the resource allocation and better customer engagement.

*Thank you.*

Toh Meng Seng