# Census Data Clustering and Analysis

This repository hosts a comprehensive analysis of census data using advanced data science techniques to understand demographic and employment-related patterns that affect income levels.

## Project Structure

### Notebooks

- **01_Data_Analysis.ipynb**
  - **Objective:** Initial exploration of the census dataset. 
  - **Techniques Used:** Uses `pandas` for data manipulation, `matplotlib` and `seaborn` for visualization. This includes generating histograms, bar charts, and boxplots to analyze distributions of age, education, workclass, etc.
  - **Outputs:** Summarized statistics and visualizations illustrating the distribution of key demographic and employment variables.

- **02_Cleaning&Feature_Engineering.ipynb**
  - **Objective:** Prepare the dataset for predictive modeling.
  - **Techniques Used:** Cleans data by handling missing values and outliers, creates new composite features that enhance model performance, and employs `pandas` for data transformations and `scikit-learn` for encoding categorical variables.
  - **Outputs:** A clean, feature-engineered dataset ready for model application.

- **03_Feature_Selection.ipynb**
  - **Objective:** Identify the most predictive features for income classification.
  - **Techniques Used:** Applies `SelectKBest` and `Recursive Feature Elimination (RFE)` from `scikit-learn` to rank features based on their importance. Utilizes `ExtraTreesClassifier` for feature importance analysis.
  - **Outputs:** A reduced set of features that have the highest impact on predicting income, enhancing model simplicity and performance.

- **04_Model.ipynb**
  - **Objective:** Develop and evaluate models to predict income levels.
  - **Technologies Used:** Utilizes `scikit-learn` for model development including logistic regression, K-nearest neighbors, decision trees, and ensemble methods such as Random Forest and AdaBoost. 
  - **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, and AUC-ROC for model performance comparison.
  - **Outputs:** Detailed performance analysis of each model, with hyperparameter tuning using `GridSearchCV` to optimize model configurations.

#### SHAP Value Analysis

- **Setup**: I initialized the SHAP explainer with the Random Forest model predictions and feature set, generating SHAP values for the test data.
- **Impactful Visualizations**:
  - **Bar Plot**: Showed the overall impact of each feature across the entire dataset.
  - **Waterfall Plot**: Detailed the contribution of each feature to specific predictions, highlighting how each feature pushes the model output from the base value.
  - **Beeswarm Plot**: Illustrated the distribution of the impacts each feature has on the model output, providing insights into the variability of feature effects across many instances.

#### Deep Dive into SHAP Insights

- **Marital Status**: Being married had a predominantly positive impact on income predictions, aligning with societal observations where dual-income families or stable relationships often have better financial standings.
- **Education**: Higher educational levels consistently led to higher income predictions, reaffirming the value of advanced education in career prospects.
- **Occupation and Workclass**: These features showed more complex relationships with income, depending on the specific job and employment sector.
- **Age and Hours per Week**: Both features demonstrated expected trends where older age and more hours worked per week contributed to higher income predictions.

### Conclusion on Model Interpretation

The use of SHAP provided profound insights into how different socio-economic and demographic factors affect income levels. The interpretation tools helped validate some common socio-economic theories while also uncovering complex patterns that only machine learning models can adequately capture.

The final accuracy achieved by the Random Forest model was **85.77%**, indicating a high level of predictiveness while maintaining generalizability across the unseen test data.



## Datasets

- The dataset originates from the U.S. Census Bureau and includes a variety of features such as age, workclass, education, marital status, occupation, race, sex, capital gain, and loss, which are used to predict whether an individual's income exceeds $50K/yr.



## Setup and Installation

You will need Python 3.x and the following packages: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.

To install necessary libraries, run:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
