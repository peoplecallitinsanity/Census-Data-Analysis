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

## Datasets

- The dataset originates from the U.S. Census Bureau and includes a variety of features such as age, workclass, education, marital status, occupation, race, sex, capital gain, and loss, which are used to predict whether an individual's income exceeds $50K/yr.

## Results and Insights

- The cleaning and feature engineering steps significantly improved the data quality and the effectiveness of the feature set.
- Feature selection identified key factors such as age, education, marital status, and hours per week as significant predictors of income.
- The best-performing model was the AdaBoost classifier, achieving an accuracy of approximately 85.25%.
- Insights from the model suggest significant relationships between income levels and features like marital status, education, and age, which are critical for socioeconomic planning and policy making.

## Setup and Installation

You will need Python 3.x and the following packages: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.

To install necessary libraries, run:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
