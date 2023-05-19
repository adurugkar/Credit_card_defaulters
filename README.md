<div id="top"></div>

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)


# Credit Card Defaulter Prediction
## About project

- Using Data Science and Machine Learning, we can build a model that predicts whether a credit card holder will default on their payment next month or not, based on their past payment history and demographic information.
- The dataset used for this project is taken from the UCI Machine Learning Repository and stored in a CSV format.
- The project uses Sklearn for pre-processing and model building, as well as Pandas, Numpy, and Matplotlib for CSV reading, data processing, data cleaning, visualization, and analysis.
- The pre-processing steps involve scaling the data, dealing with missing values, and converting categorical variables to numerical using one-hot encoding.
- The project uses several classification models such as Logistic Regression, Decision Tree, Random Forest, and XGBoost to predict the credit card defaulter probability.
- The models are evaluated using accuracy, precision, recall, F1-score, and ROC curve metrics.
- The best model is selected based on the evaluation metrics and hyperparameters tuning.
- The project is deployed as a Flask app on Heroku, where the user can input their details to get a prediction of their credit card defaulter probability.


## Getting Started

To run this project locally, follow the instructions below:

### Prerequisites

Make sure you have the following installed on your machine:

- Python 3
- Jupyter Notebook
- Required packages mentioned in the `requirements.txt` file

### Installation

1. Clone this repository using the following command:

```
git clone https://github.com/adurugkar/Credit_card_defaulters
```

2. Navigate to the project directory:

```
cd Credit_card_defaulters
```

3. Install the required packages:

```
pip install -r requirements.txt
```

## EDA
- In the Credit Card Defaulter Prediction project, we apply Exploratory Data Analysis (EDA) to extract insights from the dataset and understand which features contribute more in predicting whether a credit card holder will default on their payment next month or not.

- In this step, we use Pandas for data analysis and Matplotlib & Seaborn for data visualization. By exploring the dataset, we can gain insights into patterns and relationships within the data.

- We investigate the data for missing values, outliers, and data distributions. We also perform univariate analysis, bivariate analysis, and multivariate analysis to understand the relationships between different features and the target variable.

- By understanding the data, we can identify any data quality issues and determine which features are most important for building our predictive model. We can also identify any potential biases in the dataset that may impact the accuracy of our model.

### Model Building 

- This project aims to predict whether a credit card user is likely to default on their payment using a binary classification model.
- We utilized several models, including Logistic Regression, Decision Tree, Random Forest, XGBoost, and K-Nearest Neighbors, to predict the credit card default.
- The dataset we used contained various features, such as demographics, payment history, and other financial variables.
- We preprocessed the data by handling missing values, encoding categorical features, and scaling numerical features.
- We evaluated the models using appropriate metrics such as Accuracy, Precision, Recall, F1-score, and ROC AUC, and selected the best-performing model.

### Model Selection

- GridSearchCV was utilized to perform hyperparameter tuning, enabling us to optimize the performance of the model by identifying the most suitable combination of hyperparameters.
- In terms of classification, Stratified K-fold Cross-Validation metrics were employed, leveraging the best Mean CV Accuracy Model for deploying the model. This approach ensured that the model's performance is robust and capable of generalizing well on unseen data, enhancing its reliability and effectiveness.
- In the context of classification, the evaluation of the models involved assessing accuracy, recall, precision and AUC-ROC score metrics to determine the best-performing model. The AUC-ROC score is a valuable performance measure for classification-based machine learning models. It provides a quantitative assessment of both the model's goodness of fit to the training data and its ability to accurately predict outcomes for new, unseen data. By considering these metrics, we gain insights into the effectiveness and reliability of the classification models under consideration.
## Project Structure

The project has the following structure:

```

Credit_card_defaulters/
│
├── notebook/
│   ├──data/
│   │	  └──UCI_Credit_Card.csv
│   │──EDA.ipynb
│   └──model.ipynb
│
│
├── source/
│     │──component/
│     │	     │──data_ingestion.py
│     │      │──data_transformation.py
│     │      └──model_trainer.py
│     │──pipeline
│     │──exception.py
│     │──logger.py
│     └──utils.py
│
├── setup.py
├── requirements.txt
└── READE.md
```

- `data` directory contains the dataset used for this project.
- `notebooks` directory contains the Jupyter Notebook file with the code and instructions.
- `exception` contains the Custom exception information for this project.
- `README.md` is the file you are currently reading.
- `requirements.txt` contains a list of required packages and their versions.

## Author
- Avinash Durugkar

## Contact to author

<div id="badges">
  <a href="https://www.linkedin.com/in/adurugkar/">
    <img src="https://img.shields.io/badge/LinkedIn-blue?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn Badge"/>
  </a>
  <a href="adurugkar42@gmail.com">
    <img src="https://img.shields.io/badge/gmail-red?style=for-the-badge&logo=gmail&logoColor=white" alt="Twitter Badge"/>
  </a>
</div>
