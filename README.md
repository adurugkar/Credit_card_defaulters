

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

### Usage

1. Open Jupyter Notebook:

```
jupyter notebook
```

2. Navigate to the `notebooks` directory and open the `credit_card_defaulter_prediction.ipynb` file.

3. Follow the instructions mentioned in the notebook to run the code and make predictions.

## Project Structure

The project has the following structure:

```
credit-card-defaulter-prediction/
│
├── data/
│   ├── credit_card_default.csv
│   └── README.md
│
├── notebooks/
│   ├── credit_card_defaulter_prediction.ipynb
│   └── README.md
│
├── LICENSE
├── README.md
└── requirements.txt
```

- `data` directory contains the dataset used for this project.
- `notebooks` directory contains the Jupyter Notebook file with the code and instructions.
- `LICENSE` contains the license information for this project.
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
    <img src="https://img.shields.io/badge/gmail-blue?style=for-the-badge&logo=gmail&logoColor=red" alt="Twitter Badge"/>
  </a>
</div>
