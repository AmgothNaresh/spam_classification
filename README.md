# Spam Classification Project

## Overview

This project involves classifying SMS messages into two categories: "ham" (non-spam) and "spam". The process includes data cleaning, exploratory data analysis (EDA), text preprocessing, model building, and evaluation. The final goal is to build a robust spam detection model.

## 1. Data Cleaning

### 1.1 Loading Data
The dataset is loaded from a CSV file with ISO-8859-1 encoding to handle any invalid characters.

### 1.2 Dropping Unnecessary Columns
The dataset initially contains some unnamed columns that are not useful for the analysis. These columns are dropped.

### 1.3 Renaming Columns
To make the dataset more understandable, the columns are renamed to 'target' and 'text'. 'target' represents whether the message is spam (1) or ham (0), and 'text' contains the message content.

### 1.4 Encoding Target Variable
The target variable is encoded using `LabelEncoder` to convert categorical labels into numerical values.

### 1.5 Handling Missing and Duplicate Data
The dataset is checked for missing values and duplicates. Duplicates are removed to ensure data quality.

## 2. Exploratory Data Analysis (EDA)

### 2.1 Initial Analysis
The first few rows of the dataset are examined to understand its structure and content.

### 2.2 Class Distribution
A pie chart is plotted to visualize the distribution of spam and ham messages, highlighting any imbalance in the dataset.

### 2.3 Text Length Analysis
Histograms are used to analyze the distribution of the number of characters and words in spam and ham messages. This helps in understanding the text characteristics.

## 3. Text Preprocessing

### 3.1 Tokenization and Text Cleaning
The text data is preprocessed by converting it to lowercase, tokenizing it, and removing special characters, stop words, and punctuation. Stemming is applied to reduce words to their base form.

### 3.2 Text Transformation
The `transform_text` function performs the text preprocessing steps and transforms the raw text into a cleaned and stemmed format.

### 3.3 Word Clouds
Word clouds are generated for both spam and ham messages to visualize the most frequent words in each category.

### 3.4 Corpus Analysis
The text is split into individual words to analyze the most common words in both spam and ham messages using bar plots.

## 4. Model Building

### 4.1 Vectorization
Text data is converted into numerical format using `TfidfVectorizer`, which transforms the text into feature vectors that can be used by machine learning algorithms.

### 4.2 Model Training
Several classification algorithms are trained and evaluated, including:
- **Naive Bayes** (GaussianNB, MultinomialNB, BernoulliNB)
- **Support Vector Classifier** (SVC)
- **Logistic Regression** (LR)
- **Decision Tree** (DT)
- **Random Forest** (RF)
- **K-Nearest Neighbors** (KNC)
- **AdaBoost** (ABC)
- **Bagging** (BgC)
- **Extra Trees** (ETC)
- **Gradient Boosting** (GBDT)
- **XGBoost** (XGB)

### 4.3 Model Evaluation
Models are evaluated based on accuracy and precision. The performance of each model is compared, and the best-performing models are identified.

### 4.4 Model Improvement
Various techniques, such as adjusting `max_features` in `TfidfVectorizer` and incorporating additional features, are explored to improve model performance.

### 4.5 Voting and Stacking Classifiers
Ensemble methods, including Voting Classifier and Stacking Classifier, are used to combine the predictions of multiple models to enhance performance.

## 5. Final Model and Deployment

### 5.1 Model Saving
The final vectorizer and model are saved using `pickle` for future use or deployment.

### 5.2 Future Work
Potential improvements and deployment strategies are considered for further enhancing the model's performance and usability.

## Conclusion

This project demonstrates a complete pipeline for spam classification, including data cleaning, EDA, text preprocessing, model building, and evaluation. The methods and results provide a foundation for further exploration and application in spam detection systems.
