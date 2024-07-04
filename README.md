# TWITTER-SENTIMENTAL-ANAYLSIS

This project focuses on classifying tweets into positive or negative sentiments. We use the Sentiment140 dataset for training and testing our model, displaying the results using bar graphs and pie charts.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Usage](#usage)

## Overview
The goal of this project is to classify tweets into positive or negative sentiments using the Sentiment140 dataset. We perform data visualization to understand the distribution of sentiments and apply machine learning techniques to build a classification model.

## Dataset
The Sentiment140 dataset includes the following fields:
- `target`: Polarity of the tweet (0 = negative, 1 = positive)
- `ids`: Unique id of the tweet
- `date`: Date of the tweet
- `flag`: Query (if no query, it's NO_QUERY)
- `user`: Name of the user who tweeted
- `text`: Text of the tweet

  ## Dependencies
To run this project, you'll need the following Python packages:
```python
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
import tweepy
import warnings
```
## Exploratory Data Analysis
We start by loading and exploring the dataset to understand the distribution of sentiments and other features.

## Data Preprocessing
We clean and preprocess the data to prepare it for model training.

## Model Training and Evaluation
We train a Bernaulli Naive Bayes model and evaluate its performance.

## Results
We visualize the results using bar graphs and pie charts.

## Conclusion
Our model achieves good accuracy in classifying tweets into positive and negative sentiments. Further improvements can be made by exploring other machine learning algorithms and fine-tuning the preprocessing steps.

## Usage
To run the project:

1. Clone the repository: 
    ```sh
    git clone <[repository-url](https://github.com/Devubavariaa/TWITTER-SENTIMENTAL-ANAYLSIS)>
    ```
2. Navigate to the project directory: 
    ```sh
    cd twitter-sentiment-analysis
    ```
3. Install dependencies: 
    ```sh
    pip install -r requirements.txt
    ```
4. Run the analysis: 
    ```sh
    python analysis.py
    ```
