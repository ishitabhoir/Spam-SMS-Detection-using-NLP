# SMS Spam Classifier using NLP

## Overview
This project is a machine learning-based SMS spam classifier that categorizes text messages as Spam or Ham using Natural Language Processing (NLP) techniques and a Multinomial Naive Bayes classifier.

## Features
- Text preprocessing using NLTK
- Stopword removal
- Lancaster stemming
- TF-IDF vectorization
- Multinomial Naive Bayes classification
- Spam bigram analysis
- Predicts whether a custom SMS is Spam or Ham

## Technologies Used
- Python
- Pandas
- NLTK
- Scikit-learn

## Dataset
This project uses an SMS Spam dataset that is downloaded from Kaggle.
Download the dataset and place it in the project directory as "train.csv"

## Output
- Displays dataset information
- Shows top spam bigrams
- Trains the classifier
- Prints accuracy, classification report, and confusion matrix
- Predicts whether the SMS user entered is Spam or Ham
