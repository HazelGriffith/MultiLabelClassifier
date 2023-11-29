# MultiLabelClassifier

SYSC 2006 Project for Carleton University.

Given a large amount of data in feature sets and class labels with zero knowledge about the problem being solved we are meant to build a classifier that can correctly label samples. The samples can have 1 or more class labels so this is a multi-label problem.
We were given 6 Feature Sets with different numbers of features and have to find the best feature set for our classifier.

Our group's best available choice was to use a Bayesian classifier. This was unfortunate as we have no prior knowledge to leverage, but we made it as accurate as we could with the following methods.

# Pre-Processing

First we checked the feature sets for outliers and found no statistically significant amount of outliers.

Second we plan on checking for features within the feature sets that are highly correlated and removing them.

Third we use SMOTE to generate samples of the minority classes to fix our class imbalance with oversampling.

# Classifier

our classifier is a Binary Relevance classifier where a Gaussian Naive Bayes classifier is trained on each label. So with 19 labels we have 19 classifiers in a One Vs Rest approach.

# Meta-Learning

We aim to use boosting for each naive bayes classifier to try and reduce their bias. Currently we're looking at Adaboost.

# Goal

Our goal is to maximize the F-score Micro Average of our results.
