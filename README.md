📰 Fake News Detector: NLP Classification
Detecting misinformation using Machine Learning and Natural Language Processing.

This project focuses on building a robust classifier to distinguish between "Real" and "Fake" news articles. Utilizing the WELFake Dataset, the pipeline involves advanced text preprocessing, TF-IDF vectorization, and a comparison of various linear models to achieve high predictive accuracy.

🚀 Project Overview
In an era of information overload, identifying unreliable news sources is critical. This notebook implements an end-to-end Machine Learning workflow, from raw data ingestion to final submission generation, specifically tailored for the AWSCC Recruitment challenge.
Key Features:
Data Cleaning: Robust handling of mixed-type labels and missing text values.

Text Engineering: Combined analysis of article titles and body text for better context.

Performance Benchmarking: Comparative analysis of Logistic Regression, LinearSVC, and SGDClassifier.

High Accuracy: Achieved a Cross-Validation F1-score of ~0.982.

📊 Dataset Description
The model is trained on the WELFake Dataset, which contains over 70,000 news records.

ID: Unique identifier for each article.

Title: The headline of the news article.

Text: The full body of the article.

Label: Binary target (0 for Real, 1 for Fake).

🛠️ The Pipeline
1. Preprocessing & Cleaning
Label Sanitization: Filtered out inconsistent labels and converted the target variable to a clean integer format.

Missing Value Handling: Replaced NaN values in titles and text with empty strings.

Feature Fusion: Created a content feature by concatenating the Title and Text to capture the full narrative of the article.

2. Feature Extraction
Used TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical vectors with the following parameters:

max_features: 50,000 - 80,000

ngram_range: (1, 2) or (1, 3) to capture phrases.

sublinear_tf: Applied logarithmic scaling to diminish the influence of high-frequency words.
Model,Configuration,CV F1-Score
Logistic Regression,C=10.0,0.9784
SGD Classifier,alpha=1e−05,0.9818
LinearSVC,"C=2.0, Balanced",0.9820
Final Model: The LinearSVC with balanced class weights was selected for the final submission due to its superior stability and performance.
