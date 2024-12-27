# Movie Recommendation System with PySpark and Machine Learning

This repository contains a PySpark-based implementation of a movie recommendation system utilizing various machine learning techniques, including Gradient-Boosted Trees (GBT), Alternating Least Squares (ALS), and Linear Regression (LR).

---

## Features
- **Gradient-Boosted Trees (GBT)**: A classifier to predict user preferences (like/dislike).
- **Alternating Least Squares (ALS)**: A recommendation algorithm to predict user ratings for movies.
- **Linear Regression (LR)**: A regression model to predict user ratings.
- **Data Preprocessing**: Efficient handling of user-movie interaction data.
- **Evaluation Metrics**: AUC, F1 Score, and RMSE to assess model performance.

---

## Requirements
To run this project, install the following dependencies:
- Python 3.8+
- PySpark

---

## Libraries Used

### **PySpark** (Big Data Processing and Machine Learning)
- **`pyspark.sql.SparkSession`**: Used to create a Spark session.
- **`pyspark.sql.functions`**: Includes utility functions for data manipulation, such as:
  - **`col`**: Specify a column in a DataFrame.
  - **`when`**: Create conditional expressions.
- **`pyspark.ml.feature.VectorAssembler`**: Combines multiple columns into a single feature vector.
- **`pyspark.ml.feature.StringIndexer`**: Converts categorical columns into numerical indices.
- **`pyspark.ml.classification.GBTClassifier`**: Implements Gradient-Boosted Tree Classifier.
- **`pyspark.ml.evaluation.BinaryClassificationEvaluator`**: Evaluates binary classification models.
- **`pyspark.ml.evaluation.RegressionEvaluator`**: Evaluates regression models.
- **`pyspark.ml.recommendation.ALS`**: Alternating Least Squares recommendation algorithm.
- **`pyspark.ml.regression.LinearRegression`**: Linear regression model.

---

## Dataset
This implementation expects a dataset in the form of a DataFrame with the following columns:
- `userId`: Unique ID of the user.
- `movieId`: Unique ID of the movie.
- `rating`: Rating given by the user to the movie.

Example structure:
| userId | movieId | rating |
|--------|---------|--------|
| 1      | 101     | 4.5    |
| 2      | 102     | 3.0    |

---

## How It Works

### **Gradient-Boosted Trees (GBT)**
1. Convert `rating` into a binary column (`liked`) where ratings >= 4 are marked as 1 (liked), otherwise 0.
2. Assemble features (`userId` and `movieId`) into a single vector.
3. Train a GBTClassifier on the processed data.
4. Evaluate using AUC and F1 Score.

### **Alternating Least Squares (ALS)**
1. Index `userId` and `movieId` into numerical indices.
2. Train an ALS model to predict ratings for user-movie pairs.
3. Evaluate using RMSE.

### **Linear Regression (LR)**
1. Assemble features (`userId` and `movieId`) into a single vector.
2. Train a Linear Regression model to predict ratings.
3. Evaluate using RMSE.

---

## Results
- **GBT**:
  - Accuracy (AUC): Example output: `0.85`
  - F1 Score: Example output: `0.80`
- **ALS**:
  - RMSE: Example output: `0.95`
- **Linear Regression**:
  - RMSE: Example output: `1.20`

---


## Acknowledgments
This implementation leverages PySpark for big data processing and machine learning tasks, providing a scalable solution for recommendation systems.
