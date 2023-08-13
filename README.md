## Gnaneswar Kolla

## Project Title
### CS5293,Spring 2022 Project 3

## Project Description
This project unredacts sensitive information like names from tsv file and calculates precision,accuracy and f1 scores.


## Installation/Getting Started
---
1. Pipenv to create and manage virtual environment for the project.
   > pipenv install
2. Packages required to run this project are in pipfile which will be downloaded in step1.
3. Once, the packages are successfully installed, the project can be executed using
   > pipenv run python unredactor.py

4. Pytests can be runnable using below command
   > pipenv run python -m pytest

## Packages/Modules
---
- `pandas` is a python library used to perform data analysis and manipulation.
    - In this project, pandas is used to create a data frame for ingredients data.
- `sklearn` is a python library used for classification, predictive analytics and various machine learning tasks.
    - `KNeighborsClassifier` is a classifier implementing the k-nearest neighbors vote.
        - In this project KNeighborsClassifier is used to predict cuisine based on ingredients and also tp predict score for that particular cuisine.
- `TextBlob` returns polarity and subjectivity of a sentence. Polarity lies between [-1,1].
    - -1 defines a negative statement and 1 defines a positive statement.
- `pytest` is a framework to write small, readable tests, and can scale to support complex functional testing for applications and libraries.
    - In this project, pytest is used to create [unit tests](#tests) for each functionality
## Assumptions/Bugs
1. Assuming that the .tsv file is present in the public github repository link.
2. Assuming that the dataset contains training, validation and testing data.
3. Assuming th value of n-neighbors in KNN to be 20.
4. If it dosn't work in VM, can you please try in your local machine.
 
##  Approach to Developing the code
---
1. `get_data()`
   This function is to get latest data from github link and convert it into dataframe. Here I have given column names to each tab seperated row as github_name,type,redacted_name,redacted_sentence.
2. `get_normalized_data(df)`
   This function is used to remove some punctuations and br tag from the redacted_sentences.
3. `feature(df)`
   This function is used to create features like length of redacted_names ,redacted_sentences and to get score of sentiment.
4. `calculate_score(sentence_text)`
   This function is to get polarity score for a sentence.
5. `get_features_data(df,type)`
   This function takes dataframe and type of data to get training data.
6. `get_vectorized_data(df,training_features_list)`
   This function is used to vectorize the data.
7. `get_scores(validation_labels, predicted_labels)`
   This function is used to calculate precision, recall and f1 scores.
8. `main()`
   This function uses .tsv file from github and gets featured data in list of dictionary and gets vectorized using DictVectorizer and uses KNeighborsClassifier to predict the labels. Finally prints the scores as output. 

## Tests
---
1. **`test_main.py`**
   | Function | Test Function | Description  |   
   |   --- |   --- |   ---
   |   `main()`    |    `test_main(capfd)`    |    Tests the whole execution of the program and checks result.
2. **`test_calculate_score.py`**
   | Function | Test Function | Description  |   
   |   --- |   --- |   ---
   |   `calculate_score(sen)`    |    `test_calculate_score()`    |    Tests whether the method returns polarity of sentiment analysis in a sentence.

3. **`test_get_feature.py`**
   | Function | Test Function | Description  |   
   |   --- |   --- |   ---
   |   `get_feature(df)`    |    `test_get_feature()`    |    Tests whether the features are getting correctly from the data.

4. **`test_get_features_data.py`**
   | Function | Test Function | Description  |   
   |   --- |   --- |   ---
   |   `get_features_data(df,type)`    |    `test_get_features_data()`    |    Tests whether the extra feature data is coming.

5. **`test_get_normalized_data().py`**
   | Function | Test Function | Description  |   
   |   --- |   --- |   ---
   |   `get_normalized_data(df)`    |    `test_get_normalized_data()`    |    Tests whether the data is normalised correctly.
