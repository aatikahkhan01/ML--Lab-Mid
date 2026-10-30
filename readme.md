# Full-Stack Machine Learning Prediction App

## 1. Project Overview

This project is a full-stack Machine Learning application that provides two prediction services through a simple web interface:

1. **Stroke Prediction:** A classification model that predicts a stroke outcome based on health-related information.
2. **House Price Prediction:** A regression model that estimates a house price based on property features.

The application combines machine learning, model evaluation, model saving, a FastAPI backend, and an HTML, CSS, and JavaScript frontend.

Users can enter information into the website and receive predictions without directly interacting with Python code.

**Medical Disclaimer:** The stroke prediction feature is developed for educational purposes only. It is not a medical diagnosis or a substitute for professional medical advice.

## 2. Project Objectives

The main objectives of this project are:

- To develop a classification model for stroke prediction.
- To develop a regression model for house price estimation.
- To study and compare different classification algorithms covered in class.
- To evaluate model performance using appropriate metrics.
- To save trained models using Joblib.
- To develop a FastAPI backend for prediction requests.
- To create a user-friendly web interface.
- To integrate machine learning models into a working web application.

## 3. Technologies Used

The following technologies and libraries are used in the project:

- **Python:** Programming language used for machine learning and backend development.
- **Pandas:** Used for loading and processing datasets.
- **Scikit-learn:** Used for preprocessing, training, and evaluating machine learning models.
- **Joblib:** Used to save and load trained machine learning models.
- **FastAPI:** Used to develop the backend API.
- **Uvicorn:** Used to run the FastAPI application.
- **HTML:** Used to structure the website.
- **CSS:** Used to style the website.
- **JavaScript:** Used to handle form submissions and communicate with the backend.
- **Jupyter Notebook:** Used for model development, experimentation, and evaluation.
- **Visual Studio Code:** Used as the development environment.

## 4. Datasets

### 4.1 Stroke Prediction Dataset

The stroke dataset contains 5,110 records and 12 columns. It includes health-related and demographic features, such as:

- Gender
- Age
- Hypertension
- Heart disease
- Marital status
- Work type
- Residence type
- Average glucose level
- Body mass index (BMI)
- Smoking status
- Stroke outcome

The target variable is `stroke`, where 0 represents no recorded stroke and 1 represents a recorded stroke.

The dataset is imbalanced because the number of non-stroke records is much greater than the number of stroke records. This imbalance is important when interpreting the model evaluation results.

### 4.2 House Price Dataset

The house price dataset contains 21,613 records and 21 columns. It includes information about residential properties, such as:

- Number of bedrooms and bathrooms
- Living area and lot size
- Number of floors
- Waterfront and view information
- Property condition and grade
- Year built and year renovated
- Location information
- House sale date
- House sale price

The target variable is `price`, which represents the house price in the dataset.

The dataset represents historical US housing sales. Therefore, its predictions should not be interpreted as current Pakistani property prices.

## 5. Data Preparation and Preprocessing

Data preparation is an important step before training machine learning models. The project uses the following general preprocessing activities where applicable:

- Loading the datasets using Pandas.
- Inspecting the dataset structure and columns.
- Preparing the input features and target variables.
- Handling missing values through preprocessing where configured.
- Converting categorical features into numerical representations where required.
- Scaling numerical features where required by the model pipeline.
- Splitting data into training and testing sets.
- Using the training data to fit models and the testing data to evaluate them.

The stroke classification evaluation uses an 80:20 train-test split with stratification and a random state of 42. Stratification helps preserve the proportion of stroke and non-stroke records in both sets.

The saved website models use preprocessing pipelines where configured, allowing the required preprocessing to be applied when new inputs are submitted.

## 6. Machine Learning Models

### 6.1 Stroke Classification Model

Classification is used when the target variable represents a category. In this project, the target is the stroke outcome.

The saved stroke classification model is stored in:

`stroke_model.joblib`

It is loaded by the FastAPI backend to generate predictions from user-provided health information.

### 6.2 House Price Regression Model

Regression is used when the target variable is a continuous numerical value. In this project, the target is the sale price of a house.

The saved house price model is stored in:

`house_model.joblib`

It is loaded by the FastAPI backend to estimate house prices from property information.

### 6.3 Additional Classification Models Studied

To compare machine learning techniques covered in class, five additional classification models were trained and evaluated on the stroke dataset.

**1. Decision Tree — Gini**

A Decision Tree makes predictions using a sequence of feature-based decisions. Gini impurity measures how mixed the classes are at a node. Lower impurity indicates a more class-pure node.

**2. Decision Tree — Entropy**

This Decision Tree uses entropy to measure uncertainty or impurity. Information Gain measures how much entropy decreases after a split. A split with greater Information Gain reduces uncertainty more.

**3. Random Forest**

Random Forest combines predictions from multiple decision trees. Combining trees can improve generalization, but its performance must still be evaluated using appropriate metrics.

**4. Support Vector Machine (SVM)**

SVM attempts to find a decision boundary that separates classes. It can use different kernels to model different types of decision boundaries.

**5. Naive Bayes**

Naive Bayes is a probabilistic classifier based on Bayes' theorem. It makes a simplifying assumption that features are conditionally independent given the class.

These five models were additional classification experiments. The website continues to use the saved stroke and house price models described above.

## 7. Model Evaluation

Models are evaluated on test data to estimate how well they perform on records that were not used for training.

### 7.1 Stroke Classification Model Results

The saved website stroke model produced the following results:

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 74.56% |
| Stroke Precision | 14% |
| Stroke Recall | 80% |
| Stroke F1-score | 0.24 |
| Test records | 1,022 |

**Interpretation:**

- Accuracy measures the proportion of all predictions that are correct.
- Precision measures how many predicted stroke cases were actual stroke cases.
- Recall measures how many actual stroke cases the model identified.
- The F1-score combines precision and recall into one measure.

The model identified 40 of the 50 actual stroke cases in the test set. However, it also incorrectly flagged 250 non-stroke cases as stroke cases.

These results show why accuracy alone is not enough to judge a classifier on an imbalanced dataset.

### 7.2 House Price Regression Model Results

The saved website regression model produced the following results:

| Evaluation Metric | Result |
|---|---:|
| R² Score | 0.6965 |
| Mean Absolute Error (MAE) | $129,202.25 |

**Interpretation:**

- **R² Score:** Indicates how much variation in house prices is explained by the model. An R² of approximately 0.6965 means the model explains about 69.65% of the variation in the evaluation data.
- **Mean Absolute Error (MAE):** Measures the average absolute difference between predicted and actual house prices. The reported MAE is approximately $129,202.25.

The MAE indicates that individual predictions can differ substantially from actual sale prices.

### 7.3 Comparison of Additional Classification Models

The following table summarizes the five additional classification experiments. These results are separate from the saved stroke model used by the website.

| Model | Accuracy | Stroke Precision | Stroke Recall | Stroke F1-score |
|---|---:|---:|---:|---:|
| Decision Tree — Gini | 92.87% | 18.9% | 14% | 0.161 |
| Decision Tree — Entropy | 91.87% | 7.7% | 6% | 0.067 |
| Random Forest | 95.01% | 0% | 0% | 0.000 |
| SVM | 76.22% | 13.3% | 70% | 0.224 |
| Naive Bayes | 29.84% | 6.4% | 98% | 0.120 |

**Interpretation:**

Random Forest achieved the highest accuracy among these five additional models, but it failed to identify any actual stroke cases in the test set. Naive Bayes identified nearly all actual stroke cases but generated many false-positive predictions. SVM achieved the highest F1-score among these five models.

These results demonstrate the importance of evaluating precision, recall, and F1-score alongside accuracy when working with imbalanced data.

## 8. Model Saving and Loading

The trained website models are saved using Joblib:

- `stroke_model.joblib`
- `house_model.joblib`

The FastAPI backend loads these files when the application starts. This allows the application to generate predictions without retraining the models every time a user opens the website.

The saved model files must remain available in the project folder for the application to work.

## 9. Application Architecture

The application consists of three main parts:

### Frontend

The frontend is built using HTML, CSS, and JavaScript. It provides input forms for stroke prediction and house price estimation. JavaScript sends the submitted information to the backend and displays the returned results.

### Backend

The backend is developed using FastAPI in `app.py`. It receives requests from the frontend, prepares the input data, runs the appropriate saved model, and returns the prediction.

### Machine Learning Models

The two saved Joblib model files perform the predictions:

- The stroke classification model predicts the stroke outcome.
- The house price regression model estimates a numerical property price.

### Prediction Flow

1. The user opens the website.
2. The user selects a prediction service.
3. The user enters the required information.
4. The frontend sends the information to the appropriate API endpoint.
5. The backend passes the input to the saved model.
6. The model generates a prediction.
7. The backend returns the result.
8. The frontend displays the result to the user.

## 10. API Endpoints

The FastAPI application provides the following endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Displays the website. |
| POST | `/predict/stroke` | Receives health-related inputs and returns a stroke classification prediction. |
| POST | `/predict/house` | Receives property features and returns an estimated house price. |

The frontend uses these endpoints to communicate with the backend.

## 11. Project File Structure

The main project folder should contain the following files:

```text
ml/
├── app.py
├── index.html
├── stroke_model.joblib
├── house_model.joblib
├── kc_house_data.csv
├── ml assign.ipynb
└── readme.md
```

The stroke dataset used during model development is named:

`healthcare-dataset-stroke-data.csv`

In the current development setup, this CSV is stored outside the main project folder. If the notebook needs to be rerun on another computer, update the dataset path or place the CSV in the project folder and adjust the notebook accordingly.

The website can use the saved models without the original training CSV files, provided the model files and application files are available.

## 12. Installation and Setup Instructions

### Requirements

- Python 3.12 or a compatible Python version.
- Visual Studio Code or another code editor.
- The project files listed above.

### Step 1: Open the Project Folder

Open the `ml` project folder in Visual Studio Code.

### Step 2: Install the Required Libraries

Open the VS Code terminal in the project folder and run:

```bash
pip install fastapi uvicorn pandas scikit-learn joblib
```

### Step 3: Start the Application

Run the following command in the terminal:

```bash
python -m uvicorn app:app --reload
```

Wait until Uvicorn reports that the application has started successfully.

### Step 4: Open the Website

Open Google Chrome and visit:

http://127.0.0.1:8000

Do not open `index.html` by double-clicking it. The website should be opened through the running FastAPI server so its requests can reach the backend.

### Step 5: Make Predictions

- Select **Stroke Risk Prediction** to enter the requested health information.
- Select **House Price Estimator** to enter the requested property information.
- Submit the form and review the result displayed on the page.

Keep the terminal open while using the website.

### Step 6: Stop the Application

To stop the server, return to the terminal and press `Ctrl + C`.

## 13. Testing

The application was tested locally through the FastAPI server.

The following checks were completed:

- The website homepage loaded successfully.
- The stroke prediction endpoint returned a response.
- The house price prediction endpoint returned a response.
- The saved model files loaded successfully.
- Evaluation code was run for the saved website models.
- Additional classification models were trained and compared in the notebook.

Successful HTTP 200 responses were observed for the homepage and both prediction endpoints during testing.

## 14. Limitations and Future Improvements

### Limitations

- The stroke dataset is imbalanced, affecting the interpretation of classification metrics.
- The saved stroke model produces false-positive predictions and is not suitable for clinical decision-making.
- The house price model's MAE indicates that its predictions may differ considerably from actual prices.
- The house dataset represents historical US housing sales, so it is not directly suitable for estimating current Pakistani property prices.
- The additional classification experiments show that a model with high accuracy may still perform poorly at identifying the minority class.

### Future Improvements

- Improve stroke classification through appropriate imbalance-handling methods and threshold tuning.
- Compare models using precision, recall, F1-score, and confusion matrices.
- Perform further hyperparameter tuning and feature engineering.
- Test models using additional validation methods.
- Use more recent and geographically relevant housing data when estimating prices for a particular market.
- Improve the frontend with clearer input guidance and more detailed prediction explanations.

## 15. Conclusion

This project demonstrates the integration of machine learning models into a full-stack web application. It includes a classification task for stroke prediction, a regression task for house price estimation, model evaluation, saved models using Joblib, a FastAPI backend, and an HTML-based frontend.

Additional classification models were trained and compared to study the algorithms covered in class. The results highlight the importance of selecting suitable evaluation metrics, especially when the dataset contains imbalanced classes.

The project provides a practical example of how machine learning models can be developed, evaluated, saved, and used through a web-based application.