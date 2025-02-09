<h1 align="center">MBA Candidate Targeting for Katz School of Business</h1>

<div>
  <table align="center">
    <tr>
      <td colspan="2" align="center" style="background-color: white; color: black;"><strong>Table of Contents</strong></td>
    </tr>
    <tr>
      <td style="background-color: white; color: black; padding: 10px;">1. <a href="#project-objective" style="color: black;">Project Objective</a></td>
      <td style="background-color: gray; color: black; padding: 10px;">6. <a href="#gender-bias-assessment" style="color: black;">Gender Bias Assessment</a></td>
    </tr>
    <tr>
      <td style="background-color: gray; color: black; padding: 10px;">2. <a href="#data-overview" style="color: black;">Data Overview</a></td>
      <td style="background-color: white; color: black; padding: 10px;">7. <a href="#results" style="color: black;">Results</a></td>
    </tr>
    <tr>
      <td style="background-color: white; color: black; padding: 10px;">3. <a href="#methodology" style="color: black;">Methodology</a></td>
      <td style="background-color: gray; color: black; padding: 10px;">8. <a href="#conclusion" style="color: black;">Conclusion</a></td>
    </tr>
    <tr>
      <td style="background-color: gray; color: black; padding: 10px;">
        4. <a href="#model-selection" style="color: black;">Model Selection</a><br>
        &nbsp;&nbsp;&nbsp;- <a href="#tree-based-models" style="color: black;">Tree-Based Models</a><br>
        &nbsp;&nbsp;&nbsp;- <a href="#k-nearest-neighbors-knn" style="color: black;">K-Nearest Neighbors (KNN)</a><br>
        &nbsp;&nbsp;&nbsp;- <a href="#support-vector-machines-svm" style="color: black;">Linear Models</a><br>
        &nbsp;&nbsp;&nbsp;- <a href="#xgboost" style="color: black;">XGBoost</a>
      </td>
      <td style="background-color: gray; color: black; padding: 10px;">
        5. <a href="#evaluation-metrics" style="color: black;">Evaluation Metrics</a><br>
        &nbsp;&nbsp;&nbsp;- <a href="#why-f1-score-is-more-useful" style="color: black;">Why F1 Score Is More Useful</a><br>
        &nbsp;&nbsp;&nbsp;- <a href="#evaluation-of-f1-score-for-yes-and-no-categories" style="color: black;">Evaluation of F1 Score for Yes and No Categories</a>
      </td>
    </tr>
  </table>
</div>


## Project Objective

This project aims to build a predictive model that forecasts whether a graduate is likely to pursue an MBA, assesses potential gender bias in the MBA admissions process, and provides strategies to mitigate any identified disparities. The model leverages both categorical and numerical features to predict MBA pursuit and offers insights into the factors that influence this decision.



The core objectives of this analysis are:
1. **Predict which graduates are most likely to pursue an MBA** based on their academic background, work experience, financial factors, and other features.
2. **Assess potential gender bias** in the decision to pursue an MBA, ensuring that the model treats both genders fairly and does not disproportionately favor one gender.
3. **Provide strategies to mitigate any gender disparities**, suggesting ways to create a more equitable recruitment process in MBA programs.

## Data Overview

The dataset used in this analysis includes both **categorical** and **numerical** features that capture various aspects of each graduate's profile:

- **Categorical Features**: Gender, academic background, and location preferences.
- **Numerical Features**: Age, years of work experience, undergraduate GPA, GRE/GMAT scores.

### Key Observations:
- **Uniform Distribution**: The numerical features like age and years of work experience follow a **uniform distribution**, meaning that values are spread evenly across a range. While this can make it difficult for models to detect strong patterns, it also means that data points are independent of each other.
- **Imbalanced Gender Distribution**: The dataset contains more male graduates than female graduates, and significantly fewer observations for other gender categories. This imbalance introduces challenges in model training, as it could lead to biased predictions if not properly handled.

### Data Preprocessing:
- The dataset did not contain any missing values, eliminating the need for imputation techniques.
- Categorical variables (such as gender and academic background) were **one-hot encoded** using the `get_dummies` function, ensuring that the data could be interpreted by machine learning models.

## Methodology

### Model Selection
A variety of machine learning models were evaluated for predicting the likelihood of pursuing an MBA, including:

- **Tree-based models** (e.g., Decision Trees, XGBoost)
- **Support Vector Machines (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Regularized Regression** (e.g., Ridge, Lasso)
- **Gaussian Naive Bayes** (was excluded due to assumptions about normal distribution, which did not align with the data's uniform distribution)

#### Tree-Based Models
Decision trees are particularly useful in this context due to their ability to capture complex, non-linear relationships in the data. Furthermore, decision trees help identify the most important features driving the decision to pursue an MBA.

#### K-Nearest Neighbors (KNN)
KNN, although an intuitive and simple classifier, suffered from over-prediction issues due to the **imbalanced dataset**, often predicting that most students would pursue an MBA. This bias made it less effective in predicting the students who would not pursue an MBA.

#### Support Vector Machines (SVM)
Support Vector Machines are powerful for linear separability, but they struggled to capture the complexities of this dataset. While they worked well in some cases, they could not handle non-linear relationships as effectively as tree-based models.

#### XGBoost
XGBoost, a gradient-boosting algorithm that builds an ensemble of decision trees, showed the best performance in capturing the complexity of the dataset. It provides an accurate model by learning from errors in previous trees, making it a good fit for this task.

### Cross-Validation:
To ensure model generalizability, **5-fold cross-validation** was used. This technique splits the dataset into five subsets and trains/test the model on each fold to minimize overfitting and to get a reliable estimate of model performance.

### Evaluation Metrics:
While **accuracy** is often a common metric for evaluating models, **accuracy alone is not the best metric in this case** due to the **imbalanced nature of the dataset**. In our dataset, the number of students who choose to pursue an MBA is much larger than those who do not. If the model predicts "Pursue MBA" for all students, it would still achieve a high accuracy because the majority class is "Pursue MBA." This would not be a meaningful result, as the model wouldn't be effectively identifying those who are likely to pursue an MBA.

```
from sklearn.metrics import classification_report

# Display classification report for precision, recall, and F1 score
print(classification_report(y_test, y_pred))
```

#### Why F1 Score Is More Useful:
Instead of relying on accuracy, **F1 Score** was used as a more informative metric. The F1 Score is the harmonic mean of **precision** and **recall**

  #### Evaluation of F1 Score for Yes and No Categories:
  The **F1 score** was evaluated separately for the **Yes** and **No** categories, with a particular focus on the **No** category. This was especially important because the **No** category suffered from a lower sample size, which made it more challenging for the model to predict accurately. The model struggled to identify those students who were unlikely to pursue an MBA due to the underrepresentation of the **No** category in the training data. By evaluating the **F1 score** separately for both categories, we were able to identify this discrepancy in performance and understand where the model needed improvement.

Since our goal is to better understand which students are most likely to pursue an MBA, focusing on both precision and recall gives a more accurate picture of model performance, particularly when the data is imbalanced.

### Addressing Gender Bias:
To assess gender bias, the model predictions were examined based on gender. We specifically:
1. Grouped the predictions by **gender** and compared the likelihood of each gender being predicted to pursue an MBA.
2. Calculated the **Disparate Impact (DI) Ratio**, which compares the predicted likelihood for women versus men. A DI ratio near 1 indicates no gender bias.

## Results

### Model Performance
- **XGBoost** provided the most robust predictions with an overall accuracy of **63%**. The recall for predicting students who would **not** pursue an MBA was **0.43**, indicating that the model, while successful at predicting those who would pursue an MBA, struggled more with identifying those who would not.

#### Feature Importance:
The model highlighted several key features that influenced a graduate's likelihood to pursue an MBA:
1. **Location preference**: Graduates with a preference for staying within their home country were more likely to pursue an MBA.
2. **Undergraduate major**: Graduates with a background in business or related fields were more inclined to pursue an MBA compared to those in arts, sciences, or engineering.
3. **Financial considerations**: The ability to fund the MBA, whether through scholarships, loans, or self-funding, played a significant role.
4. **Career goals**: Aspirations for career advancement or entrepreneurship were strong motivators for pursuing further education.

On the other hand, features like **gender**, **online versus on-campus MBA preference**, and **academic performance metrics** (e.g., GRE/GMAT scores, GPA) were less influential in the model's predictions.

### Gender Bias Assessment:
- The **DI Ratio** was approximately **1**, meaning that the model predicted both **men** and **women** equally (approximately **73%** for both genders). This indicates that there was no significant gender bias in the model's predictions.
- Gender did not feature prominently in the list of **most important predictors**, reinforcing that the model is not inherently biased towards either gender.

```
# Example: Gender bias assessment
# Calculate the probability for each gender
gender_probs = model.predict_proba(X_test)

# Extract probabilities for women and men
female_probs = gender_probs[y_test == 1, 1]  # Assuming 1 is female in the dataset
male_probs = gender_probs[y_test == 0, 1]    # Assuming 0 is male

# Disparate Impact Ratio (DI)
di_ratio = female_probs.mean() / male_probs.mean()
print(f'Disparate Impact Ratio: {di_ratio}')
```
### Considerations for Model Deployment:
- The **over-prediction of MBA pursuers** by the model raises concerns about recruitment efforts. If resources (marketing campaigns, recruitment outreach) are heavily targeted at individuals predicted to pursue an MBA, there is a risk of wasted resources on individuals who do not follow through.
- It is crucial to consider the **cost-benefit** of targeting over-predicted MBA candidates, as it can lead to inefficiency and unnecessary expenses in the recruitment process.

## Conclusion

This analysis provides valuable insights into the factors influencing a graduate's decision to pursue an MBA. The **XGBoost** model proved to be the most effective tool for capturing the complexities of the data, with a focus on financial, career-driven, and academic factors that influence MBA decisions. Gender bias was assessed and found to be minimal, with no significant disparities in the prediction of MBA pursuit between men and women.

The findings suggest that the model can be used for recruitment efforts without introducing significant gender bias. However, due to the over-prediction of MBA pursuers, careful consideration must be given to the resource allocation for targeted recruitment.

```
# Example: Making a prediction for a new candidate
new_candidate = pd.DataFrame({
    'Age': [30],
    'Years_Work_Experience': [5],
    'GPA': [3.5],
    'GRE_Score': [320],
    'Gender_Female': [1],
    'Gender_Male': [0],
    # Include other features here...
})

# Predict if the new candidate will pursue an MBA
prediction = model.predict(new_candidate)
print("Will pursue MBA:" , "Yes" if prediction[0] == 1 else "No")

```
## Technologies Used

- **Python**: Primary programming language.
- **XGBoost**: For gradient boosting and ensemble learning.
- **scikit-learn**: For data preprocessing, model building, and evaluation.
- **Pandas** and **NumPy**: For data manipulation.
- **Matplotlib** and **Seaborn**: For data visualization.

## Acknowledgments

- **XGBoost** for its gradient boosting framework.
- **scikit-learn** for machine learning algorithms.
- **Pandas** and **NumPy** for efficient data manipulation and handling.
- The dataset, which provided the foundation for this analysis.

Feel free to explore and contribute to the project. Feedback and suggestions for improvement are always welcome!
