# MBA-Candidate-Targeting-for-Katz-School-of-Business

# Predictive Model for MBA Pursuit: Gender Bias and Decision Factors

This project aims to build a predictive model that forecasts whether a graduate is likely to pursue an MBA, assesses potential gender bias in the MBA admissions process, and provides strategies to mitigate any identified disparities. The model leverages both categorical and numerical features to predict MBA pursuit and offers insights into the factors that influence this decision.

## Project Objective

The core objectives of this analysis are:
1. **Predict which graduates are most likely to pursue an MBA** based on their academic background, work experience, financial factors, and other features.
2. **Assess potential gender bias** in the decision to pursue an MBA, ensuring that the model treats both genders fairly and does not disproportionately favor one gender.
3. **Provide strategies to mitigate any gender disparities**, suggesting ways to create a more equitable recruitment process in MBA programs.

## Data Overview

The dataset used in this analysis includes both **categorical** and **numerical** features that capture various aspects of each graduate's profile:

- **Categorical Features**: Gender, academic background, and location preferences.
- **Numerical Features**: Age, years of work experience, undergraduate GPA, GRE/GMAT scores.

### Key Observations:
- **Uniform Distribution**: The numerical features like age and years of work experience follow a **uniform distribution**, meaning that values are spread evenly across a range. While this can make it difficult for models to detect strong patterns, it also means that data points are independent of each other, potentially making the prediction task more challenging.
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
While **accuracy** is often a common metric for evaluating models, **accuracy alone is not the best metric in this case** due to the **imbalanced nature of the dataset**. In our dataset, the number of students who choose to pursue an MBA is much smaller than those who do not. If the model predicts "no MBA" for all students, it would still achieve a high accuracy because the majority class is "no MBA." This would not be a meaningful result, as the model wouldn't be effectively identifying those who are likely to pursue an MBA.

#### Why F1 Score Is More Useful:
Instead of relying on accuracy, **F1 Score** was used as a more informative metric. The F1 Score is the harmonic mean of **precision** and **recall**:
- **Precision** tells us how many of the predicted MBA pursuers actually went on to pursue an MBA.
- **Recall** tells us how many of the actual MBA pursuers were correctly identified by the model.

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

### Considerations for Model Deployment:
- The **over-prediction of MBA pursuers** by the model raises concerns about recruitment efforts. If resources (marketing campaigns, recruitment outreach) are heavily targeted at individuals predicted to pursue an MBA, there is a risk of wasted resources on individuals who do not follow through.
- It is crucial to consider the **cost-benefit** of targeting over-predicted MBA candidates, as it can lead to inefficiency and unnecessary expenses in the recruitment process.

## Conclusion

This analysis provides valuable insights into the factors influencing a graduate's decision to pursue an MBA. The **XGBoost** model proved to be the most effective tool for capturing the complexities of the data, with a focus on financial, career-driven, and academic factors that influence MBA decisions. Gender bias was assessed and found to be minimal, with no significant disparities in the prediction of MBA pursuit between men and women.

The findings suggest that the model can be used for recruitment efforts without introducing significant gender bias. However, due to the over-prediction of MBA pursuers, careful consideration must be given to the resource allocation for targeted recruitment.

## Technologies Used

- **Python**: Primary programming language.
- **XGBoost**: For gradient boosting and ensemble learning.
- **scikit-learn**: For data preprocessing, model building, and evaluation.
- **Pandas** and **NumPy**: For data manipulation.
- **Matplotlib** and **Seaborn**: For data visualization.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **XGBoost** for its gradient boosting framework.
- **scikit-learn** for machine learning algorithms.
- **Pandas** and **NumPy** for efficient data manipulation and handling.
- The dataset, which provided the foundation for this analysis.

## Appendix A: Supplementary Data for Predictive Models

This appendix includes figures and tables supporting the results, offering a more in-depth look at the model’s performance, key features, and gender analysis.

---

Feel free to explore and contribute to the project. Feedback and suggestions for improvement are always welcome!
