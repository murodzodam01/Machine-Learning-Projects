# kNN Classification & Linear Regression

This project explores two classic machine learning approaches: **k-Nearest Neighbors (kNN)** for classification and **linear regression** for predicting continuous outcomes.

The notebook focuses on understanding how model parameters and feature selection affect predictions. In the first part, I investigate how changing `k` changes the decision boundary of a kNN classifier. In the second part, I use linear regression to predict diamond prices and explore how **multicollinearity**, **Ridge**, and **Lasso** regularization affect the model.

## Project Overview

The project is divided into two main parts:

1. **kNN Classification**
   - Use the Wine dataset from `scikit-learn`
   - Train kNN classifiers with different numbers of neighbors
   - Compare training and test accuracy
   - Visualize how the decision boundary changes as `k` increases

2. **Linear Regression & Regularization**
   - Use the Diamonds dataset
   - Explore relationships between numerical features and diamond price
   - Encode categorical variables
   - Train a linear regression model
   - Investigate multicollinearity
   - Compare Ridge and Lasso regularization
   - Examine how regularization changes model coefficients
   - Evaluate models using Mean Squared Error (MSE)

## 1. kNN Classification

The first part uses the built-in **Wine dataset**, which contains chemical measurements for three different wine varieties. The dataset has 178 observations and 13 numerical features.

Before training the models, the data is checked for missing values. No missing values are found, and all features are numerical, so no categorical encoding is required.

### Model setup

The data is split into training and test sets, and kNN classifiers are trained using two features:

- `alcohol`
- `magnesium`

Before applying kNN, both features are standardized using `StandardScaler`. Six different values of `k` are tested:

```text
k = 1, 3, 5, 10, 15, 25
```

The resulting train and test accuracies are:

| k | Train Accuracy | Test Accuracy |
|---:|---:|---:|
| 1 | 1.00 | 0.67 |
| 3 | 0.81 | 0.78 |
| 5 | 0.75 | 0.72 |
| 10 | 0.73 | 0.78 |
| 15 | 0.73 | 0.80 |
| 25 | 0.67 | 0.80 |

These results illustrate the effect of `k` on model complexity. With `k = 1`, the classifier perfectly fits the training data but performs worse on the test set. As `k` increases, the decision boundary becomes smoother and the gap between training and test performance decreases.

The decision regions for all six models are also visualized using `mlxtend`, making it easier to see how the classification boundary changes with different values of `k`.

## 2. Diamond Price Prediction

The second part focuses on predicting diamond prices using the **Diamonds dataset**, which contains 53,940 observations. The target variable is `price`.

The dataset contains information about:

- Carat
- Cut
- Color
- Clarity
- Depth
- Table
- Price
- Dimensions (`x`, `y`, `z`)

The dataset contains no missing values. The `Unnamed: 0` column is removed because it does not contain useful predictive information.

### Feature relationships

A correlation matrix is calculated for the numerical variables.

The features most strongly correlated with diamond price are:

- `carat`: **0.922**
- `x`: **0.884**
- `y`: **0.865**
- `z`: **0.861**

At the same time, these variables are also strongly correlated with each other. For example, `carat` and `x` have a correlation of approximately **0.975**. This provides evidence of substantial **multicollinearity** in the dataset.

### Data preprocessing

The categorical variables `cut`, `color`, and `clarity` are converted into dummy variables using one-hot encoding. This produces a dataset with 26 predictor variables.

The data is then split into training and test sets, and the features are standardized using `StandardScaler`.

## Linear Regression

A standard linear regression model is fitted to the processed dataset.

The resulting Mean Squared Error values are:

| Dataset | MSE |
|---|---:|
| Training | 1,284,662 |
| Test | 1,259,159 |

The model's coefficients are then examined. Some coefficients, particularly those associated with the encoded categorical variables, are substantially larger in magnitude than others. Combined with the strong correlations between several numerical predictors, this motivates the use of regularization.

## Ridge & Lasso Regularization

To address multicollinearity, two regularized regression methods are considered:

- **Ridge Regression (L2 regularization)**
- **Lasso Regression (L1 regularization)**

Both models are initially fitted with:

```python
alpha = 10
```

Regularization shrinks the coefficient values and makes the model less sensitive to highly correlated predictors. Lasso has an additional advantage: because of its L1 penalty, it can shrink some coefficients all the way to zero, effectively performing feature selection.

The notebook also compares how the overall coefficient magnitude changes for different values of `alpha`:

```text
0.1, 1, 10, 100, 200
```

This provides a visual comparison of how aggressively Ridge and Lasso shrink the model weights.

## Results

An important finding from the analysis is that **regularization does not automatically improve predictive performance**.

For this particular train/test split, the Lasso model achieves a test MSE of approximately:

```text
1,528,321
```

compared with approximately:

```text
1,259,159
```

for the unregularized linear regression model.

Therefore, the plain linear regression model performs better in terms of test-set MSE in this experiment.

However, Lasso remains useful when the goal is not only prediction but also **feature selection and model simplification**. Its ability to shrink coefficients to zero can help identify a smaller set of potentially useful predictors.

## Key Takeaways

- The choice of `k` has a clear impact on the complexity of a kNN classifier.
- Small values of `k` can lead to overfitting, while larger values produce smoother decision boundaries.
- Standardization is important for distance-based algorithms such as kNN.
- Diamond price is strongly related to `carat`, `x`, `y`, and `z`.
- Several numerical features are highly correlated with each other, creating a multicollinearity problem.
- One-hot encoding allows categorical diamond characteristics to be incorporated into a linear regression model.
- Ridge and Lasso can reduce the impact of multicollinearity through regularization.
- Lasso can also perform feature selection by shrinking coefficients toward zero.
- In this particular experiment, the unregularized linear regression model achieved the lowest test MSE.

## Technologies & Libraries

The project is implemented in Python using:

- Python
- NumPy
- pandas
- Matplotlib
- scikit-learn
- mlxtend

## Project Structure

```text
.
├── Knn_LinReg.ipynb
└── README.md
```

## How to Run

1. Clone the repository:

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

2. Install the required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn mlxtend
```

3. Open the notebook:

```bash
jupyter notebook Knn_LinReg.ipynb
```

You can also run the notebook directly in **Google Colab**.

## Data

The project uses two datasets:

- **Wine dataset** — provided through `scikit-learn`
- **Diamonds dataset** — the dataset used in the original notebook

The notebook contains the corresponding data-loading steps and preprocessing.