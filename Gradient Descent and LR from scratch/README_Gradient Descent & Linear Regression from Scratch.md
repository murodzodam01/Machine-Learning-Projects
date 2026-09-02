# Gradient Descent & Linear Regression from Scratch

This project implements **gradient descent and linear regression from scratch using NumPy**, without relying on scikit-learn's regression implementation.

The main goal is to understand what happens under the hood when training a linear regression model. The notebook starts with the mathematical formulation of Mean Squared Error (MSE) and its gradient, then builds gradient descent step by step. It later extends the implementation to stochastic gradient descent, learning-rate decay, regularization, and alternative loss functions.

The project finishes with an experiment comparing gradient descent with **simulated annealing** as an alternative optimization method.

## Project Overview

The notebook is divided into two main parts:

### Part 1 — Gradient Descent

- Implement MSE loss from scratch
- Calculate the analytical gradient
- Implement full-batch gradient descent
- Visualize the optimization trajectory
- Study the effect of the learning rate
- Implement stochastic gradient descent (SGD)
- Investigate the relationship between learning rate and batch size
- Add a decaying learning-rate schedule
- Compare gradient descent and SGD

### Part 2 — Linear Regression

- Build a custom `LinearRegression` class
- Train it using the gradient descent implementation
- Apply it to the UCI Automobile dataset
- Preprocess numerical and categorical variables
- Investigate overfitting
- Implement L2 regularization from scratch
- Tune the regularization coefficient
- Implement Huber loss
- Compare different loss functions
- Compare gradient descent with simulated annealing

## 1. MSE Loss from Scratch

The project starts by implementing **Mean Squared Error (MSE)** as a custom loss function.

For a linear model, the loss is defined as:

$$
Q(w, X, y) =
\frac{1}{\ell}
\sum_{i=1}^{\ell}
(\langle x_i,w\rangle-y_i)^2
$$

In matrix form:

$$
Q(w,X,y)=\frac{1}{\ell}\|Xw-y\|^2
$$

The corresponding gradient is:

$$
\nabla_w Q(w,X,y)
=
\frac{2}{\ell}X^T(Xw-y)
$$

Both the loss and its gradient are implemented manually in the `MSELoss` class.

The implementation is also checked on a small synthetic example to make sure the calculated loss and gradient match the expected values.

## 2. Gradient Descent from Scratch

Once the loss and gradient are available, the next step is implementing the actual optimization algorithm.

A single gradient descent update is:

$$
w^t =
w^{t-1}
-
\eta\nabla_w Q(w^{t-1},X,y)
$$

where:

- $w$ — model parameters
- $\eta$ — learning rate
- $Q$ — loss function

The custom `gradient_descent()` function stores the complete trajectory of the weights, which makes it possible to visualize how the algorithm moves toward the minimum.

### Synthetic Dataset

To study the optimization process, a synthetic two-feature dataset is generated with a known underlying weight vector.

The dataset contains:

- **300 observations**
- **2 features**
- Normally distributed noise

The model starts from randomly initialized weights and gradually moves toward the region that minimizes the MSE.

## 3. Effect of the Learning Rate

Several learning rates are tested:

```text
0.0001
0.0005
0.001
0.005
0.01
0.1
```

The experiments show that the learning rate has a major impact on convergence:

- **Very small learning rates** make the optimization extremely slow.
- A reasonable learning rate allows the model to approach the minimum efficiently.
- A slightly larger learning rate can cause noticeable oscillations.
- **Very large learning rates** can cause the algorithm to overshoot the minimum and fail to converge.

The notebook visualizes these behaviors using the trajectory of the weight vector over the loss surface.

## 4. Stochastic Gradient Descent

The project then extends the optimization algorithm to **Stochastic Gradient Descent (SGD)**.

Instead of calculating the gradient using the entire dataset at every iteration, SGD randomly samples a mini-batch:

```python
batch_indices = np.random.choice(
    X.shape[0],
    size=batch_size,
    replace=False
)
```

The gradient is then calculated using only this subset of observations.

This makes each update cheaper, but introduces some randomness into the optimization path.

### Learning Rate vs. Batch Size

The notebook experiments with different combinations of:

**Learning rates:**

```text
0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1
```

**Batch sizes:**

```text
3, 5, 10, 25, 50, 100
```

The main observations are:

- Very small learning rates do not allow SGD to converge quickly.
- Small batches create a noisier optimization trajectory.
- Increasing the batch size reduces the amount of oscillation.
- Very large learning rates can prevent convergence.
- SGD can still approach the optimum while using only a fraction of the observations at each iteration.

## 5. Learning-Rate Decay

SGD can benefit from using larger steps at the beginning of optimization and smaller steps later.

The notebook implements a decaying learning rate:

$$
\eta_t =
\lambda
\left(
\frac{s_0}{s_0+t}
\right)^p
$$

where:

- $\lambda$ is the initial learning rate
- $s_0=1$
- $t$ is the current iteration
- $p$ controls how quickly the learning rate decreases

Different values of `p` are tested between `0.1` and `1`.

The experiments show that:

- If `p` is too small, the early steps can be too large and noisy.
- A suitable value allows the algorithm to move quickly at first and then stabilize.
- If `p` is too large, the learning rate can decrease too quickly, preventing the model from converging effectively.

## 6. Gradient Descent vs. SGD

The notebook compares the loss trajectory of full-batch gradient descent and stochastic gradient descent.

Using the same general settings, SGD initially has a higher error because of the randomness introduced by mini-batches. However, after roughly the first several iterations, its loss approaches the performance of full-batch gradient descent.

This experiment demonstrates the main trade-off:

> SGD introduces noise into the optimization process, but each update is based on less data.

For this particular experiment, SGD provides a reasonable alternative to full-batch gradient descent.

## 7. Linear Regression from Scratch

The second part of the project builds a custom `LinearRegression` class with an interface similar to `scikit-learn`.

The class contains two main methods:

```python
fit()
predict()
```

The `fit()` method:

1. Adds a column of ones for the intercept.
2. Initializes the model weights.
3. Runs the custom gradient descent algorithm.
4. Stores the final weight vector.

The `predict()` method uses the fitted weights to generate predictions.

This means the regression model is trained entirely using the optimization code developed earlier in the notebook.

## 8. UCI Automobile Dataset

The custom regression model is tested on the classic **UCI Automobile dataset**.

The target variable is automobile price.

Before training, the data is prepared by:

- Splitting the data into training and test sets
- Handling missing values
- Separating numerical and categorical features
- One-hot encoding categorical variables
- Scaling numerical features using `MinMaxScaler`

After preprocessing, the custom linear regression model is trained using gradient descent.

### Baseline Results

The unregularized model achieves:

| Dataset | MSE |
|---|---:|
| Training | 5,702,570 |
| Test | 13,321,284 |

The considerably higher test error indicates that the model is **overfitting** the training data.

## 9. L2 Regularization

To address overfitting, the project implements **L2 regularization** from scratch.

The regularized objective is:

$$
Q(w,X,y)
=
\frac{1}{\ell}\|Xw-y\|^2
+
\lambda\|w\|^2
$$

The corresponding gradient is:

$$
\nabla_w Q(w,X,y)
=
\frac{2}{\ell}X^T(Xw-y)
+
2\lambda w
$$

The implementation also makes sure that the **intercept term is not regularized**.

Different regularization coefficients are tested:

```text
0.001
0.005
0.01
0.05
0.1
0.5
1
2
5
10
```

For this dataset and implementation, adding L2 regularization does **not** improve the test-set MSE. The best tested coefficient is `0.001`, but its test MSE is still slightly higher than the unregularized model.

This is an important result: regularization can help with overfitting in general, but it does not guarantee better test performance for every dataset or hyperparameter choice.

## 10. Huber Loss

The project also implements **Huber Loss**.

Huber Loss behaves like squared error for small prediction errors but becomes less sensitive to large errors. This makes it potentially useful when the dataset contains outliers.

The notebook applies Huber Loss to the automobile dataset and compares its performance with the standard MSE-based regression.

With `eps=100`, the results are:

| Model | Train MSE | Test MSE |
|---|---:|---:|
| MSE Loss | 5,702,570 | 13,321,284 |
| Huber Loss | 50,416,187 | 99,744,126 |

For this experiment, Huber Loss performs substantially worse than the standard MSE approach.

This suggests that, despite its robustness to large errors, Huber Loss is not always the best choice for predictive accuracy on a particular dataset.

## 11. Simulated Annealing

The final experiment explores **simulated annealing** as an alternative optimization method.

Instead of following the gradient, the algorithm generates candidate weight vectors by adding random perturbations and evaluates whether the new solution should be accepted.

Different degrees of freedom for the random perturbation are tested:

```text
1, 2, 3, 4, 5, 6
```

The results are compared with the gradient-descent model.

The simulated annealing implementation performs significantly worse on this problem. For example, with one degree of freedom, the final MSE is approximately:

| Method | Train MSE | Test MSE |
|---|---:|---:|
| Simulated Annealing | 181,413,321 | 284,944,969 |
| Gradient Descent | 5,702,570 | 13,321,284 |

The optimization trajectories are also plotted to visually compare how the two methods approach the solution.

For this linear regression problem, gradient descent is clearly more effective.

## Key Takeaways

This project demonstrates several important ideas behind machine learning optimization:

- Gradient descent can be implemented using only NumPy and the mathematical definition of a gradient.
- The learning rate strongly affects both the speed and stability of convergence.
- SGD introduces noise into the optimization trajectory by using random mini-batches.
- Larger mini-batches generally produce smoother SGD trajectories.
- Learning-rate decay can help balance fast initial progress with more precise optimization later.
- A custom linear regression model can be built on top of a general loss-function interface.
- L2 regularization can be incorporated by modifying both the loss and its gradient.
- Huber Loss provides a more robust alternative to MSE, but it does not necessarily produce better predictions.
- Alternative optimization methods such as simulated annealing can be explored, but gradient-based optimization is much more effective for this particular linear regression problem.

## Technologies & Libraries

The project is implemented in Python using:

- Python
- NumPy
- pandas
- Matplotlib
- scikit-learn

The main implementation is written from scratch using NumPy rather than relying on a pre-built linear regression estimator.

## Project Structure

```text
.
├── Gradient_Descent_and_Linear_Regression_from_scratch.ipynb
└── README.md
```

## How to Run

1. Clone the repository:

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

2. Install the required dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn
```

3. Open the notebook:

```bash
jupyter notebook Gradient_Descent_and_Linear_Regression_from_scratch.ipynb
```

You can also run the notebook in Google Colab.

## Data

The project uses:

- A synthetic dataset for studying gradient descent and SGD
- The **UCI Automobile dataset** for testing the custom linear regression implementation

The Automobile dataset is loaded directly in the notebook from the UCI Machine Learning Repository.