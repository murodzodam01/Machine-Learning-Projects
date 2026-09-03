# Boosting & Clustering: From-Scratch Implementations

This project explores two machine learning topics from the ground up:
**gradient boosting** and **DBSCAN clustering**.

The main idea was to understand how these algorithms work internally
rather than treating them as black boxes. I implemented gradient
boosting and DBSCAN from scratch, compared the results with
scikit-learn, and then applied boosting to a real flight-delay
classification problem.

## What's inside

The notebook is split into two main parts:

1.  **Boosting**
    -   Gradient Boosting for regression implemented from scratch
    -   Optional optimization of the step size for each boosting
        iteration
    -   Comparison with Random Forest
    -   Flight-delay prediction with XGBoost
2.  **Clustering**
    -   DBSCAN implemented from scratch
    -   Comparison with scikit-learn's DBSCAN
    -   Experiments on different synthetic datasets
    -   DBSCAN applied to flight data using `Distance` and `DepTime`

------------------------------------------------------------------------

## 1. Gradient Boosting From Scratch

### How it works

The implementation follows the basic gradient boosting idea:

-   Start with an initial prediction.
-   Train a decision tree on the current residuals.
-   Add the tree's predictions to the existing ensemble.
-   Repeat this process for a fixed number of trees.
-   Scale each new tree using the learning rate.

Two versions are implemented.

### Basic Gradient Boosting

`GradientBoosting1` uses a fixed step size:

``` text
prediction += learning_rate × tree_prediction
```

The trees are `DecisionTreeRegressor` models with a configurable maximum
depth.

### Gradient Boosting with an optimized step size

The second implementation, `GradientBoosting`, also searches for an
optimal coefficient `γ` at every iteration using
`scipy.optimize.minimize`.

This gives each tree its own weight instead of using exactly the same
step size throughout the ensemble.

------------------------------------------------------------------------

## Housing Price Prediction

The boosting implementations were tested on the Boston Housing-style
dataset (`HousingData.csv`).

The data is split into training and test sets using a 75/25 split with
`random_state=13`.

### Results

I used a Random Forest as a baseline and then compared it with both
versions of the custom gradient boosting model.

  Model                                         Test MSE
  ----------------------------------------- ------------
  Random Forest                                   5.8111
  Gradient Boosting from scratch              **5.2537**
  Gradient Boosting + optimized step size     **5.2485**

The from-scratch gradient boosting implementation performed better than
the Random Forest baseline on this test split.

The optimized step-size version gave a small additional improvement.

### Parameters used

For the boosting models:

-   `n_estimators = 50`
-   `max_depth = 4`
-   `learning_rate = 0.2`

------------------------------------------------------------------------

## 2. Flight Delay Prediction

The next part uses flight information to predict whether a flight will
be delayed by at least 15 minutes.

The target variable is:

-   `Y` → delayed
-   `N` → not delayed

The model uses:

-   `Distance`
-   `DepTime`
-   `Month`
-   `DayofMonth`
-   `DayOfWeek`
-   `UniqueCarrier`
-   `Origin`
-   `Dest`

Categorical variables are converted using one-hot encoding.

### XGBoost

For the classification task, I used `XGBClassifier` from XGBoost.

The experiment uses:

``` python
learning_rate = 0.1
n_estimators = 150
```

The validation ROC-AUC obtained in the notebook was:

**0.7422**

The notebook uses a train/validation split and evaluates the model using
predicted probabilities and ROC-AUC.

------------------------------------------------------------------------

# 3. DBSCAN From Scratch

The second major part of the notebook implements **DBSCAN (Density-Based
Spatial Clustering of Applications with Noise)**.

DBSCAN groups points based on the density of their neighborhoods rather
than requiring the number of clusters in advance.

The implementation uses two main parameters:

-   `eps` --- radius used to define a neighborhood
-   `min_samples` --- minimum number of points required to form a dense
    region

The implementation also distinguishes between:

-   core points
-   reachable points
-   noise/outliers

### Custom implementation

The `dbscan` class contains:

-   `fit_predict()` for creating cluster labels
-   `grow_cluster()` for expanding a cluster
-   `range_query()` for finding neighboring points
-   a distance function based on `scipy.spatial.distance`

------------------------------------------------------------------------

## Testing DBSCAN

I first tested the implementation on synthetic datasets where the
cluster structure is easy to visualize.

### Two moons

The custom DBSCAN was tested on a noisy two-moons dataset and compared
directly with scikit-learn's `DBSCAN`.

The goal was to check whether the from-scratch version could recover the
same kind of non-linear cluster structure as the reference
implementation.

### More complex shapes

I also experimented with combinations of:

-   circles
-   moons
-   S-curves
-   Swiss-roll projections
-   blobs

These examples show how DBSCAN can handle irregularly shaped clusters.

------------------------------------------------------------------------

## Runtime comparison

I also compared the runtime of the custom implementation with
scikit-learn's version.

For the run recorded in the notebook:

  Implementation          Wall time
  --------------------- -----------
  scikit-learn DBSCAN       6.82 ms
  Custom DBSCAN              4.50 s

The custom implementation was roughly **660× slower** in this particular
run.

This is a useful example of the difference between implementing an
algorithm for learning purposes and building an optimized production
implementation.

------------------------------------------------------------------------

# 4. Applying DBSCAN to Flight Data

Finally, DBSCAN was applied to the flight-delay dataset using only:

-   `Distance`
-   `DepTime`

The notebook visualizes the resulting clusters and then experiments with
different combinations of `eps` and `min_samples`.

The tested values include:

``` python
eps = [0.1, 0.5, 1, 5, 10, 50, 100, 250, 500, 1000]
min_samples = [3, 4, 5, 10, 15, 25]
```

This makes it possible to see how sensitive DBSCAN is to its two main
hyperparameters.

------------------------------------------------------------------------

# Main Takeaways

A few things stood out from the experiments:

-   A relatively simple gradient boosting implementation can perform
    very well compared with a Random Forest baseline.
-   Optimizing the step size for each boosting iteration gave a small
    improvement over using a fixed step size.
-   Gradient boosting works by repeatedly fitting new trees to what the
    current ensemble is still getting wrong.
-   DBSCAN is useful when clusters have irregular shapes or when the
    data contains noise.
-   The custom DBSCAN reproduced the basic clustering behavior of the
    reference implementation, but it was much slower.
-   The choice of `eps` and `min_samples` has a large effect on how
    DBSCAN separates clusters and labels noise.

The main purpose of the project was not just to get the best score, but
to understand what is happening inside these algorithms and then compare
the result with established implementations.

------------------------------------------------------------------------

## Technologies

-   Python
-   NumPy
-   pandas
-   scikit-learn
-   SciPy
-   XGBoost
-   Matplotlib

## Project structure

``` text
Boosting & Clustering/
│
├── Boosting_clustering_scratch.ipynb
├── HousingData.csv
└── README.md
```

## How to run

Install the required packages:

``` bash
pip install numpy pandas scipy scikit-learn xgboost matplotlib
```

Then open the notebook:

``` bash
jupyter notebook Boosting_clustering_scratch.ipynb
```

The notebook contains the complete implementations, experiments,
visualizations, model comparisons, and evaluation results.

## Data

The notebook uses two main datasets:

-   `HousingData.csv` for the regression experiments
-   `flight_delays_train.csv` and `flight_delays_test.csv` for the
    flight-delay classification and clustering experiments

The datasets are loaded from the URLs used in the notebook, so an
internet connection is needed when running those cells.
