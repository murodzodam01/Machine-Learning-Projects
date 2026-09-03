# Tweet Classification: Real Disaster or Not?

This project builds a text classifier that predicts whether a tweet is
about a real disaster or incident.

The notebook uses the **Kaggle "Real or Not? NLP with Disaster Tweets"**
dataset and focuses on a practical NLP workflow: exploring the data,
building custom text preprocessing, comparing different vectorization
methods, and training a logistic regression classifier.

The main goal is to see how different ways of representing text affect
model quality and the size of the resulting feature matrix.

## Dataset

The project uses the `train.csv` file from Kaggle's [Real or Not? NLP
with Disaster
Tweets](https://www.kaggle.com/competitions/nlp-getting-started/data)
competition.

The dataset contains **7,613 tweets** with the following columns:

-   `id` --- tweet identifier
-   `keyword` --- keyword associated with the tweet
-   `location` --- location mentioned in the tweet
-   `text` --- tweet text
-   `target` --- target variable: `1` for a real disaster and `0`
    otherwise

The data is split into training and test sets using a 70/30 split with
`random_state=42`.

After the split, the training set contains:

-   **2,305** positive examples
-   **3,024** negative examples

Missing categorical values are replaced with empty strings, and the
`keyword`, `location`, and `text` columns are combined into a single
text field.

## What I explored

### 1. Initial data exploration

I first looked at the class distribution and the relationship between
the most common keywords and the target variable.

This helps give a quick idea of how balanced the problem is and whether
some keywords are more strongly associated with real disaster-related
tweets.

### 2. CountVectorizer

The first text representation uses `CountVectorizer`.

This produces a sparse matrix with:

-   **5,329 training tweets**
-   **18,455 features**
-   **86,671 stored values**

I also inspected the vocabulary to see how many tokens contained digits,
punctuation, hashtags, and mentions.

### 3. TweetTokenizer

Instead of relying on the default tokenization, I experimented with
NLTK's `TweetTokenizer`, which is designed specifically for tweets.

This makes a difference because tweets contain a lot of things that
normal tokenizers do not handle especially well, such as:

-   mentions
-   hashtags
-   URLs
-   punctuation
-   short informal words

### 4. Custom tokenizer

I then built my own tokenizer using several preprocessing steps:

-   convert text to lowercase
-   tokenize with `TweetTokenizer`
-   remove stop words
-   keep alphabetic tokens and selected punctuation
-   keep hashtags
-   apply Snowball stemming

The resulting representation was used with logistic regression.

### 5. CountVectorizer + Logistic Regression

Using the custom tokenizer with `CountVectorizer` produced:

-   **9,159 features**
-   **F1-score: 0.742**

This gives a solid baseline, but there is still room to improve the
representation of the text.

### 6. TF-IDF

Next, I replaced raw word counts with **TF-IDF**.

This slightly improved the result:

  Method                                 Features   F1-score
  ------------------------------------ ---------- ----------
  CountVectorizer + custom tokenizer        9,159      0.742
  TF-IDF + custom tokenizer                 9,159      0.749

TF-IDF gives more weight to words that are important for particular
documents and less weight to words that appear everywhere.

I also experimented with `max_df=0.9`. In this case, it did not change
the feature count or the model quality.

### 7. Reducing the feature matrix

I then used `min_df=0.0004` with `TfidfVectorizer` to remove very rare
words.

This reduced the number of features considerably:

-   **2,971 features**
-   **F1-score: 0.752**

So the model became smaller while slightly improving its performance.

### 8. HashingVectorizer

Finally, I tried the hashing trick with only **5,000 features**.

It was the fastest way to control the size of the feature space, and it
also avoids storing an explicit vocabulary.

However, in this experiment it performed worse than TF-IDF:

-   **5,000 features**
-   **F1-score: 0.726**

So reducing the feature space too aggressively came at the cost of
classification quality.

## Final result

The best result in the notebook came from:

**TF-IDF + custom tokenizer + `min_df=0.0004` + Logistic Regression**

  Metric                    Result
  ---------------- ---------------
  Feature matrix     5,329 × 2,971
  Accuracy                    0.80
  F1-score               **0.752**

For the positive class (`target = 1`):

-   Precision: **0.79**
-   Recall: **0.72**
-   F1-score: **0.75**

The final model reached the target of at least **0.75 F1-score**.

## Main takeaways

A few things stood out from the experiments:

-   A good tokenizer matters for short and noisy text such as tweets.
-   TF-IDF performed better than simple word counts.
-   Removing very rare words reduced the feature matrix while slightly
    improving the F1-score.
-   HashingVectorizer was useful for limiting dimensionality, but
    performed worse in this experiment.
-   The final model is relatively simple --- **Logistic Regression** ---
    but the text representation has a large impact on the result.

Overall, the best approach was not the one with the fewest features, but
the one that kept useful information while removing a lot of unnecessary
vocabulary.

## Technologies

-   Python
-   pandas
-   NumPy
-   scikit-learn
-   NLTK
-   Matplotlib
-   Seaborn

## Project structure

``` text
NLP Classification/
│
├── NLP_Classification.ipynb
├── train.csv
└── README.md
```

## How to run

Clone the repository and install the required libraries:

``` bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn
```

Then open the notebook:

``` bash
jupyter notebook NLP_Classification.ipynb
```

The notebook contains the complete preprocessing, feature engineering,
model training, evaluation, and comparison of different NLP approaches.

## Notes

The dataset comes from the Kaggle competition **Real or Not? NLP with
Disaster Tweets**.

The notebook is mainly focused on understanding the NLP pipeline and
comparing text representations rather than building a highly optimized
competition solution.
