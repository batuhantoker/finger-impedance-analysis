"""Image feature-based gesture classifier using Canny/Harris features from HD-sEMG grids.

Loads preprocessed activation-map feature data (Canny edges, Harris corners)
from a pickle file, evaluates multiple classifiers via k-fold cross-validation,
and produces a boxplot comparison.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, make_scorer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def data_preprocess(X: pd.DataFrame) -> np.ndarray:
    """Flatten and concatenate all feature columns from a DataFrame.

    Args:
        X: DataFrame where each cell contains an array-like feature vector.

    Returns:
        2D array of shape (n_samples, total_flattened_features).
    """
    x_new = np.asarray(X[X.columns[0]])
    x_sep = np.asarray([rr.reshape(-1) for rr in x_new])
    for i in range(1, len(X.columns)):
        x_new = np.asarray(X[X.columns[i]])
        x_sep1 = np.asarray([rr.reshape(-1) for rr in x_new])
        x_sep = np.hstack((x_sep, x_sep1))
    return x_sep


def classification_report_with_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification report and return accuracy score for use as a scorer."""
    classification_report(y_true, y_pred, output_dict=True)
    return accuracy_score(y_true, y_pred)


if __name__ == "__main__":
    data = pd.read_pickle("./data4s3.pkl").reset_index(drop=True)
    data = data[data.movement_id != 0]

    X = data.loc[:, data.columns != 'movement_id']
    x_new = np.asarray(X['canny_ext'])
    x_sep7 = np.asarray([rr.reshape(-1) for rr in x_new])
    x_new = np.asarray(X['canny_flex'])
    x_sep8 = np.asarray([rr.reshape(-1) for rr in x_new])
    x_new = np.asarray(X['Harris_ext'])
    x_sep11 = np.asarray([rr.reshape(-1) for rr in x_new], dtype=object)
    x_new = np.asarray(X['Harris_flex'])
    x_sep12 = np.asarray([rr.reshape(-1) for rr in x_new], dtype=object)

    harris_min = [np.min([rr.shape for rr in x_sep11]), np.min([rr.shape for rr in x_sep12])]
    x_sep11 = np.asarray([rr[0:np.min(harris_min)] for rr in x_sep11])
    x_sep12 = np.asarray([rr[0:np.min(harris_min)] for rr in x_sep12])
    x_sep = np.hstack((x_sep7, x_sep8, x_sep11, x_sep12))
    print(x_sep.shape)

    y = np.asarray(data['movement_id']).reshape(-1)
    y = y.astype('int')
    print(np.unique(y))
    print((y.shape))

    sc = StandardScaler()
    x_sep = sc.fit_transform(x_sep)

    plt.rcParams["figure.figsize"] = (20, 10)
    Y = y
    X = x_sep

    number_of_k_fold = 10
    random_seed = 42
    outcome = []
    model_names = []
    models = [
        ('LogReg', LogisticRegression()),
        # ('SVM', SVC()),
        ('DecTree', DecisionTreeClassifier()),
        ('KNN', KNeighborsClassifier(n_neighbors=15)),
        ('LinDisc', LinearDiscriminantAnalysis()),
        ('GaussianNB', GaussianNB()),
        ('MLPC', MLPClassifier(activation='relu', solver='adam', max_iter=500)),
        ('RFC', RandomForestClassifier()),
        ('ABC', AdaBoostClassifier()),
    ]
    for model_name, model in models:
        k_fold_validation = model_selection.KFold(
            n_splits=number_of_k_fold, random_state=random_seed, shuffle=True
        )
        results = model_selection.cross_val_score(
            model, X, Y,
            cv=k_fold_validation,
            scoring=make_scorer(classification_report_with_accuracy_score),
        )
        outcome.append(results)
        model_names.append(model_name)
        output_message = "%s| Mean=%f STD=%f" % (model_name, results.mean(), results.std())
        print(output_message)
    fig = plt.figure()
    fig.suptitle('Machine Learning Model Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(outcome)
    plt.ylabel('Accuracy [%]')
    ax.set_xticklabels(model_names)
    plt.savefig('myimage.png', format='png', dpi=1000)
    plt.show()
