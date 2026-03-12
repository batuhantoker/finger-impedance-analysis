"""EMG feature-based gesture classifier using extracted pickle data.

Combines time-domain and frequency-domain features, evaluates multiple classifiers
(LogReg, SVM, LDA, KNN, MLP, RFC, etc.) via k-fold cross-validation, outputs
boxplot comparison.
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def classification_report_with_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification report and return accuracy score for use as a scorer."""
    classification_report(y_true, y_pred, output_dict=True)
    return accuracy_score(y_true, y_pred)


if __name__ == "__main__":
    with open("data_s1_1_2.pkl.pkl", "rb") as handle:
        features = pickle.load(handle)

    (
        rms_flex,
        mav_flex,
        iav_flex,
        var_flex,
        wl_flex,
        mf_flex,
        pf_flex,
        mp_flex,
        tp_flex,
        sm_flex,
        msf_flex,
        rms_ext,
        mav_ext,
        iav_ext,
        var_ext,
        wl_ext,
        mf_ext,
        pf_ext,
        mp_ext,
        tp_ext,
        sm_ext,
        msf_ext,
        movement_id,
        force,
    ) = list(
        map(
            features.get,
            [
                "rms_flex",
                "mav_flex",
                "iav_flex",
                "var_flex",
                "wl_flex",
                "mf_flex",
                "pf_flex",
                "mp_flex",
                "tp_flex",
                "sm_flex",
                "msf_flex",
                "rms_ext",
                "mav_ext",
                "iav_ext",
                "var_ext",
                "wl_ext",
                "mf_ext",
                "pf_ext",
                "mp_ext",
                "tp_ext",
                "sm_ext",
                "msf_ext",
                "movement_id",
                "force",
            ],
        )
    )
    movement_id = np.asarray(movement_id)
    valid_classes = np.r_[
        np.array([i for i, v in enumerate(movement_id) if v.is_integer()])
    ]
    print(valid_classes.shape)
    movement_id = movement_id[valid_classes]
    movement_id[movement_id == 58] = 60
    print(np.unique(movement_id))

    iav_features = np.hstack((iav_flex, iav_ext))[valid_classes, :]
    var_features = np.hstack((var_flex, var_ext))[valid_classes, :]
    rms_features = np.hstack((rms_flex, rms_ext))[valid_classes, :]
    wl_features = np.hstack((wl_flex, wl_ext))[valid_classes, :]
    pf_features = np.hstack((pf_flex, pf_ext))[valid_classes, :]
    mp_features = np.hstack((mp_flex, mp_ext))[valid_classes, :]
    tp_features = np.hstack((tp_flex, tp_ext))[valid_classes, :]
    sm_features = np.hstack((sm_flex, sm_ext))[valid_classes, :]
    msf_features = np.hstack((msf_flex, msf_ext))[valid_classes, :]

    tdf1 = np.hstack((rms_features, iav_features))
    tdf2 = np.hstack((var_features, wl_features))
    tdf3 = np.hstack((rms_features, tdf2))
    tdf4 = np.hstack((rms_features, wl_features))
    fd1 = np.hstack((pf_features, mp_features))
    fd2 = np.hstack((tp_features, sm_features))
    fd3 = np.hstack((fd2, mp_features))
    fd_features = np.hstack((fd1, fd2))
    td_features = np.hstack((tdf1, tdf2))
    tfd_features = np.hstack((fd_features, td_features))
    print(movement_id)
    X_train, X_test, y_train, y_test = train_test_split(
        rms_features, movement_id, test_size=0.25, random_state=42
    )
    feature_set = np.hstack((fd2, tdf4))
    sc = StandardScaler()
    feature_set = sc.fit_transform(feature_set)

    plt.rcParams["figure.figsize"] = (20, 10)
    Y = movement_id
    X = feature_set

    number_of_k_fold = 10
    random_seed = 42
    outcome = []
    model_names = []
    models = [
        ("LogReg", LogisticRegression()),
        ("SVM", SVC()),
        # ('DecTree', DecisionTreeClassifier()),
        # ('KNN', KNeighborsClassifier(n_neighbors=15)),
        # ('LinDisc', LinearDiscriminantAnalysis()),
        # ('GaussianNB', GaussianNB()),
        # ('MLPC', MLPClassifier(activation='relu', solver='adam', max_iter=500)),
        # ('RFC',RandomForestClassifier()),
        # ('ABC', AdaBoostClassifier())
    ]

    for model_name, model in models:
        k_fold_validation = model_selection.KFold(
            n_splits=number_of_k_fold, random_state=random_seed, shuffle=True
        )
        results = model_selection.cross_val_score(
            model,
            X,
            Y,
            cv=k_fold_validation,
            scoring=make_scorer(classification_report_with_accuracy_score),
        )
        outcome.append(results)
        model_names.append(model_name)
        output_message = "%s| Mean=%f STD=%f" % (model_name, results.mean(), results.std())
        print(output_message)

    fig = plt.figure()
    fig.suptitle("Machine Learning Model Comparison")
    ax = fig.add_subplot(111)
    plt.boxplot(outcome)
    plt.ylabel("Accuracy [%]")
    ax.set_xticklabels(model_names)
    plt.savefig("myimage.png", format="png")
    plt.show()
