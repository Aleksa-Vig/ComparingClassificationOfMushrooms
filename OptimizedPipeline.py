from datetime import datetime as dt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from CSVWriter import CSVWriter
from DataCleaner import DataCleaner
from Logger import Logger

TESTS = [accuracy_score, precision_score, recall_score, f1_score]
TESTS_WITH_SAMPLE_NAMES = []
for test in TESTS:
    TESTS_WITH_SAMPLE_NAMES.append(f"train-{test.__name__}")
    TESTS_WITH_SAMPLE_NAMES.append(f"val-{test.__name__}")
    TESTS_WITH_SAMPLE_NAMES.append(f"test-{test.__name__}")

TESTS_WITH_SAMPLE_NAMES.append(f"train-time")
TESTS_WITH_SAMPLE_NAMES.append(f"val-time")
TESTS_WITH_SAMPLE_NAMES.append(f"test-time")

CSV_COLUMNS = ["Model", "Model Best Parameters", "Total Compile Time",
               "Total Sample Size", "Compile Time Per Sample"]
CSV_COLUMNS.extend(TESTS_WITH_SAMPLE_NAMES)

CSV_FORMAT = {CSV_COLUMNS[i]: i for i in range(len(CSV_COLUMNS))}

# create a string with the current date and time
now_str = str(dt.now())
now_str = now_str.replace(':', ';').replace('.', '-')
now_str = now_str.replace(' ', 'AtTime')
now_str = ''.join(now_str.split())
logger = Logger(f"Optimized-Models-Scores-allModels"+now_str+".txt")
csvWriter = CSVWriter(f"Optimized-Models-Scores-allModels"+now_str+".csv", CSV_COLUMNS)

def createFeatureImportancePlot(model, features):

    # plot the feature importance to file
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    featureslist = features.columns.values.tolist()
    forest_importances = pd.Series(importances, index=featureslist)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

    plt.savefig('featureImportancePlot.png')


def buildModel(features: pd.DataFrame, answers: pd.DataFrame, classifier, param_grid):
    # from tutorial: https://machinelearningmastery.com/calculate-feature-importance-with-python/

    scorers = {
        'precision_score': make_scorer(precision_score, average='macro'),
        'recall_score': make_scorer(recall_score, average='macro'),
        'accuracy_score': make_scorer(accuracy_score)
    }

    # fit the model
    optimizedModel, optimizedModelParameters = grid_search_wrapper(classifier, param_grid, scorers, features, answers)

    return optimizedModel, optimizedModelParameters

def grid_search_wrapper(classifier, param_grid, scorers, features: pd.DataFrame, answers: pd.DataFrame, refit_score='precision_score'):
    """
    fits a GridSearchCV classifier using refit_score for optimization
    prints classifier performance metrics
    from https://www.kaggle.com/code/kevinarvai/fine-tuning-a-classifier-in-scikit-learn/notebook
    """
    grid_search = GridSearchCV(classifier, param_grid, scoring=scorers, refit=refit_score, cv=5, return_train_score=True, n_jobs=-1)
    grid_search.fit(features, answers)

    y_pred = grid_search.predict(features)

    logger.log('Best params for {}'.format(refit_score))
    logger.log(f"Best params found!{grid_search.best_params_}")

    return grid_search, grid_search.best_params_

def pipeline():

    logger.log(f"Getting data from datacleaner...")
    answers, features = DataCleaner().getAttributesAndFeatures()
    logger.log(f"Got data!")

    # 20 / 20 / 60
    logger.log(f"Splitting data into train test val")
    X_train, X_test, Y_train, Y_test = train_test_split(features, answers, test_size=0.4, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=42)

    # X_train, X_test, y_train, y_test = train_test_split(features, answers, test_size=0.2, random_state=42)
    Y_train = np.ravel(Y_train)
    Y_test = np.ravel(Y_test)
    Y_val = np.ravel(Y_val)

    # DEFINE CLASSIFIERS HERE AND PUT THEM INSIDE
    rf = RandomForestClassifier()
    dtc = DecisionTreeClassifier()
    knn = KNeighborsClassifier()
    svm = SVC()
    mlp = MLPClassifier()

    classifiers = [rf, dtc, knn, svm, mlp]

    logger.log(
        f"Created classifiers for {classifiers}")

    classifierNames = [
        "Random Forest",
        "Decision Tree",
        "K-Nearest Neighbors",
        "Support Vector Machine",
        "Neural Network (MLP)"
    ]

    classifiersParamsToOptimize = [
        #RandomForestClassifier
        {
            'min_samples_split': [3, 5, 10],
            'n_estimators': [100, 300],
            'max_depth': [3, 5, 15],
            'max_features': [3, 5, 20]
        },
        #Decision Tree Params
        {
            "splitter": ["best", "random"],
            "max_depth": [5, 10, 15],
            "max_features": [5, 10, 20]
        },
        #KNN params
        {
            "n_neighbors": [5, 10, 15],
            "weights": ["uniform", "distance"]
        },
        # SVM params
        {
            "kernel": ["linear", "poly", "rbf"]
        },
        #Neural Network
        {
            "activation": ["identity", "logistic", "tanh", "relu"],
            "batch_size": [4, 6, 8],
            "learning_rate": ["constant", "invscaling", "adaptive"]
        }

    ]

    logger.log(
        f"Starting to build models..... {classifiers}")

    for i, classifier in enumerate(classifiers):
        model_name = 'model-' + \
                     f"{classifierNames[i]}-" + f"" + '.model'
        modelCompileTime = (dt.now() - dt.now())

        model = None

        logger.log(f"Building Model on: {classifierNames[i]}")

        startTime = dt.now()
        model, modelBestParameters = buildModel(X_train, Y_train, classifier, classifiersParamsToOptimize[i])
        modelCompileTime = (dt.now() - startTime)
        logger.log(
            f"Time elapsed: (hh:mm:ss:ms) {modelCompileTime}")

        logger.log(f"Saving Model as: {model_name}")
        startTime = dt.now()
        logger.log(
            f"Time elapsed: (hh:mm:ss:ms) {dt.now() - startTime}")

        row = [" "] * len(CSV_COLUMNS)
        row[CSV_FORMAT["Model"]] = classifierNames[i]
        row[CSV_FORMAT["Model Best Parameters"]] = modelBestParameters
        row[CSV_FORMAT["Total Compile Time"]] = modelCompileTime
        row[CSV_FORMAT["Total Sample Size"]] = len(X_train.index)
        row[CSV_FORMAT["Compile Time Per Sample"]
        ] = modelCompileTime.total_seconds() / len(X_train.index)

        logger.log(f"Possible tests:", metrics.SCORERS.keys())

        logger.log("Testing model on Train")
        startTime = dt.now()
        y_pred = model.predict(X_train)
        timeElapsed = dt.now() - startTime
        logger.log(f"Time elapsed: (hh:mm:ss:ms) {timeElapsed}")
        row[CSV_FORMAT[f"train-time"]] = timeElapsed.total_seconds() / \
                                         len(X_train.index)

        for test_type in TESTS:
            res = None
            if (test_type.__name__ == accuracy_score.__name__):
                res = test_type(Y_train, y_pred)
            else:
                res = test_type(Y_train, y_pred, average='macro')
            logger.log(f"train-{test_type.__name__}:", res)
            row[CSV_FORMAT[f"train-{test_type.__name__}"]] = res

        logger.log("Testing model on val")
        startTime = dt.now()
        y_pred = model.predict(X_val)
        timeElapsed = dt.now() - startTime
        logger.log(f"Time elapsed: (hh:mm:ss:ms) {timeElapsed}")
        row[CSV_FORMAT[f"val-time"]] = timeElapsed.total_seconds() / \
                                       len(X_val.index)
        for test_type in TESTS:
            res = None
            if (test_type.__name__ == accuracy_score.__name__):
                res = test_type(Y_val, y_pred)
            else:
                res = test_type(Y_val, y_pred, average='macro')
            logger.log(f"val-{test_type.__name__}:", res)
            row[CSV_FORMAT[f"val-{test_type.__name__}"]] = res

        logger.log("Testing model on test")
        startTime = dt.now()
        y_pred = model.predict(X_test)
        timeElapsed = dt.now() - startTime
        logger.log(f"Time elapsed: (hh:mm:ss:ms) {timeElapsed}")
        row[CSV_FORMAT[f"test-time"]] = timeElapsed.total_seconds() / \
                                        len(X_test.index)
        for test_type in TESTS:
            res = None
            if (test_type.__name__ == accuracy_score.__name__):
                res = test_type(Y_test, y_pred)
            else:
                res = test_type(Y_test, y_pred, average='macro')
            logger.log(f"test-{test_type.__name__}:", res)
            row[CSV_FORMAT[f"test-{test_type.__name__}"]] = res

        csvWriter.addRow(row)

        #plot only works for random forest
        #createFeatureImportancePlot(classifier, features)


if __name__ == '__main__':
    pipeline()
