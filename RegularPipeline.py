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

CSV_COLUMNS = ["Model", "Model Parameters", "Total Compile Time",
               "Total Sample Size", "Compile Time Per Sample"]
CSV_COLUMNS.extend(TESTS_WITH_SAMPLE_NAMES)

CSV_FORMAT = {CSV_COLUMNS[i]: i for i in range(len(CSV_COLUMNS))}

# create a string with the current date and time
now_str = str(dt.now())
now_str = now_str.replace(':', '-').replace('.', '-')
now_str = now_str.replace(' ', 'AtTime')
now_str = ''.join(now_str.split())
logger = Logger(f"Reg-Models-Scores-allModels"+now_str+".txt")
csvWriter = CSVWriter(f"Reg-Models-Scores-allModels"+now_str+".csv", CSV_COLUMNS)

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


def buildModel(features: pd.DataFrame, answers: pd.DataFrame, classifier):
    # from tutorial: https://machinelearningmastery.com/calculate-feature-importance-with-python/

    classifier.fit(features, answers)

    return classifier

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
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    dtc = DecisionTreeClassifier(random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    svm = SVC(kernel='linear', random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

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
    logger.log(
        f"Starting to build models..... {classifiers}")

    for i, classifier in enumerate(classifiers):
        model_name = 'model-' + \
                     f"{classifierNames[i]}-" + f"" + '.model'
        modelCompileTime = (dt.now() - dt.now())

        model = None

        logger.log(f"Building Model on: {classifierNames[i]}")

        startTime = dt.now()
        model = buildModel(X_train, Y_train, classifier)
        modelCompileTime = (dt.now() - startTime)
        logger.log(
            f"Time elapsed: (hh:mm:ss:ms) {modelCompileTime}")

        logger.log(f"Saving Model as: {model_name}")
        startTime = dt.now()
        logger.log(
            f"Time elapsed: (hh:mm:ss:ms) {dt.now() - startTime}")

        row = [" "] * len(CSV_COLUMNS)
        row[CSV_FORMAT["Model"]] = classifierNames[i]
        row[CSV_FORMAT["Model Parameters"]] = model.get_params()
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
