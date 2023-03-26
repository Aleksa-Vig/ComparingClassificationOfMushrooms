from datetime import datetime as dt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

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

CSV_COLUMNS = ["Model", "Total Compile Time",
               "Total Sample Size", "Compile Time Per Sample"]
CSV_COLUMNS.extend(TESTS_WITH_SAMPLE_NAMES)

CSV_FORMAT = {CSV_COLUMNS[i]: i for i in range(len(CSV_COLUMNS))}


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


def buildModel(features: pd.DataFrame, answers: pd.DataFrame, model):
    # from tutorial: https://machinelearningmastery.com/calculate-feature-importance-with-python/

    # fit the model
    model.fit(features, answers)

    return model


def pipeline():
    logger = Logger(f"Models-Scores-allModels-nobk-macro-concat-ECG.txt")
    csvWriter = CSVWriter(f"Models-Scores-allModels-nobk-macro-concat-ECG.csv", CSV_COLUMNS)

    answers, features = DataCleaner().getAttributesAndFeatures()
    # print(features.shape)
    # print(labelFrame.shape)
    # print(labelFrame.head())

    # 20 / 20 / 60
    X_train, X_test, Y_train, Y_test = train_test_split(features, answers, test_size=0.4, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=42)

    # X_train, X_test, y_train, y_test = train_test_split(features, answers, test_size=0.2, random_state=42)
    Y_train = np.ravel(Y_train)
    Y_test = np.ravel(Y_test)
    Y_val = np.ravel(Y_val)

    # print(X_train.shape)
    # print(X_test.shape)
    # print(y_train.shape)
    # print(y_test.shape)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # # Train random forest classifier
    # rf = RandomForestClassifier(n_estimators=100, random_state=42)
    # rf.fit(X_train, Y_train)
    #
    # # Predict on testing set
    # Y_pred = rf.predict(X_test)
    #
    # # Evaluate accuracy
    # accuracy = accuracy_score(Y_test, Y_pred)
    # print("Accuracy:", accuracy)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    classifiers = [rf]

    classifierNames = [
        "Random Forest",
    ]

    for i, classifier in enumerate(classifiers):
        model_name = 'model-' + \
                     f"{classifierNames[i]}-" + f"" + '.model'
        modelCompileTime = (dt.now() - dt.now())

        # model = ModelSaver[StackingClassifier]().readModel(model_name)
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

    createFeatureImportancePlot(rf, features)


if __name__ == '__main__':
    pipeline()
