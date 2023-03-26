import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np
import matplotlib.pyplot as plt

from DataCleaner import DataCleaner


def pipeline():
    # example with cross validation
    # Generate mock dataset

    labelFrame, features = DataCleaner().getAttributesAndFeatures()
    # print(features.shape)
    # print(labelFrame.shape)
    # print(labelFrame.head())

    X_train, X_test, y_train, y_test = train_test_split(features, labelFrame, test_size=0.2, random_state=42)
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    # print(X_train.shape)
    # print(X_test.shape)
    # print(y_train.shape)
    # print(y_test.shape)

    # Train random forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Predict on testing set
    y_pred = rf.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # plot the feature importance to file
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    featureslist = features.columns.values.tolist()
    forest_importances = pd.Series(importances, index=featureslist)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

    plt.savefig('featureImportancePlot.png')

if __name__ == '__main__':
    pipeline()
