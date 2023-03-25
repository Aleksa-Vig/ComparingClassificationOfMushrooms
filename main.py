def pipeline():
    # example with cross validation

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    import numpy as np

    # Generate mock dataset
    X = np.random.rand(100, 10)  # 100 samples with 10 features each
    y = np.random.randint(2, size=100)  # Binary labels (0 or 1)

    # Train random forest classifier with cross-validation
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(rf, X, y, cv=5)  # 5-fold cross-validation

    # Print average accuracy and standard deviation across folds
    print("Accuracy:", np.mean(scores))
    print("Standard deviation:", np.std(scores))

if __name__ == '__main__':
    pipeline()
