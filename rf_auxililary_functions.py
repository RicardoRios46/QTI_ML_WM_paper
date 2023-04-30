from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import sklearn.metrics as metrics

def gridSearchRF(XA_train, yA_train, seed, n_jobs):
    # Build classificer model
    rf = RandomForestClassifier(random_state=seed)

    # Create the parameter grid based on the results of random search 
    rf_param_grid = {
        'bootstrap': [True],
        'max_depth': [4, 5, 6, 7, 8],
        'min_samples_leaf': [2, 3, 4, 5],
        'min_samples_split': [3, 4, 5, 6, 7],
        'n_estimators': [50, 100, 200, 300]
    }

    # Instantiate the grid search model
    rf_grid_search = GridSearchCV(estimator = rf, param_grid = rf_param_grid,  cv = 4, n_jobs = n_jobs)

    # Fit the random search model
    rf_grid_search.fit(XA_train, yA_train)

    return rf_grid_search


def get_metrics(model, X_test, y_test, average_value):
    
    y_pred = model.predict(X_test)
    
    #metrics.ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    #plt.show()
    print('Accuracy: %.3f' % metrics.accuracy_score(y_test, y_pred))
    print('Precision: %.3f' % metrics.precision_score(y_test, y_pred, average=average_value))
    print(metrics.precision_score(y_test, y_pred, average=None))
    print('Recall: %.3f' % metrics.recall_score(y_test, y_pred, average=average_value))
    print(metrics.recall_score(y_test, y_pred, average=None))
    print('F1: %.3f' % metrics.f1_score(y_test, y_pred, average=average_value))
    print(metrics.f1_score(y_test, y_pred, average=None))
