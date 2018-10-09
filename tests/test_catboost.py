from catboost import CatBoostRegressor
import numpy as np

def test_catboost():
    # Initialize data
    cat_features = [0, 1, 2]
    train_data = [
        ["a", "b", 1, 4, 5, 6],
        ["a", "b", 4, 5, 6, 7],
        ["c", "d", 30, 40, 50, 60]
    ]
    test_data = [
        ["a", "b", 2, 4, 6, 8],
        ["a", "d", 1, 4, 50, 60]
    ]
    train_labels = [10, 20, 30]
    # Initialize CatBoostRegressor
    model = CatBoostRegressor(
        iterations=2,
        learning_rate=1,
        depth=2,
        random_seed=100
    )
    # Fit model
    model.fit(train_data, train_labels, cat_features)
    # Get predictions
    preds = model.predict(test_data)
    print(preds)
    assert np.array_equal(preds, np.array([9.6, 9.6]))
