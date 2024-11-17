def test_model():
    import pandas as pd
    import numpy as np
    import pickle
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler

    print("Libraries loaded successfully")

    # Load the validation data
    validation_data = pd.read_csv('validation_data_unscaled.csv')
    print("Validation data loaded successfully")

    # Extract features (X) and labels (y)
    X_validation = np.array([np.array(eval(v), dtype=np.float32) for v in validation_data['vector']])
    y_validation = validation_data['category'].values

    # Load the trained model
    model_file = 'sgd_model_unscaled.pkl'
    with open(model_file, 'rb') as f:
        clf = pickle.load(f)
    print("Model loaded successfully")

    # Standardize the validation features
    # scaler = StandardScaler()
    # X_validation_scaled = scaler.fit_transform(X_validation)

    # Predict using the model
    y_pred = clf.predict(X_validation)

    # Calculate overall metrics
    print("\n### Overall Metrics ###")
    print(f"Accuracy: {accuracy_score(y_validation, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_validation, y_pred))

    # Split results by category
    print("\n### Metrics Split by Category ###")
    report_dict = classification_report(y_validation, y_pred, output_dict=True)
    category_metrics = pd.DataFrame(report_dict).transpose()

    # Exclude 'accuracy' row from the split metrics
    category_metrics = category_metrics.drop(['accuracy'], errors='ignore')
    print(category_metrics)

    # Save category-wise metrics to CSV
    metrics_csv = 'category_metrics_unscaled.csv'
    category_metrics.to_csv(metrics_csv, index=True)
    print(f"Category-wise metrics saved to {metrics_csv}")

    # Calculate and display confusion matrix
    print("\n### Confusion Matrix ###")
    conf_matrix = confusion_matrix(y_validation, y_pred, labels=clf.classes_)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=clf.classes_, columns=clf.classes_)
    print(conf_matrix_df)

    # Save confusion matrix to CSV
    confusion_matrix_csv = 'confusion_matrix_unscaled.csv'
    conf_matrix_df.to_csv(confusion_matrix_csv)
    print(f"Confusion matrix saved to {confusion_matrix_csv}")

    print("\n#### Testing Completed Successfully ####")

test_model()