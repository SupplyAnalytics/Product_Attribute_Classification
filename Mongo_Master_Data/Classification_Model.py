def model_run():

    import pandas as pd
    import numpy as np
    import json
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    print('Libraries loaded successfully')

    # Define the model
    clf = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, warm_start=True)
    scaler = StandardScaler()

    # Load the base data
    base_data = pd.read_csv('Variants_Base_Data.csv')
    print('Base Data Pulled')

    # Collect all unique classes across all data files before training
    overall_classes = set()

    for i in range(1, 50):
        file_path = f'D:/Vectors_Data_2/variants_image_vectors_{i}.json'
        with open(file_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df_final = pd.merge(df, base_data, on='variantId', how='left')
        overall_classes.update(df_final['category'].dropna().astype(str).unique())

    # Convert to a sorted numpy array to ensure consistent order
    all_classes = np.array(sorted(overall_classes))
    print('All unique classes collected:', all_classes)

    # Loop over the files and train the model incrementally
    for i in range(1, 50):
        file_path = f'D:/Vectors_Data_2/variants_image_vectors_{i}.json'
        with open(file_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        print(f'file {i} json data prepared')

        # Merge the base data with the file data
        df_final = pd.merge(df, base_data, on='variantId', how='left')
        df_final = df_final[['variantId', 'vector', 'category']]
        df_final['category'] = df_final['category'].str.strip()  # Ensure no trailing spaces
        df_final = df_final.dropna(subset=['category'])
        print(f'file {i} training data prepared')

        # Ensure the 'vector' column contains numpy arrays of consistent length
        fixed_length = 1280  # Set the desired fixed length

        def process_vector(x):
            x = np.array(x, dtype=np.float32)
            if len(x) < fixed_length:
                x = np.pad(x, (0, fixed_length - len(x)), 'constant')
            elif len(x) > fixed_length:
                x = x[:fixed_length]
            return x

        # Prepare the data in batches and train incrementally
        batch_size = 25000  # Adjust batch size according to your available memory
        num_samples = len(df_final)

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            
            # Get the current batch of data
            batch_df = df_final.iloc[start:end]
            
            # Process the vectors for the batch
            vectors = np.array([process_vector(x) for x in batch_df['vector']])
            
            # Standardize the features incrementally
            scaler.partial_fit(vectors)
            X_batch_scaled = scaler.transform(vectors)
            
            # Prepare the labels for the current batch, ensuring they are strings
            y_batch = batch_df['category'].dropna().astype(str).values
            
            # Print unique class labels in y_batch and check against all_classes
            unique_classes_in_batch = np.unique(y_batch)
            print(f"Unique classes in current batch (batch {start // batch_size + 1}):", unique_classes_in_batch)
            
            # Check if any class in y_batch is not in all_classes
            unexpected_classes = np.setdiff1d(unique_classes_in_batch, all_classes)
            if len(unexpected_classes) > 0:
                print("Warning: The following classes in y_batch are not in all_classes:", unexpected_classes)
                raise ValueError(f"Found unexpected classes in y_batch: {unexpected_classes}")
            
            # Ensure classes are aligned before calling partial_fit
            if start == 0:
                # For the first batch, pass all_classes to ensure the classifier knows all possible classes
                clf.partial_fit(X_batch_scaled, y_batch, classes=all_classes)
            else:
                clf.partial_fit(X_batch_scaled, y_batch)

            print(f"Batch {start // batch_size + 1} of file {i} processed and trained.")

        print(f"Training of file {i} completed!")

    print('#### Final Training Completed Successfully ####')

    import pickle

    with open('sgd_model_1.pkl','wb') as f:
        pickle.dump(clf,f)
    
    print('Trained Model Version 1.0 Downloaded Successfully')

model_run()

