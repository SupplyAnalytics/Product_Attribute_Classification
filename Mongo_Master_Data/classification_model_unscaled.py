def model_run():
    ''' This model is as similar to the previous one but with as slight change as we are not scaling the input and directly fitting it in the classifier
    '''
    import pandas as pd
    import numpy as np
    import json
    import pickle
    from sklearn.linear_model import SGDClassifier
    import os

    print('Libraries loaded successfully')

    # Define the model
    clf = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, warm_start=True)

    # Load the base data
    base_data = pd.read_csv('Variants_Base_Data.csv')
    print('Base Data Pulled')

    # Collect all unique classes across all data files before training
    overall_classes = set()

    # Validation data storage
    validation_data = []

    for i in range(1, 50):
        file_path = f'D:/Vectors_Data_2/variants_image_vectors_{i}.json'
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            df_final = pd.merge(df, base_data, on='variantId', how='left')
            overall_classes.update(df_final['category'].dropna().astype(str).unique())
        except Exception as e:
            print(f"Error while processing file {file_path}: {e}")
            continue

    # Convert to a sorted numpy array to ensure consistent order
    all_classes = np.array(sorted(overall_classes))
    print('All unique classes collected:', all_classes)

    # Loop over the files and train the model incrementally
    model_file = 'sgd_model_unscaled.pkl'  # Single model file
    for i in range(1, 50):
        file_path = f'D:/Vectors_Data_2/variants_image_vectors_{i}.json'
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            print(f'File {i} json data prepared')

            # Merge the base data with the file data
            df_final = pd.merge(df, base_data, on='variantId', how='left')
            df_final = df_final[['variantId', 'vector', 'category']]
            df_final['category'] = df_final['category'].str.strip()  # Ensure no trailing spaces
            df_final = df_final.dropna(subset=['category'])
            print(f'File {i} training data prepared')

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
                
                # Prepare the labels for the current batch
                y_batch = batch_df['category'].dropna().astype(str).values
                
                # Append every 5000th record to validation_data
                validation_data.extend(
                    [[x.tolist(), y] for x, y in zip(vectors[::5000], y_batch[::5000])]
                )

                # Check if any class in y_batch is not in all_classes
                unique_classes_in_batch = np.unique(y_batch)
                unexpected_classes = np.setdiff1d(unique_classes_in_batch, all_classes)

                if len(unexpected_classes) > 0:
                    print(f"Warning: Skipping batch due to unexpected classes: {unexpected_classes}")
                    continue
                
                # Train the classifier incrementally
                if start == 0:
                    # For the first batch, pass all_classes to ensure the classifier knows all possible classes
                    clf.partial_fit(vectors, y_batch, classes=all_classes)
                else:
                    clf.partial_fit(vectors, y_batch)

                print(f"Batch {start // batch_size + 1} of file {i} processed and trained.")

            # Save the model after processing the file (overwrite existing model file)
            with open(model_file, 'wb') as f:
                pickle.dump(clf, f)
            print(f"Model saved after processing file {i} to {model_file}")

        except Exception as e:
            print(f"Error while processing file {file_path}: {e}")
            continue

    # Save validation data to CSV
    validation_df = pd.DataFrame(validation_data, columns=['vector', 'category'])
    validation_csv_file = 'validation_data_unscaled.csv'
    validation_df.to_csv(validation_csv_file, index=False)
    print(f"Validation data saved to {validation_csv_file}")

    print('#### Final Training Completed Successfully ####')

model_run()
