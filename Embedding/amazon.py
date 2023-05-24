from tqdm import tqdm
import pickle
import os
import pandas as pd
from sentence_transformers import SentenceTransformer


def get_embedding(model, x):
    return model.encode(x)


def create_embedding(DATA_SIZE, name, BATCH_SIZE=1000):
    # Load the DataFrame
    with open('../../MyExperiments/datasets/amazon/book/split/hp_data.pkl', 'rb') as file:
        grp_df = pickle.load(file)
        if DATA_SIZE is not None:
            grp_df = grp_df.head(DATA_SIZE)

    # Initialize the SBERT model outside the function
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Process the records in batches
    batch_size = BATCH_SIZE
    total_records = len(grp_df)
    total_batches = (total_records - 1) // batch_size + 1

    # Create a directory to store the batch results
    os.makedirs('batch_results', exist_ok=True)

    # Iterate over the DataFrame rows
    for i in tqdm(range(total_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_records)
        batch = grp_df.iloc[start_idx:end_idx]

        # Apply the embedding function to the 'review' column
        batch['review'] = batch['review'].apply(lambda x: get_embedding(model, x))

        # Save the batch results to a separate file
        batch_filename = f'../../MyExperiments/datasets/amazon/book/batch_results/{name}_batch_{i}.pkl'
        with open(batch_filename, 'wb') as batch_file:
            pickle.dump(batch, batch_file)

        # Update the processed records count and estimated time
        processed_records = min((i + 1) * batch_size, total_records)
        remaining_records = total_records - processed_records
        tqdm.write(
            f"Processed {processed_records}/{total_records} records. Estimated time remaining: {remaining_records / batch_size:.2f} batches")

    # Combine the batch results into one DataFrame
    combined_df = pd.DataFrame()

    # Load and combine the batch files
    for i in tqdm(range(total_batches)):
        batch_filename = f'../../MyExperiments/datasets/amazon/book/batch_results/{name}_batch_{i}.pkl'
        with open(batch_filename, 'rb') as batch_file:
            batch = pickle.load(batch_file)
        combined_df = pd.concat([combined_df, batch], ignore_index=True)

    # Save the combined DataFrame
    file_name = f'../../MyExperiments/datasets/amazon/book/{name}_integratedReview_combined.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(combined_df, file)

    # Clean up the batch results directory
    # for i in range(total_batches):
    #     batch_filename = f'../../MyExperiments/datasets/amazon/book/batch_results/batch_{i}.pkl'
    #     os.remove(batch_filename)
    # os.rmdir('batch_results')


create_embedding(None, 'hp', 5000)
