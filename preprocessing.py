import pickle
from sklearn.model_selection import train_test_split


def load_data(path, filename):
    with open(path + filename, 'rb') as f:
        return pickle.load(f)


# The following method split dataset to Train, Validation, and Test sets. First, It create a test set from the whole
# dataset based on ts_size. Then, then remaining part of the dataset will be split in order to have tr_size of that
# remaining part as training set and the reset will be consider as validation set.
def split_data(data, tr_size=0.8, ts_size=0.1):
    min_rating = 1
    max_rating = 5

    # Shuffle dataframe
    data.sample(frac=1, replace=True, random_state=40)

    # Set (user_is, item_id) as 'x' and (rating) as 'y' or target
    x = data[["user_id", "book_id"]]
    y = data['rating']

    x_rem, x_test, y_rem, y_test = train_test_split(x, y, test_size=ts_size)

    x_train, x_val, y_train, y_val = train_test_split(x_rem, y_rem, train_size=tr_size)

    return x_train, y_train, x_val, y_val, x_test, y_test


