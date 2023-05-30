import pickle
from sklearn.model_selection import train_test_split

RANDOM_STATE = 40


def load_data(path, filename):
    with open(path + filename, 'rb') as f:
        return pickle.load(f)


# this function needs to find which user and which book has embedding. If there is not embedding for
# a user or a book we should ignore the interaction that contains that particular user or book
def filter_data(data, users_embeddings, items_embeddings, data_name):

    if data_name == 'GDR':
        # Filter out those users who do not have embedding
        data = data[data.user_id.isin(users_embeddings.keys())]
        # Filter out those book who do not have embedding
        data = data[data.book_id.isin(items_embeddings.keys())]

    if data_name == 'AMZ':
        data = data[data.user.isin(users_embeddings.keys())]
        data = data[data.item.isin(items_embeddings.keys())]

    return data


# The following method split dataset to Train, Validation, and Test sets. First, It create a test set from the whole
# dataset based on ts_size. Then, then remaining part of the dataset will be split in order to have tr_size of that
# remaining part as training set and the reset will be considered as validation set.
def split_data(data, data_name, tr_size=0.8, ts_size=0.1):
    min_rating = 1
    max_rating = 5

    # Shuffle dataframe
    data.sample(frac=1, replace=True, random_state=RANDOM_STATE)

    # Set (user_is, item_id) as 'x' and (rating) as 'y' or target
    if data_name == 'GDR':
        x = data[["user_id", "book_id"]]
        y = data['rating']

    if data_name == 'AMZ':
        x = data[["user", "item"]]
        y = data['rate']


    x_rem, x_test, y_rem, y_test = train_test_split(x, y, test_size=ts_size, stratify=y, random_state=RANDOM_STATE)

    x_train, x_val, y_train, y_val = train_test_split(x_rem, y_rem, train_size=tr_size, stratify=y_rem,
                                                      random_state=RANDOM_STATE)

    return x_train, y_train, x_val, y_val, x_test, y_test
