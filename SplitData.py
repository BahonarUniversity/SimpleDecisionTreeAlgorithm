

def train_test_data(data, train_percent):
    shuffled_data = data.sample(frac=1, ignore_index=True)

    count = shuffled_data.shape[0]
    n = count*train_percent/100;
    train_data = shuffled_data.loc[0:n, :]
    test_data = shuffled_data.loc[n:count, :]
    return train_data, test_data
