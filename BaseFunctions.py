import math


def entropy(samples_df, output):
    summation = 0.0
    class_set = set(samples_df.loc[:, output])
    number_of_samples = samples_df.shape[0]
    value_counts = samples_df.loc[:, output].value_counts()
    for c in class_set:
        p = value_counts[c]/number_of_samples
        summation += p * math.log2(p)
    return -summation


def weighted_entropy(samples_df, attribute_column, output_column):
    summation = 0.0
    attribute_values_set = set(samples_df.loc[:, attribute_column])
    value_counts = samples_df.loc[:, attribute_column].value_counts()
    number_of_samples = len(samples_df[output_column])
    print('attribute_values_set:',attribute_values_set)
    for v in attribute_values_set:
        value_ratio = value_counts[v]/number_of_samples
        value_entropy = entropy(samples_df.loc[samples_df[attribute_column] == v], output_column)
        summation += value_ratio*value_entropy
    return summation


def gain(samples_df, attribute_column, output_column):

    return entropy(samples_df, output_column) - weighted_entropy(samples_df, attribute_column, output_column)

