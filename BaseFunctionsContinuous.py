import math
from BaseFunctions import entropy


def weighted_entropy_T(samples_df, cut_point, attribute_column, output_column):
    number_of_samples = len(samples_df[output_column])
    s1 = samples_df.loc[samples_df[attribute_column] < cut_point]
    s2 = samples_df.loc[samples_df[attribute_column] >= cut_point]
    s1_size = len(s1)
    s2_size = len(s2)

    return (s1_size*entropy(s1, output_column)+s2_size*entropy(s2, output_column))/number_of_samples


def gain_T(samples_df, cut_point, attribute, output):
    return entropy(samples_df, output) - weighted_entropy_T(samples_df, cut_point, attribute, output)


def delta_T(samples_df, cut_point, attribute, output):
    class_set = set(samples_df.loc[:, output])
    samples_1_df = samples_df.loc[samples_df[attribute] < cut_point]
    samples_2_df = samples_df.loc[samples_df[attribute] >= cut_point]

    k = len(class_set)
    k1 = len(set(samples_1_df))
    k2 = len(set(samples_2_df))
    ent1 = k1*entropy(samples_1_df, output)
    ent2 = k2*entropy(samples_2_df, output)
    entropies = k * entropy(samples_df, output) - ent1 - ent2
    return math.log2(3**k-2) - entropies


def error(test_samples, test_results):
    sum_of_square_error = 0;
    for i in range(len(test_samples)):
        sum_of_square_error += (test_samples[i]-test_results[i])**2
    #sum_of_square_error *= 1/m;
    return sum_of_square_error


def get_region_function(value, cut_points):
    if len(cut_points) == 0:
        print('cut_points is empty for value:', value)
        return
    upper_bound = max(cut_points)
    if value <= cut_points[0]:
        return 'lambda val: val <= '+str(round(cut_points[0], 3));
    if value > upper_bound:
        return 'lambda val: val > '+str(round(upper_bound, 3));

    lower_bound = cut_points[0]
    for i in range(1, len(cut_points)-1):
        if value <= cut_points[i]:
            upper_bound = cut_points[i]
            break
        lower_bound = cut_points[i]

    return 'lambda val: '+str(round(lower_bound, 3))+' < val <= '+str(round(upper_bound, 3));
