import math

import pandas as pd
import numpy as np
from BaseFunctionsContinuous import *


class DiscretizeContinuousAttributes:

    def __init__(self, data_path):
        self.name = "dis"
        self.data = pd.read_csv(data_path)
        self.data = self.data.sample(frac=1, ignore_index=True)
        self.discrete_data = self.data.copy();

    def discretize_data(self, output):
        self.split_dataset();
        for attr in self.data.columns:
            if attr == output:
                continue
            cut_points = self.discretize_attribute(attr, output)
            self.update_discrete_array(cut_points, attr)
        print(self.discrete_data)


    def update_discrete_array(self, cut_points, attribute):
        for i in range(self.data.shape[0]):
            self.discrete_data.at[i, attribute] = get_region_number(self.data.loc[i, attribute], cut_points)

    def discretize_attribute(self, attribute, output):
        cut_points = []
        regions_classes = [];
        regions_samples_count = []
        df = self.data.loc[:, [attribute, output]]
        df = df.sort_values(by=attribute, ignore_index=True)
        number_of_samples = len(df[attribute])

        last_class = df.loc[0, output];
        last_attr_value = df.loc[0, attribute];
        region_samples_count = 0
        region_classes = set()
        region_classes.add(last_class)

        for row in df.iterrows():
            if row[1][attribute] != last_attr_value and last_class != row[1][output]:
                potential_cut_point = (last_attr_value + row[1][attribute])/2
                gain = gain_T(df, potential_cut_point, attribute, output)
                delta = delta_T(df, potential_cut_point, attribute, output)
                if gain > (math.log2(number_of_samples - 1) + delta) / number_of_samples:
                    regions_classes.append(region_classes)
                    region_classes = set()
                    regions_samples_count.append(region_samples_count)
                    region_samples_count = 0
                    cut_points.append(potential_cut_point)

                last_class = row[1][output]
                last_attr_value = row[1][attribute]
                region_classes.add(last_class)
            region_samples_count += 1

        return cut_points #[cut_points, regions_classes, regions_samples_count]

    def split_dataset(self):
        count = self.data.shape[0]
        n = count*0.7;
        train_data = self.data.loc[0:n, :]
        test_data = self.data.loc[n:count, :]

        #print(train_data);
        #print("*\n*\n*\n*\n*");
        #print(test_data);
        return train_data, test_data


