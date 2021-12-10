import math

import pandas as pd
import numpy as np
from BaseFunctionsContinuous import *


class DiscretizeContinuousAttributes:

    def __init__(self, data, output):
        self.name = "dis"
        self.data = data

        r_size = self.data.shape[0]
        c_size = self.data.shape[1]
        item_count = r_size * c_size
        npa = np.array(['' for _ in range(item_count)]).reshape(r_size, c_size)
        self.discrete_data = pd.DataFrame(npa, columns=self.data.columns, index=self.data.index)
        for i in range(self.data.shape[0]):
            self.discrete_data.loc[i, output] = self.data.loc[i, output]

    def discretize_data(self, output):
        for attr in self.data.columns:
            if attr == output:
                continue
            cut_points = self.discretize_attribute(attr, output)
            self.update_discrete_array(cut_points, attr)
        return self.discrete_data

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



