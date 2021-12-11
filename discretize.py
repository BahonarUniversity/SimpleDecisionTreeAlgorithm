import math

import pandas as pd
import numpy as np
from BaseFunctionsContinuous import *


class DiscretizeContinuousAttributes:

    def __init__(self, data, output):
        self.name = "dis"
        self.data = data
        self.cut_points = []
        self.output = output

        r_size = self.data.shape[0]
        c_size = self.data.shape[1]
        item_count = r_size * c_size
        npa = np.array(['' for _ in range(item_count)]).reshape(r_size, c_size)
        self.discrete_data = pd.DataFrame(npa, columns=self.data.columns, index=self.data.index)
        for i in range(self.data.shape[0]):
            self.discrete_data.loc[i, self.output] = self.data.loc[i, self.output]

    def discretize_data(self):
        for attr in self.data.columns:
            if attr == self.output:
                continue
            self.__discretize_attribute(attr)
            self.__update_discrete_array(self.cut_points, attr)
            print("Discretize: ", attr)
        return self.discrete_data

    def __update_discrete_array(self, cut_points, attribute):
        if len(cut_points) == 0:
            print('cut_points:', cut_points)

        for i in range(self.data.shape[0]):
            self.discrete_data.at[i, attribute] = get_region_function(self.data.loc[i, attribute], cut_points)

    def __discretize_attribute(self, attribute):
        df = self.data.loc[:, [attribute, self.output]]
        df = df.sort_values(by=attribute, ignore_index=True)
        self.__calculate_cut_points(df, attribute)

    def __calculate_cut_points(self, dataframe, attribute):
        size = dataframe.shape[0]
        if size < 2 or len(set(dataframe[self.output])) < 2:
            return
        midpoint = int(size/2)

        upward = midpoint
        downward = midpoint
        last_class = dataframe.loc[midpoint, self.output]
        last_up_attr_value = dataframe.loc[midpoint, attribute]
        last_down_attr_value = dataframe.loc[midpoint, attribute]

        number_of_samples = dataframe.shape[0]

        while upward < size-1 and downward >= 0:
            upward += 1
            if last_class != dataframe.loc[upward, self.output]:
                if self.__calculate_one_direction(dataframe, attribute, last_up_attr_value, upward, number_of_samples):
                    return
                last_class = dataframe.loc[upward, self.output]
            if last_class != dataframe.loc[downward, self.output]:
                if self.__calculate_one_direction(dataframe, attribute, last_down_attr_value, downward, number_of_samples):
                    return
                last_class = dataframe.loc[downward, self.output]

            downward -= 1

            last_up_attr_value = dataframe.loc[upward, attribute];
            last_down_attr_value = dataframe.loc[downward, attribute];

    def __calculate_one_direction(self, dataframe, attribute, last_attr, indx, number_of_samples):
        potential_cut_point = (last_attr + dataframe.loc[indx, attribute]) / 2
        gain = gain_T(dataframe, potential_cut_point, attribute, self.output)
        delta = delta_T(dataframe, potential_cut_point, attribute, self.output)
        if gain > (math.log2(number_of_samples - 1) + delta) / number_of_samples:
            self.cut_points.append(potential_cut_point)
            upper_bound = dataframe.loc[dataframe[attribute] >= potential_cut_point].reset_index(drop=True)
            lower_bound = dataframe.loc[dataframe[attribute] < potential_cut_point].reset_index(drop=True)
            self.__calculate_cut_points(upper_bound, attribute)
            self.__calculate_cut_points(lower_bound, attribute)
            return True
        return False
