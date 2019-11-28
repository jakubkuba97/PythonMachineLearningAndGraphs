
"""
    Using K-Means Clustering to classify semi-random data
"""

import pandas as pd


class TheData:
    def __init__(self, graph_size: list = None, number_of_examples: int = 100, percentage_of_random: int = 8,
                 coordinates_one: list = None, coordinates_two: list = None, middle_distance: int = 9) -> None:
        if graph_size is None:
            graph_size = [100, 100]
        self.graph_size = graph_size
        self.number_of_examples = number_of_examples
        self.percentage_of_random = percentage_of_random
        if coordinates_one is None:
            coordinates_one = [(self.graph_size[0] / 50), (self.graph_size[1] / 50)]
        if coordinates_two is None:
            coordinates_two = [(self.graph_size[0] - (self.graph_size[0] / 50)), (self.graph_size[1] - (self.graph_size[1] / 50))]
        self.middle_distance = middle_distance
        self.coor_base_one = coordinates_one
        self.coor_base_two = coordinates_two
        self.columns = ['Coordinate 1', 'Coordinate 2', 'Classifier']
        self.all_data = pd.DataFrame(columns=self.columns)
        self.get_data()

    def get_data(self) -> None:
        from random import randint as rd
        for _ in range(self.number_of_examples):
            closer_x = closer_y = 0
            coor = [1, 1]
            while closer_x + closer_y < (self.graph_size[0] / self.middle_distance + self.graph_size[1] / self.middle_distance) and (
                    closer_x + closer_y > -(self.graph_size[0] / self.middle_distance + self.graph_size[1] / self.middle_distance)):
                coor = [rd(1, self.graph_size[0]), rd(1, self.graph_size[1])]
                closer_x = int(coor[0] - self.coor_base_one[0]) - (int((1 - (coor[0] / self.coor_base_two[0])) * self.graph_size[0]))
                closer_y = int(coor[1] - self.coor_base_one[1]) - (int((1 - (coor[1] / self.coor_base_two[1])) * self.graph_size[1]))
            belong = 1 if closer_x + closer_y > 0 else 0
            if rd(1, 100) <= self.percentage_of_random:
                belong = rd(0, 1)
            self.all_data = self.all_data.append({self.columns[0]: coor[0], self.columns[1]: coor[1], self.columns[2]: belong}, ignore_index=True)

    def visualise(self, cluster: bool = True) -> None:
        from matplotlib import pyplot as plt
        first_group = pd.DataFrame(columns=[self.columns[0], self.columns[1]])
        second_group = pd.DataFrame(columns=[self.columns[0], self.columns[1]])
        for _, row in self.all_data.iterrows():
            if row[self.columns[2]] == 1:
                first_group = first_group.append({self.columns[0]: row[self.columns[0]], self.columns[1]: row[self.columns[1]]}, ignore_index=True)
            else:
                second_group = second_group.append({self.columns[0]: row[self.columns[0]], self.columns[1]: row[self.columns[1]]}, ignore_index=True)
        plt.figure()
        plt.scatter(x=self.coor_base_one[0], y=self.coor_base_one[1], c='indigo')
        plt.scatter(x=self.coor_base_two[0], y=self.coor_base_two[1], c='indigo')
        plt.scatter(x=first_group[self.columns[0]], y=first_group[self.columns[1]], c='green')
        plt.scatter(x=second_group[self.columns[0]], y=second_group[self.columns[1]], c='orange')
        plt.suptitle(self.columns[2])
        plt.xlabel(self.columns[0])
        plt.ylabel(self.columns[1])
        if cluster:
            from sklearn.cluster import KMeans
            import numpy as np
            plt.figure()
            means = KMeans(n_clusters=2)
            means.fit(self.all_data)
            means.predict(self.all_data)
            labels = means.labels_
            plt.scatter(x=self.all_data[self.columns[0]], y=self.all_data[self.columns[1]], c=labels.astype(np.float), edgecolor="k")
            plt.suptitle(self.columns[2] + ' cluster')
            plt.xlabel(self.columns[0] + ' cluster')
            plt.ylabel(self.columns[1] + ' cluster')
        plt.show()


if __name__ == '__main__':
    my_data = TheData(graph_size=[400, 400], number_of_examples=200, percentage_of_random=20, middle_distance=11)
    my_data.visualise()
