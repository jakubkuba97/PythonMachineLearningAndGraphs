
"""
    Using K-Nearest Neighbor to predict evenness of multiplication of two random numbers
"""


import pandas as pd
import matplotlib.pyplot as plt


class ExampleData:
    def __init__(self) -> None:
        self.columns = ['Number1', 'Number2', 'Multiplication is even']
        self.learn = pd.DataFrame(columns=self.columns)
        self.test_values = pd.DataFrame(columns=[self.columns[0], self.columns[1]])
        self.test_results = []
        self.learn_quantity = 500
        self.test_quantity = 25
        self.max_number = 50
        self.create_frames()

    def create_frames(self) -> None:
        from random import randint as rd
        for _ in range(self.learn_quantity):
            one = rd(1, self.max_number)
            two = rd(1, self.max_number)
            self.learn = self.learn.append({self.columns[0]: one, self.columns[1]: two, self.columns[2]: (1 if ((one * two) % 2) == 0 else 0)},
                                           ignore_index=True)
        for _ in range(self.test_quantity):
            one = rd(1, self.max_number)
            two = rd(1, self.max_number)
            self.test_values = self.test_values.append({self.columns[0]: one, self.columns[1]: two}, ignore_index=True)

    def visualize(self, only_learn: bool = True) -> plt.subplot:
        even_rows = pd.DataFrame(columns=[self.columns[0], self.columns[1]])
        uneven_rows = pd.DataFrame(columns=[self.columns[0], self.columns[1]])
        for _, row in self.learn.iterrows():
            if row[self.columns[2]] == 1:
                even_rows = even_rows.append({self.columns[0]: row[self.columns[0]], self.columns[1]: row[self.columns[1]]}, ignore_index=True)
            else:
                uneven_rows = uneven_rows.append({self.columns[0]: row[self.columns[0]], self.columns[1]: row[self.columns[1]]}, ignore_index=True)
        ax = plt.scatter(x=even_rows[self.columns[0]], y=even_rows[self.columns[1]], c='green')
        ax = plt.scatter(x=uneven_rows[self.columns[0]], y=uneven_rows[self.columns[1]], c='red')
        plt.xlabel(self.columns[0])
        plt.ylabel(self.columns[1])
        plt.suptitle('Is the multiplication even')
        if not only_learn:
            even_predict = pd.DataFrame(columns=[self.columns[0], self.columns[1]])
            uneven_predict = pd.DataFrame(columns=[self.columns[0], self.columns[1]])
            for index, row in self.test_values.iterrows():
                if '1' in self.test_results[index]:
                    even_predict = even_predict.append({self.columns[0]: row[self.columns[0]], self.columns[1]: row[self.columns[1]]},
                                                       ignore_index=True)
                else:
                    uneven_predict = uneven_predict.append({self.columns[0]: row[self.columns[0]], self.columns[1]: row[self.columns[1]]},
                                                           ignore_index=True)
            ax = plt.scatter(x=even_predict[self.columns[0]], y=even_predict[self.columns[1]], facecolors='none', edgecolors='green')
            ax = plt.scatter(x=uneven_predict[self.columns[0]], y=uneven_predict[self.columns[1]], facecolors='none', edgecolors='red')
        plt.show()
        return ax

    def neighbor_model(self) -> None:
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=5)
        lrn = []
        tst = []
        for _, value in self.learn.iterrows():
            lrn.append([value[self.columns[0]], value[self.columns[1]]])
        for _, value in self.test_values.iterrows():
            tst.append([value[self.columns[0]], value[self.columns[1]]])

        knn.fit(lrn, list(self.learn[self.columns[2]]))
        self.test_results.append(knn.predict(tst))
        self.test_results = str(self.test_results[0]).replace('[', '').replace(']', '').split()


if __name__ == '__main__':
    my_data = ExampleData()

    my_data.neighbor_model()
    my_data.visualize(only_learn=False)
