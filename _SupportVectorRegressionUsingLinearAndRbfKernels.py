
"""
    Predicting next values of a semi-random regression and drawing a graph based on the values
"""

import pandas as pd
from matplotlib import pyplot as plt


class SupportVectorMachineModel:
    def __init__(self) -> None:
        self.columns = ['Number', '(Number * 2) - (Number / 3 * 2)']
        self.learn = pd.DataFrame(columns=self.columns)
        self.test_values = pd.DataFrame(columns=[self.columns[0]])
        self.test_results = []
        self.learn_quantity = 1000
        self.test_quantity = 80
        self.max_random = 4
        self.skip_how_many = int(self.learn_quantity / 20)
        self.set_data()

    def set_data(self) -> None:
        from random import randint as rd
        for i in range(self.learn_quantity):
            if rd(1, self.skip_how_many if self.skip_how_many >= 1 else 1) == 1:
                plus = rd(0, 1)
                if plus == 1 or i < self.max_random:
                    x = i + 1 + rd(0, self.max_random)
                else:
                    x = i + 1 - rd(0, self.max_random)
                x = (x * self.learn_quantity) - (x * (x / 2))
                self.learn = self.learn.append({self.columns[0]: i, self.columns[1]: x}, ignore_index=True)
        for _ in range(self.test_quantity):
            i = rd(0, self.learn_quantity)
            self.test_values = self.test_values.append({self.columns[0]: i}, ignore_index=True)
        self.learn = self.learn.sort_values(by=self.columns[0])
        self.test_values = self.test_values.sort_values(by=self.columns[0])

    def svr_model(self, kernel: str = 'rbf') -> None:
        if kernel != 'linear' and kernel != 'rbf':
            input('Only linear and rbf functions allowed!')
        from sklearn.svm import SVR
        if self.learn_quantity >= 9500:
            calculation = 8
        elif self.learn_quantity >= 4500:
            calculation = 7
        elif self.learn_quantity >= 450:
            calculation = 6
        elif self.learn_quantity >= 180:
            calculation = 5
        elif self.learn_quantity >= 80:
            calculation = 4
        else:
            calculation = 1
        pw = (1 * (10 ** int(calculation))) if (1 * (10 ** int(calculation))) >= 1e3 else 1e3
        npw = (1 * (10 ** -int(calculation))) if (1 * (10 ** -int(calculation))) <= 1e-2 else 1e-2
        svr_model = SVR(kernel=kernel, C=pw, gamma=npw)
        learn_values = self.learn.iloc[:, :len(self.learn.columns) - 1].values
        learn_results = self.learn.iloc[:, len(self.learn.columns) - 1].values
        test_values = self.test_values.iloc[:, :len(self.learn.columns) - 1].values
        svr_model.fit(learn_values, learn_results)
        self.test_results = svr_model.predict(test_values)

    def visualise(self) -> None:
        plt.scatter(x=self.learn[self.columns[0]], y=self.learn[self.columns[1]], c='green')
        plt.plot(self.test_values[self.columns[0]], self.test_results, color='red')
        plt.show()


if __name__ == '__main__':
    my_data = SupportVectorMachineModel()

    my_data.svr_model(kernel='rbf')
    my_data.visualise()
