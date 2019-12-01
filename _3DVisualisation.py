
"""
    3D visualisation
"""

import pandas as pd
from matplotlib import pyplot as plt


class PlusAndResult:
    def __init__(self) -> None:
        from random import randint
        learn_quantity = 300
        test_quantity = 5
        max_number = 100
        self.column_names = ['Number1', 'Number2', 'Result']
        self.learn_frame = pd.DataFrame(columns=self.column_names)
        self.test_frame = pd.DataFrame(columns=self.column_names)

        for _ in range(learn_quantity):
            x = randint(1, max_number)
            y = randint(1, max_number)
            self.learn_frame = self.learn_frame.append({self.column_names[0]: x, self.column_names[1]: y, self.column_names[2]: x + y}, ignore_index=True)
        for _ in range(test_quantity):
            x = randint(1, max_number)
            y = randint(1, max_number)
            self.test_frame = self.test_frame.append({self.column_names[0]: x, self.column_names[1]: y, self.column_names[2]: x + y}, ignore_index=True)

    def visualize(self, learn: bool = True) -> None:
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np

        ax = plt.figure()
        ax = Axes3D(ax)
        if learn:
            ax.scatter(
                np.asarray(self.learn_frame[self.column_names[0]], dtype='float'),
                np.asarray(self.learn_frame[self.column_names[1]], dtype='float'),
                np.asarray(self.learn_frame[self.column_names[2]], dtype='float'))
        else:
            ax.scatter(
                np.asarray(self.test_frame[self.column_names[0]], dtype='float'),
                np.asarray(self.test_frame[self.column_names[1]], dtype='float'),
                np.asarray(self.test_frame[self.column_names[2]], dtype='float'))
        plt.show()


if __name__ == '__main__':
    my_base = PlusAndResult()

    my_base.visualize()
