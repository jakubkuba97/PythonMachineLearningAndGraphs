
"""
    Linear regression with 2d and 3d visualization
"""

import pandas as pd
from matplotlib import pyplot as plt


class NumberTimesTwo:
    def __init__(self) -> None:
        from random import randint
        self.learn_quantity = 20
        self.test_quantity = 5
        self.max_number = 100
        self.column_names = ['Number1', 'Result']
        self.learn_frame = pd.DataFrame(columns=self.column_names)
        self.test_frame = pd.DataFrame(columns=self.column_names)

        for _ in range(self.learn_quantity):
            x = randint(1, self.max_number)
            self.learn_frame = self.learn_frame.append({self.column_names[0]: x, self.column_names[1]: x * 2}, ignore_index=True)
        for _ in range(self.test_quantity):
            x = randint(1, self.max_number)
            self.test_frame = self.test_frame.append({self.column_names[0]: x, self.column_names[1]: x * 2}, ignore_index=True)

    def visualize(self, learn: bool = True) -> plt.subplot:
        if learn:
            ax = plt.scatter(self.learn_frame[self.column_names[0]], self.learn_frame[self.column_names[1]])
        else:
            ax = plt.scatter(self.test_frame[self.column_names[0]], self.test_frame[self.column_names[1]])
        plt.show()
        return ax


class NumberPlusNumber:
    def __init__(self) -> None:
        from random import randint
        self.learn_quantity = 1000
        self.test_quantity = 5
        self.max_number = 100
        self.column_names = ['Number1', 'Number2', 'Result']
        self.learn_frame = pd.DataFrame(columns=self.column_names)
        self.test_frame = pd.DataFrame(columns=self.column_names)

        for _ in range(self.learn_quantity):
            x = randint(1, self.max_number)
            y = randint(1, self.max_number)
            self.learn_frame = self.learn_frame.append({self.column_names[0]: x, self.column_names[1]: y, self.column_names[2]: x + y}, ignore_index=True)
        for _ in range(self.test_quantity):
            x = randint(1, self.max_number)
            y = randint(1, self.max_number)
            self.test_frame = self.test_frame.append({self.column_names[0]: x, self.column_names[1]: y, self.column_names[2]: x + y}, ignore_index=True)

    def visualize(self, learn: bool = True) -> plt.figure():
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
        return ax


class LinearLearningModel:
    def __init__(self, learning_frame: pd.DataFrame, testing_frame: pd.DataFrame, border_min_value: int = 0, border_max_value: int = 100) -> None:
        from sklearn.linear_model import LinearRegression

        learn_value = learning_frame.iloc[:, :len(learning_frame.columns) - 1].values
        learn_result = learning_frame.iloc[:, len(learning_frame.columns) - 1].values
        self.lm = LinearRegression()
        self.lm.fit(learn_value, learn_result)
        self.absolute_min_value = border_min_value

        # This is so graph shows all range, not just training range
        border_mini_value = [border_min_value for _ in range(len(learning_frame.columns) - 1)]
        border_maxi_value = [border_max_value for _ in range(len(learning_frame.columns) - 1)]
        border_mini_value.append(self.lm.predict([border_mini_value]))
        border_maxi_value.append(self.lm.predict([border_maxi_value]))
        if len(learning_frame.columns) == 3:            # 3 dimensional
            border_right_value = [border_min_value, border_max_value]
            border_left_value = [border_max_value, border_min_value]
            border_right_value.append(self.lm.predict([border_right_value]))
            border_left_value.append(self.lm.predict([border_left_value]))
            testing_frame.loc[len(testing_frame)] = [x for x in border_right_value]
            testing_frame.loc[len(testing_frame)] = [x for x in border_left_value]
        testing_frame.loc[len(testing_frame)] = [x for x in border_mini_value]
        testing_frame.loc[len(testing_frame)] = [x for x in border_maxi_value]

        self.test_value = testing_frame.iloc[:, :len(learning_frame.columns) - 1].values
        self.test_result = self.lm.predict(self.test_value)

    def draw_graph(self, compare_to_dataframe: pd.DataFrame = None, with_line: bool = False) -> plt.subplot:
        if len(self.test_value[0, :]) == 1:         # 2 dimensional
            ax = plt.plot(self.test_value, self.test_result, color='orange')
            if compare_to_dataframe is not None:
                ax = plt.scatter(compare_to_dataframe.iloc[:, 0], compare_to_dataframe.iloc[:, 1])
            plt.show()
            return ax
        elif len(self.test_value[0, :]) == 2:       # 3 dimensional
            from mpl_toolkits.mplot3d import Axes3D
            import numpy as np

            test_dataframe = pd.DataFrame(columns=['One', 'Two', 'Result'])
            for index, value in enumerate(self.test_value):
                cut_value = str(value).replace('[', '').replace(']', '').split()
                test_dataframe = test_dataframe.append({'One': cut_value[0], 'Two': cut_value[1], 'Result': self.test_result[index]}, ignore_index=True)

            ax = plt.figure()       # TODO: fix the additional blank graph
            ax = Axes3D(ax)
            ax.scatter(
                np.asarray(test_dataframe['One'], dtype='float'),
                np.asarray(test_dataframe['Two'], dtype='float'),
                np.asarray(test_dataframe['Result'], dtype='float'),
                c='orange')
            ax.set_xlabel('One')
            ax.set_ylabel('Two')
            ax.set_zlabel('Result')

            if with_line:
                for _, values in test_dataframe.iterrows():
                    ax.plot(
                        np.asarray([self.absolute_min_value, values['One']], dtype='float'),
                        np.asarray([self.absolute_min_value, values['Two']], dtype='float'),
                        np.asarray([self.absolute_min_value, values['Result']], dtype='float'),
                        color='orange'
                    )

            if compare_to_dataframe is not None:
                column_names = compare_to_dataframe.columns
                ax.scatter(
                    np.asarray(compare_to_dataframe[column_names[0]], dtype='float'),
                    np.asarray(compare_to_dataframe[column_names[1]], dtype='float'),
                    np.asarray(compare_to_dataframe[column_names[2]], dtype='float'),
                    c='blue')
            plt.show()
            return ax
        else:
            print('\n\tOnly up to 3 dimensions allowed in this universe!\n')


if __name__ == '__main__':
    my_base = NumberPlusNumber()

    model = LinearLearningModel(learning_frame=my_base.learn_frame, testing_frame=my_base.test_frame, border_max_value=my_base.max_number)
    model.draw_graph(compare_to_dataframe=my_base.learn_frame, with_line=True)
    print()
