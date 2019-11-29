
"""
    Using Decision Tree Classification to predict whether 2 types of people will like the beer.
    One likes very soft and smooth taste, other the opposite.
"""

import pandas as pd


class BeerData:
    def __init__(self, amount_of_data: int = 100, tolerance: int = 1, max_score: int = 2) -> None:
        self.title = 'Beers classification per two types of people'
        self.columns = ['Sweetness', 'Intensitivity of taste', 'Amount of alcohol', 'Frothiness', 'Acidicy', 'Flavorness',
                        'Beer index', 'Good for hardness', 'Good for softness']     # higher scores are always in favor of softness
        self.wagers = [1.2, 1.1, 1.0, 0.9, 1.4, 1.2]
        self.all_data = pd.DataFrame(columns=self.columns)
        self.amount_of_data = amount_of_data
        self.guess_data = [0, 0]
        self.tolerance = tolerance
        self.max_score = max_score
        self.get_data()

    def get_data(self) -> None:
        from random import randint as rd
        for i in range(self.amount_of_data):
            row = []
            for _ in range(len(self.columns[:-3])):
                row.append(rd(0, self.max_score))
            row.append(i)
            row.append(self.get_result(row[:-1])[0])
            row.append(self.get_result(row[:-2])[1])
            to_add = pd.DataFrame(columns=self.columns)
            to_add = to_add.append({}, ignore_index=True)
            for j in range(len(row)):
                to_add[self.columns[j]] = int(row[j])
            self.all_data = self.all_data.append(to_add, ignore_index=True)

    def get_result(self, list_of_attributes: list) -> [int, int]:
        total = 0
        for i, attribute in enumerate(list_of_attributes):
            total += (attribute - (self.max_score / 2)) * self.wagers[i]
        if total < -self.tolerance:
            result = [1, 0]
        elif self.tolerance < total:
            result = [0, 1]
        else:
            result = [1, 1]
        return result

    def visualise(self) -> None:
        import matplotlib.pyplot as plt
        from matplotlib_venn import venn2 as ven
        first = []
        second = []
        for index, row in self.all_data.iterrows():
            if row[self.columns[-1]] == 1:
                second.append(row[self.columns[-3]])
            if row[self.columns[-2]] == 1:
                first.append(row[self.columns[-3]])
        ven([set(first), set(second)], set_labels=(self.columns[-2], self.columns[-1]))
        plt.suptitle('%i random %s' % (self.amount_of_data, self.title.lower()))
        if self.guess_data[-1] != 0 or self.guess_data[-2] != 0:
            coor = [0, 0.12]
            if self.guess_data[-2] == 1 and self.guess_data[-1] == 0:
                coor[0] = -0.45
            elif self.guess_data[-2] == 0 and self.guess_data[-1] == 1:
                coor[0] = 0.45
            plt.plot(coor[0], coor[1], 'bo')
            plt.text((coor[0] - 0.1), (coor[1] + 0.03), 'Your beer')
        plt.show()

    def input_data(self) -> None:
        new_data = []
        for column in self.columns[:-3]:
            new = -1
            while 0 > new or new > self.max_score:
                try:
                    print('Rate the %s of the beer (0 to %i),' % (column, self.max_score))
                    print('where 0 means very hardened taste and %i means very soft one: ' % self.max_score)
                    new = int(input('>> '))
                except ValueError:
                    print('\tEnter a valid integer!')
            new_data.append(new)
        new_data.append(len(self.all_data))
        results = self.classify(new_data[:-1])
        new_data.append(results[0])
        new_data.append(results[1])
        self.guess_data = new_data

    def classify(self, list_of_attributes: list) -> [int, int]:
        from sklearn.tree import DecisionTreeClassifier
        dt = DecisionTreeClassifier()
        learning_data = self.all_data.iloc[:, :-3].values.tolist()
        learning_results = self.all_data.iloc[:, -2:].values.tolist()
        dt.fit(learning_data, learning_results)
        result = dt.predict([list_of_attributes])[0]
        return result


if __name__ == '__main__':
    my_data = BeerData()
    my_data.input_data()
    my_data.visualise()
