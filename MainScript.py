
"""
    Main script that will show most possible actions
"""

import _3DVisualisation
import _LinearRegressionWith2d3d
import _KNearestNeighborPrediction
import _SupportVectorRegressionUsingLinearAndRbfKernels
import _KMeansClustering
import _DecisionTreeClassificationPrediction


class AllFunctions:
    def __init__(self) -> None:
        print('Starting...\n')

    @staticmethod
    def go_through_all() -> None:
        three_vis = _3DVisualisation.PlusAndResult()
        print('\nHi! I will guide you through the world of graphs and simple machine learning!')
        print('First, I will show you a simple 3-dimensional graph, showing the relation between two random numbers (0 - 100) and their sum.')
        while input("\tPress 'y' and 'enter' to continue: ") != 'y':
            continue
        three_vis.visualize()
        del three_vis

        linear_two_and_three = _LinearRegressionWith2d3d.NumberPlusNumber()
        print('\nNow I will input this data into linear regression model to teach it and make it guess a sum of two random numbers!')
        print('Imagine making paying taxes simple using model like this!')
        while input("\tPress 'y' and 'enter' to continue: ") != 'y':
            continue
        model = _LinearRegressionWith2d3d.LinearLearningModel(learning_frame=linear_two_and_three.learn_frame,
                                                              testing_frame=linear_two_and_three.test_frame,
                                                              border_max_value=linear_two_and_three.max_number)
        model.draw_graph(compare_to_dataframe=linear_two_and_three.learn_frame, with_line=True)
        del linear_two_and_three

        k_nearest_neighbor = _KNearestNeighborPrediction.ExampleData()
        print('\nNext visualisation will present two numbers on a matrix. If their multiplication is even - they will be green, otherwise - red.')
        print('The data will contain empty circles, that did not have their evenness calculated.')
        print('Instead - KNN model will guess their color, based on their 5 nearest neighbors.')
        while input("\tPress 'y' and 'enter' to continue: ") != 'y':
            continue
        k_nearest_neighbor.neighbor_model()
        k_nearest_neighbor.visualize(only_learn=False)
        del k_nearest_neighbor

        support_vector_regression = _SupportVectorRegressionUsingLinearAndRbfKernels.SupportVectorMachineModel()
        print("\nThat's amazing, but let's go back to regression.")
        print("This graph will represent a series of semi-random values following a clearly-seen pattern. Let's imagine values represent real-world examples!")
        print("I will use a support vector regression model to predict its pattern, to see future values.")
        while input("\tPress 'y' and 'enter' to continue: ") != 'y':
            continue
        support_vector_regression.svr_model(kernel='linear')
        support_vector_regression.visualise()

        print("\nLet me now use a non-linear model to calculate the pattern. Simple RBF should do!")
        while input("\tPress 'y' and 'enter' to continue: ") != 'y':
            continue
        support_vector_regression.svr_model(kernel='rbf')
        support_vector_regression.visualise()
        del support_vector_regression

        print('\nRemember! Every value in this program is random and every model predicts values without knowing their correct results!')
        while input("\tPress 'y' and 'enter' to continue: ") != 'y':
            continue

        k_means_cluster = _KMeansClustering.TheData(graph_size=[400, 400], number_of_examples=200, percentage_of_random=25, middle_distance=11)
        print('\nNow let us not give the model any hints whatsoever.')
        print("It will take the data collected about - imagine some customer groups - and cluster them without any guidance.")
        print('First graph will show the collected data with border values cut off, second one will show a k-means cluster model.')
        while input("\tPress 'y' and 'enter' to continue: ") != 'y':
            continue
        k_means_cluster.visualise()
        del k_means_cluster

        decision_tree = _DecisionTreeClassificationPrediction.BeerData(max_score=5)
        print("\nSomething with a great use real-life? How about some beer?")
        print("Imagine we will classify 100 random beer brands into three groups - ones for someone that likes very harsh taste, "
              "ones for someone who likes soft, and the ones for both  of them.")
        print('You will now enter data about a new beer, and the program will try to tell you where this one would belong, using a decision-tree classification.')
        while input("\tPress 'y' and 'enter' to continue: ") != 'y':
            continue
        print()
        decision_tree.input_data()
        decision_tree.visualise()
        del decision_tree

        print("\n\n\n\n\nThank you for using this program!")
        print("I hope you will have a great day!")
        input("\tPress any key to close...")


if __name__ == '__main__':
    all_fun = AllFunctions()
    all_fun.go_through_all()
