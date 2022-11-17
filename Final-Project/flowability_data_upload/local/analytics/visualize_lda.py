"""Visualize LDA

this module visualizes the performance of a trained LDA model, looking at its success at seperating the dataset into
clusters through created components as well as predicted flowability
"""
from flowability_data_upload.local.model.LDA import LDA
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np


class VisualizeLDA:

    def __init__(self, lda_model: LDA, x_test: list, y_test: list):
        self.lda_model = lda_model
        self.x_test = x_test
        self.y_test = y_test
        self.visualize_lda_results()

    def visualize_lda_results(self):
        """
        visualize components, counts # of components and reports prediction accuracy on test set
        """
        self.visualize_components()
        components = self.count_components()
        accuracy = self.get_accuracy()

    def visualize_components(self):
        """
        takes x and y data and visualizes lda transformed components on scatterplot with flow classes linked to colors
        """
        plt.figure()
        colors = ['navy', 'darkorange', 'turquoise']
        x_transform = self.lda_model.get_lda_x_transform(self.x_test)
        component_1 = x_transform[:, 0]
        component_2 = x_transform[:, 1]
        scatter = plt.scatter(component_1, component_2, alpha=.2, c=self.y_test, cmap=matplotlib.colors.ListedColormap(colors))
        plt.xlabel("LDA Component 1")
        plt.ylabel("LDA Component 2")
        plt.legend(handles=scatter.legend_elements()[0], labels=["<=15", "15<n<=30", ">30"])
        plt.title('LDA Component Visualization' + " (" + self.lda_model.name + ")")
        plt.show()

    def count_components(self):
        """
        count the number of feature components the x data was transformed into
        :return: number of components
        """
        count = 0

        for i in self.lda_model.get_lda_x_transform(self.x_test)[0]:
            count += 1
        print("Count", count)
        return count

    def get_accuracy(self):
        """
        get accuracy of model on test set
        :return: model accuracy
        """
        return -1
