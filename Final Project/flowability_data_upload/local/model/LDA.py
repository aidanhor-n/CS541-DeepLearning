from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class LDA:
    """
    LDA
    take preprocessed microtrac features and dependent y flowability values
    and use scikit-learn LinearDiscriminantAnaliyis (LDA) to
    create a predictive model that can predict the flowability given the features
    or transform the features into their componenets

    """

    def __init__(self, x: list, y: list, solver='eigen', shrinkage='auto', name="LDA"):
        """

        :param x: microtrac features
        :param y: the flowability of the powder
        :param solver: parameters for specific LDA solution, "svd, lsqr, eigen"
        :param shrinkage: default auto. give a float between 0 and 1 (lsgr and eigen)
        :param name: # TODO

        initialize and train LDA model

        """

        self.x = x
        self.y = y
        self.model = self.train_lda(solver, shrinkage)
        self.name = name

    def train_lda(self, solver='eigen', shrinkage='auto'):
        """

        :param solver: parameters for specific LDA solution, "svd, lsqr, eigen"
        :param shrinkage: default auto. give a float between 0 and 1 (lsgr and eigen)

        :return: trained lda model

        train the LDA model on class X and Y values given LDA parameters

        """

        lda = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage, n_components=2)

        trained_model = lda.fit(self.x, self.y)
        return trained_model

    def get_lda_x_transform(self, x: list = None) -> list:
        """

        :param x: a list of microtrac features

        :return: a list of transformed microtrac features

        use fitted LDA model to transform x data features into components

        """

        if x is None:
            x = self.x
        transformed = self.model.transform(x)
        return transformed

    def get_lda_y_pred(self, x: list = None) -> list:
        """

        :param x: # TODO

        :return: # TODO

        use fitted LDA model to predict y flowability value based on the x values

        """

        if x is None:
            x = self.x
        y_predictions = self.model.predict(x)
        return y_predictions
