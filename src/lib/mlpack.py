"""mlpack wrapper
"""
import mlpack


class HoeffdingTree():
    def __init__(self, **kwargs):
        self.config = kwargs

    def fit(self, train_X, train_y):
        self.fit_output = mlpack.hoeffding_tree(
            training=train_X,
            labels=train_y,
            **self.config
        )

    def predict(self, test_X):
        self.predict_output = mlpack.hoeffding_tree(
            input_model=self.fit_output['output_model'],
            test=test_X,
            **self.config
        )
        return self.predict_output['predictions']
