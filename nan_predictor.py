from sklearn.impute import IterativeImputer
import pandas as pd


class NanPredictor:
    def __init__(self, training_data, testing_data, categorical_col):
        self.training_data = training_data
        self.testing_data = testing_data
        self.categorical_col = categorical_col

    def replace_nan_values(self):
        # we have almost 30 patients that have missing values or nan values in their information
        # as we don't want to delete these rows, we will replace these values using IterativeImputer from sklearn
        # it models each feature with missing values as a function of other features, and uses that estimate for imputation.
        # more info : https://scikit-learn.org/stable/modules/impute.html#iterative-imputer
        imp = IterativeImputer(max_iter=50, random_state=0)
        imp.fit(self.training_data)

        transformed_train = imp.transform(self.training_data)
        transformed_test = imp.transform(self.testing_data)

        index_train = self.training_data.index
        index_test = self.testing_data.index

        columns = self.training_data.columns

        self.training_data = pd.DataFrame(transformed_train, index=index_train,
                                          columns=columns)

        self.testing_data = pd.DataFrame(transformed_test, index=index_test,
                                         columns=columns)

        # this method will predict float values while we want int values for categorical data
        for cat in self.categorical_col:
            self.training_data[cat] = self.training_data[cat].round()
            self.testing_data[cat] = self.testing_data[cat].round()

        return self.training_data, self.testing_data
