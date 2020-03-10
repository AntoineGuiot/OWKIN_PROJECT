import numpy as np
import pandas as pd
from pysurvival.models.survival_forest import ConditionalSurvivalForestModel, ExtraSurvivalTreesModel, \
    RandomSurvivalForestModel
from pysurvival.models.semi_parametric import NonLinearCoxPHModel
from pysurvival.models.multi_task import LinearMultiTaskModel, NeuralMultiTaskModel
from metrics import cindex
from lifelines.fitters.piecewise_exponential_regression_fitter import PiecewiseExponentialRegressionFitter
from lifelines.fitters.aalen_additive_fitter import AalenAdditiveFitter
from lifelines.fitters.weibull_aft_fitter import WeibullAFTFitter
from lifelines.fitters.log_normal_aft_fitter import LogNormalAFTFitter
from lifelines.fitters.coxph_fitter import CoxPHFitter
from lifelines.utils import k_fold_cross_validation


class Model:

    def __init__(self, name):
        self.name = name

    # We define different models

    def build_aalenAdditive(self):
        self.model = AalenAdditiveFitter(coef_penalizer=0.01, smoothing_penalizer=1000)

    def build_piecewise_exponential_regression(self):
        self.model = PiecewiseExponentialRegressionFitter(breakpoints=[1, 400], penalizer=0.01)

    def build_weibullAFT(self):
        self.model = WeibullAFTFitter(penalizer=0.01)

    def build_logNormal(self):
        self.model = LogNormalAFTFitter(penalizer=0.01)

    def build_cox(self):
        self.model = CoxPHFitter(penalizer=0.01)

    def build_multitask(self):
        self.model = LinearMultiTaskModel()

    def build_random_forest(self, num_trees=500):
        self.model = RandomSurvivalForestModel(num_trees=num_trees)

    def build_extra_survival_trees(self, num_trees=500):
        self.model = ExtraSurvivalTreesModel(num_trees=num_trees)

    def build_forest(self, num_trees=500):
        self.model = ConditionalSurvivalForestModel(num_trees=num_trees)

    def train(self, X, Y):  # the fit method depend on the model type
        if ('semi_parametric' in str(type(self.model))) or ('multi_task' in str(type(self.model))):
            self.model.fit(X=X, T=Y['SurvivalTime'], E=Y['Event'], init_method='zeros', num_epochs=500)

        if 'survival_forest' in str(type(self.model)):
            self.model.fit(X=X, T=Y['SurvivalTime'], E=Y['Event'], max_features='all', max_depth=20,
                           sample_size_pct=0.33)

        if 'lifelines' in str(
                type(self.model)):  # else we want to fit a model from lifeline library using cross validation
            k_fold_cross_validation(self.model, pd.concat([X, Y], axis=1), 'SurvivalTime', event_col='Event', k=5)
            # self.model.fit(pd.concat([X, Y], axis=1), 'SurvivalTime', event_col='Event', show_progress=False)

    def predict_survival_function(self, X):
        return self.model.pred(X)

    def predict_expectation(self, X):
        # the expectation is different depending on the package we use
        # as the expectation does not exist in pysurvival ->  we will use predict_risk
        if 'pysurvival' in str(type(self.model)):
            return self.model.predict_risk(X)
        else:

            return self.model.predict_expectation(X)

    def c_index(self, X, Y):
        Y_prediction = self.predict_expectation(X)

        # Y_prediction = 2 * max(Y_prediction) - Y_prediction
        if 'pysurvival' in str(type(self.model)):
            Y_prediction = 10 * max(Y_prediction) - Y_prediction
            Y_prediction = pd.DataFrame(Y_prediction, index=Y.index,
                                        columns=['SurvivalTime'])
        else:
            Y_prediction = pd.DataFrame(Y_prediction.values, index=Y.index,
                                        columns=['SurvivalTime'])

        Y_prediction['Event'] = np.nan  # Y['Event']
        return cindex(Y, Y_prediction)

    def predict_and_format(self, X, filename):
        Y_prediction = self.predict_expectation(X)

        if 'pysurvival' in str(type(self.model)):
            Y_prediction = 10 * max(Y_prediction) - Y_prediction
            Y_prediction = pd.DataFrame(Y_prediction, index=X.index,
                                        columns=['SurvivalTime'])
        else:
            Y_prediction = pd.DataFrame(Y_prediction.values, index=X.index,
                                        columns=['SurvivalTime'])

        Y_prediction['Event'] = np.nan  # Y['Event']
        # Y_prediction.to_csv(filename)
        return Y_prediction

    def fit_and_score(self, X, Y):
        # for each feature : create a model, train it (only on the selected feature) and compute the c-index score
        scores = []
        for feature in X.columns.values:
            Xj = X[feature].values.reshape((len(X), 1))

            self.train(Xj, Y)
            scores.append(self.c_index(Xj, Y))

        scores = pd.Series(scores, index=X.columns).sort_values(ascending=False)
        return scores
