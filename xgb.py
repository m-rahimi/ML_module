import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
from sklearn.metrics import mean_absolute_error

class XGBoostReg(object):
    def __init__(self, num_boost_round=10, **kwargs):
        self.clf = None
        self.num_boost_round = num_boost_round
        self.params = kwargs
        self.params.update({'objective': 'reg:linear', 'silent' : True})

    def fit(self, X, y, num_boost_round=None):
        num_boost_round = num_boost_round or self.num_boost_round
        dtrain = xgb.DMatrix(X, label=y)
        self.clf = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=num_boost_round)

    def fit_eval(self, X, y, X_test, y_test, verbose_eval = False, num_boost_round = None, early_stopping = 2):
        num_boost_round = num_boost_round or self.num_boost_round
        dtrain = xgb.DMatrix(X, label=y)
        dtest = xgb.DMatrix(X_test, label=y_test)
        watchlist = [(dtrain, 'train'), (dtest, 'eval')]
        self.clf = xgb.train(params = self.params, dtrain = dtrain, num_boost_round = num_boost_round,
                             evals = watchlist, verbose_eval = verbose_eval, early_stopping_rounds = early_stopping)

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.clf.predict(dtest)

    def score(self, X, y):
        Y = self.predict(X)
        return mean_absolute_error(y, Y)

    def get_params(self, deep=True):
        return self.params

    def get_importance(self):
        return self.clf.get_fscore()

    def plot_importance(self, N = 10):
        importance = self.clf.get_fscore()

        import operator
        importance = sorted(importance.items(), key=operator.itemgetter(1))

        importance_df = pd.DataFrame(importance, columns=['feature', 'fscore'])
        importance_df['fscore'] = importance_df['fscore'] / importance_df['fscore'].sum()

        plt.figure()
        importance_df[-N:].plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 5))
        plt.title('XGBoost Feature Importance')
        plt.xlabel('relative importance')
        plt.ylabel('')
        plt.show()

    def set_params(self, **params):
        if 'num_boost_round' in params:
            self.num_boost_round = params.pop('num_boost_round')
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self

class XGBoostOpt(object):
    def __init__(self, num_boost_round=100, **kwargs):
        self.clf = None
        self.grid_params = None
        self.result = None
        self.grid_list = []
        self.num_boost_round = num_boost_round
        self.params = kwargs
        self.params.update({'objective': 'reg:linear', 'silent' : True})

    def grid(self, keys, comb, it=0):
        if len(keys) == it:
            return

        for x in self.grid_params[keys[it]]:
            comb.append(x)
            self.grid(keys, comb, it+1)
            if len(comb) == len(keys):
                self.grid_list.append(comb[:])
            comb.pop()

    def make_grid(self, **kwargs):
        self.grid_list = []
        self.grid_params = kwargs
        print(self.params)
        print(self.grid_params)
        self.grid(list(self.grid_params.keys()), [])
        print(self.grid_list)

    def optimization(self, X, y, X_test, y_test, verbose_eval = False, num_boost_round = None, early_stopping = 2):

        num_boost_round = num_boost_round or self.num_boost_round
        dtrain = xgb.DMatrix(X, label=y)
        dtest = xgb.DMatrix(X_test, label=y_test)
        watchlist = [(dtrain, 'train'), (dtest, 'eval')]

        col = list(self.grid_params.keys()) + ['iterations', 'score']
        self.result = pd.DataFrame(columns=col)

        for i in range(len(self.grid_list)):
            temp = {}
            for j, key in enumerate(self.grid_params.keys()):
                temp[key] = self.grid_list[i][j]
            print(temp)
            temp.update(self.params)
#            print(temp)

            self.clf = xgb.train(params = temp, dtrain = dtrain, num_boost_round = num_boost_round,
                             evals = watchlist, verbose_eval = verbose_eval, early_stopping_rounds = early_stopping)

            self.result = self.result.append(pd.DataFrame([self.grid_list[i] + [self.clf.best_iteration, self.clf.best_score]], columns=col))


    def scores(self):
        return self.result.sort_values("score")
