'''
XGBOOST module

the module can be use for both classification and regression problem the appropriate objective method should be defined.
more information can be find here
https://github.com/dmlc/xgboost/blob/master/doc/parameter.md

User can define a customized objective function and evaluation function

optimization with grid search is easy first make a grid

make_grid(eta = [1, 2, 3], max_depth = [3, 4, 5])

then run the optimization
optimization(self, X, y, X_test, y_test, verbose_eval = False, num_boost_round = None, early_stopping = 2)

to see the result and select best parameters
scores()


get the importance features and plot them is another awesome part of this module :)

get_importance()
plot_importance()


'''
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

color = sns.color_palette()

class XGBoost(object):
    def __init__(self, num_boost_round=10, objective=None, feval=None, **kwargs):
        self.clf = None
        self.grid_params = None
        self.result = None
        self.grid_list = []
        self.feval = feval
        self.num_boost_round = num_boost_round
        self.params = kwargs
        if objective:
            print("Built a XGBoost with defiend objective")
            if feval:
                if type(feval).__name__ == 'str':
                    print("Use eval_metric: ", feval)
                    self.params.update({'objective' : objective, 'eval_metric' : feval, 'silent' : True})
                else:
                    print("Use defiend funcation for eval")
                    self.params.update({'objective' : objective, 'silent' : True})
            else:
                print("Use default eval_metric")
                self.params.update({'objective' : objective, 'silent' : True})
                
        else:
            print("Built a XGBoost with default objective reg:linear")
            self.params.update({'objective' : 'reg:linear', 'silent' : True})

    def evalerror(self, preds, dtrain):
        labels = dtrain.get_label()
        return 'error', self.feval(labels, preds)

    def fit(self, X, y, num_boost_round=None):
        num_boost_round = num_boost_round or self.num_boost_round
        dtrain = xgb.DMatrix(X, label=y)
        self.clf = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=num_boost_round)

    def fit_eval(self, X, y, X_test, y_test, verbose_eval = False, num_boost_round = None, early_stopping = 2):
        num_boost_round = num_boost_round or self.num_boost_round
        dtrain = xgb.DMatrix(X, label=y)
        dtest = xgb.DMatrix(X_test, label=y_test)
        watchlist = [(dtrain, 'train'), (dtest, 'eval')]
        
        if type(self.feval).__name__ == 'str' or self.feval == None:
            self.clf = xgb.train(params = self.params, dtrain = dtrain, num_boost_round = num_boost_round,
                             evals = watchlist, verbose_eval = verbose_eval,
                             early_stopping_rounds = early_stopping)
        else:
            self.clf = xgb.train(params = self.params, dtrain = dtrain, num_boost_round = num_boost_round,
                             evals = watchlist, feval = self.evalerror, verbose_eval = verbose_eval,
                             early_stopping_rounds = early_stopping)
        

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.clf.predict(dtest)

    def score(self, X, y):
        Y = self.predict(X)
        return self.feval(y, Y)

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
        print("Number of iteration for Optimization: ", len(self.grid_list))
        
    def optimization(self, X, y, X_test, y_test, verbose_eval = False, num_boost_round = None, early_stopping = 2):
        num_boost_round = num_boost_round or self.num_boost_round
        dtrain = xgb.DMatrix(X, label=y)
        dtest = xgb.DMatrix(X_test, label=y_test)
        watchlist = [(dtrain, 'train'), (dtest, 'eval')]

        col = list(self.grid_params.keys()) + ['iterations', 'score']
        self.result = pd.DataFrame(columns=col)

        for i in tqdm(range(len(self.grid_list))):
            temp = {}
            for j, key in enumerate(self.grid_params.keys()):
                temp[key] = self.grid_list[i][j]
#            print(temp)
            temp.update(self.params)
#            print(temp)

            if type(self.feval).__name__ == 'str' or self.feval == None:
                self.clf = xgb.train(params = temp, dtrain = dtrain, num_boost_round = num_boost_round,
                             evals = watchlist, verbose_eval = verbose_eval,
                             early_stopping_rounds = early_stopping)
            else:
                self.clf = xgb.train(params = temp, dtrain = dtrain, num_boost_round = num_boost_round,
                             evals = watchlist, feval = self.evalerror, verbose_eval = verbose_eval,
                             early_stopping_rounds = early_stopping)

            self.result = self.result.append(pd.DataFrame([self.grid_list[i] + [self.clf.best_iteration, self.clf.best_score]], columns=col))


    def scores(self, N=4):
        return self.result.sort_values("score")[:N]
