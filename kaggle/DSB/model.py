import altair as alt
from bayes_opt import BayesianOptimization
import catboost as cat
from catboost import CatBoostRegressor
from category_encoders.ordinal import OrdinalEncoder
from collections import Counter
from collections import defaultdict
import datetime
import eli5
from functools import partial
import gc
from IPython.display import HTML
from itertools import product
from joblib import Parallel, delayed
import json
import lightgbm as lgb
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import networkx as nx
from numba import jit
import numpy as np
import os
import pandas as pd
from random import choice
import re
import seaborn as sns
import scipy as sp
from scipy import stats
import shap
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.svm import NuSVR, SVR
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit, RepeatedStratifiedKFold
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import linear_model
import tensorflow as tf
from time import time
from tqdm import tqdm_notebook as tqdm
from typing import List, Any
import warnings
import xgboost as xgb
from xgboost import plot_importance
from xgboost import XGBClassifier, XGBRegressor

np.random.seed(566)
sns.set(style='darkgrid')
warnings.filterwarnings("ignore")


@jit
def qwk(a1, a2):
    max_rat = 3
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)

    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o += (i - j) * (i - j)

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / e


def eval_qwk_lgb_regr(y_true, y_pred):
    y_pred[y_pred <= 1.12232214] = 0
    y_pred[np.where(np.logical_and(
        y_pred > 1.12232214, y_pred <= 1.73925866))] = 1
    y_pred[np.where(np.logical_and(
        y_pred > 1.73925866, y_pred <= 2.22506454))] = 2
    y_pred[y_pred > 2.22506454] = 3

    # y_pred = y_pred.reshape(len(np.unique(y_true)), -1).argmax(axis=0)

    return'cappa', qwk(y_true, y_pred), True


class Base_Model(object):

    def __init__(self, train_df, test_df, features, categoricals=[], n_splits=5, ps={}, target='accuracy_group', plot=True):
        self.train_df = train_df
        self.test_df = test_df
        self.features = features
        self.n_splits = n_splits
        self.categoricals = categoricals
        self.target = target
        self.cv = self.get_cv()
        self.params = self.set_params(ps)
        self.evals_result = {}
        self.evals_results = []
        self.models = []
        self.random_seed = True
        self.seed = 42
        self.plot = plot
        self.y_pred = self.fit()

    def train_model(self, train_set, val_set):
        raise NotImplementedError

    def get_cv(self):
        cv = StratifiedKFold(n_splits=self.n_splits,
                             shuffle=True, random_state=42)

        return cv.split(self.train_df, self.train_df[self.target])

    def set_params(self, ps={}):
        return None if ps == {} else ps

    def fit(self):
        oof_pred = np.zeros((len(self.train_df), ))
        y_pred = np.zeros((len(self.test_df), ))

        for fold, (train_idx, val_idx) in enumerate(self.cv):
            x_train, x_val = self.train_df[self.features].iloc[train_idx], self.train_df[self.features].iloc[val_idx]
            y_train, y_val = self.train_df[self.target][train_idx], self.train_df[self.target][val_idx]
            train_set, val_set = self.convert_dataset(
                x_train, y_train, x_val, y_val)
            x_val = self.convert_x(x_val)
            x_test = self.convert_x(self.test_df[self.features])

            model = self.train_model(train_set, val_set)
            self.models.append(model)
            self.evals_results.append(self.evals_result)
            oof_pred[val_idx] = model.predict(
                x_val).reshape(oof_pred[val_idx].shape)

            y_pred += model.predict(x_test).reshape(y_pred.shape) / \
                self.n_splits
        self.oof_pred = oof_pred
        self.score = np.sqrt(mean_squared_error(
            self.train_df[self.target], self.oof_pred))
        if self.plot:
            self.report_plot()
        return y_pred

    def convert_dataset(self, x_train, y_train, x_val, y_val):
        raise NotImplementedError

    def convert_x(self, X):
        return X

    def plot_feature_importance(self, model):
        pass

    def report_plot(self):
        pass


class Lgb_Model(Base_Model):

    def train_model(self, train_set, val_set):
        if self.random_seed:
            self.params['seed'] = self.seed
            self.seed += 1
        return lgb.train(self.params, train_set, valid_sets=[train_set, val_set], verbose_eval=100, evals_result=self.evals_result)

    def convert_dataset(self, x_train, y_train, x_val, y_val):
        train_set = lgb.Dataset(
            x_train, y_train, categorical_feature=self.categoricals)
        val_set = lgb.Dataset(
            x_val, y_val, categorical_feature=self.categoricals)
        return train_set, val_set

    def set_params(self, ps):
        params = {'n_estimators': 5000,
                  'boosting_type': 'gbdt',
                  'objective': 'regression',
                  'metric': 'rmse',
                  'subsample': 0.75,
                  'subsample_freq': 1,
                  'learning_rate': 0.01,
                  'feature_fraction': 0.9,
                  'max_depth': 15,
                  'lambda_l1': 1,
                  'lambda_l2': 1,
                  'early_stopping_rounds': 100
                  }

        if not ps == {}:
            for key in ps.keys():
                params[key] = ps[key]
        return params

    def report_plot(self):
        fig, ax = plt.subplots(figsize=(16, 12))
        plt.subplot(2, 2, 1)
        self.plot_feature_importance()
        plt.subplot(2, 2, 2)
        self.plot_metric()
        plt.subplot(2, 2, 3)
        plt.hist(self.train_df[self.target] - self.oof_pred)
        plt.title('Distribution of errors')
        plt.subplot(2, 2, 4)
        plt.hist(self.oof_pred)
        plt.title('Distribution of oof predictions')

    def get_feature_importance(self):
        n = len(self.models)
        feature_imp_df = pd.DataFrame()
        for i in range(n):
            tmp = pd.DataFrame(zip(self.models[i].feature_importance(
            ), self.features), columns=['Value', 'Feature'])
            tmp['n_models'] = i
            feature_imp_df = pd.concat([feature_imp_df, tmp])
            del tmp
        self.feature_importance = feature_imp_df
        return feature_imp_df

    def plot_feature_importance(self, n=20):
        imp_df = self.get_feature_importance().groupby(
            ['Feature'])[['Value']].mean().reset_index(False)
        imp_top20_df = imp_df.sort_values('Value', ascending=False).head(20)
        sns.barplot(data=imp_top20_df, x='Value', y='Feature', orient='h')
        plt.title('Feature importances')

    def plot_metric(self):

        full_evals_results = pd.DataFrame()
        for result in self.evals_results:
            evals_result = pd.DataFrame()
            for k in result.keys():
                evals_result[k] = result[k][self.params['metric']]
            evals_result = evals_result.reset_index().rename(
                columns={'index': 'iteration'})
            full_evals_results = full_evals_results.append(evals_result)

        full_evals_results = full_evals_results.melt(id_vars=['iteration']).rename(
            columns={'value': self.params['metric'], 'variable': 'dataset'})
        sns.lineplot(data=full_evals_results, x='iteration',
                     y=self.params['metric'], hue='dataset')
        plt.title('Training progress')


class Xgb_Model(Base_Model):

    def train_model(self, train_set, val_set):
        return xgb.train(self.params, train_set,
                         num_boost_round=5000, evals=[(train_set, 'train'), (val_set, 'val')],
                         verbose_eval=100, early_stopping_rounds=100)

    def convert_dataset(self, x_train, y_train, x_val, y_val):
        train_set = xgb.DMatrix(x_train, y_train)
        val_set = xgb.DMatrix(x_val, y_val)
        return train_set, val_set

    def convert_x(self, x):
        return xgb.DMatrix(x)

    def set_params(self, ps):
        params = {'colsample_bytree': 0.8,
                  'learning_rate': 0.01,
                  'max_depth': 10,
                  'subsample': 1,
                  'objective': 'reg:squarederror',
                  # 'eval_metric':'rmse',
                  'min_child_weight': 3,
                  'gamma': 0.25,
                  'n_estimators': 5000}
        if not ps == {}:
            for key in ps.keys():
                params[key] = ps[key]

        return params


class Catb_regr_Model(Base_Model):

    def train_model(self, train_set, val_set):
        clf = CatBoostRegressor(**self.params)
        clf.fit(train_set['X'],
                train_set['y'],
                eval_set=(val_set['X'], val_set['y']),
                verbose=100,
                cat_features=self.categoricals)
        return clf

    def convert_dataset(self, x_train, y_train, x_val, y_val):
        train_set = {'X': x_train, 'y': y_train}
        val_set = {'X': x_val, 'y': y_val}
        return train_set, val_set

    def set_params(self, ps):
        params = {'loss_function': 'RMSE',
                  'task_type': "CPU",
                  'iterations': 5000,
                  'od_type': "Iter",
                  'depth': 10,
                  'colsample_bylevel': 0.5,
                  'early_stopping_rounds': 300,
                  'l2_leaf_reg': 18,
                  'random_seed': 42,
                  'use_best_model': True
                  }
        if not ps == {}:
            for key in ps.keys():
                params[key] = ps[key]

        return params


class Wrapper_bayesOpt:

    def __init__(self, train, test, features, categoricals):
        self.train = train
        self.test = test
        self.features = features
        self.categoricals = categoricals

    def LGB_Beyes(self,
                  subsample_freq,
                  learning_rate,
                  feature_fraction,
                  max_depth,
                  lambda_l1,
                  lambda_l2,
                  num_leaves):
        params = {}
        params['num_leaves'] = int(num_leaves)
        params['subsample_freq'] = int(subsample_freq)
        params['learning_rate'] = learning_rate
        params['feature_fraction'] = feature_fraction
        params['lambda_l1'] = lambda_l1
        params['lambda_l2'] = lambda_l2
        params['max_depth'] = int(max_depth)
        lgb_model = Lgb_Model(self.train, self.test,
                              self.features, categoricals=self.categoricals, ps=params)
        print('score:', lgb_model.score)
        return lgb_model.score

    def fit(self, bounds_LGB):
        LGB_BO = BayesianOptimization(
            self.LGB_Beyes, bounds_LGB, random_state=1029)
        init_points = 16
        n_iter = 16
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            LGB_BO.maximize(init_points=init_points, n_iter=n_iter,
                            acq='ucb', xi=0.0, alpha=1e-6)
        return LGB_BO


class Null_importance_selecter:

    def __init__(self, train, test, features, categoricals, target='accuracy_group', num_runs=80):
        self.train = train
        self.test = test
        self.features = features
        self.categoricals = categoricals
        self.params = self.set_params()
        self.target = target
        self.num_runs = num_runs
        np.random.seed(123)
        self.fit()

    def fit(self):
        actual_model = self.train_model(shuffle=False)
        self.actual_imp_df = self.get_imp_df(actual_model)

        null_imp_df = pd.DataFrame()
        print('Build Null Importances distribution')
        for i in tqdm(range(self.num_runs)):
            null_model = self.train_model(shuffle=True)
            imp_df = self.get_imp_df(null_model)
            imp_df['run'] = i + 1
            null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
        self.null_imp_df = null_imp_df
        self.display_null_importance_hists()
        self.correlation_scores = self.get_correlation_scores()
        print('Start Simulation of feature selection')
        self.best_score_features = self.score_feature_selection()

    def train_model(self, shuffle=False):

        y = self.train[self.target].copy()
        if shuffle:
            y = self.train[self.target].copy().sample(frac=1.0)
        lgb_train = lgb.Dataset(
            self.train[self.features], y, free_raw_data=False, silent=True)
        model = lgb.train(params=self.params, train_set=lgb_train,
                          num_boost_round=200, categorical_feature=self.categoricals)
        return model

    def get_imp_df(self, lgb_model):
        imp_df = pd.DataFrame()
        imp_df["feature"] = list(self.features)
        imp_df["importance_gain"] = lgb_model.feature_importance(
            importance_type='gain')
        imp_df["importance_split"] = lgb_model.feature_importance(
            importance_type='split')
        return imp_df

    def display_null_importance_hists(self):
        top5_feature = self.actual_imp_df.sort_values(
            ['importance_gain', 'importance_split'], ascending=False)['feature'][:5].reset_index(drop=True)
        fig, axes = plt.subplots(5, 2, figsize=(14, 18), dpi=100)
        for i in range(5):
            self.plot_distributions(top5_feature[i], 'split', axes[i % 5, 0])
            self.plot_distributions(top5_feature[i], 'gain', axes[i % 5, 1])

    def plot_distributions(self, feature, type, ax):
        hist = ax.hist(self.null_imp_df.loc[self.null_imp_df['feature'] == feature, 'importance_{}'.format(
            type)].values, label='Null importances')
        ax.vlines(x=self.actual_imp_df.loc[self.actual_imp_df['feature'] == feature, 'importance_split'].mean(
        ), ymin=0, ymax=np.max(hist[0]), color='r', linewidth=10, label='Real Target')
        ax.legend()
        plt.xlabel(
            'Null Importance ({}) Distribution for {}'.format(type, feature))
        ax.set_title('{} Importance of {}'.format(
            type, feature), fontweight='bold')

    def get_correlation_scores(self):
        correlation_scores = []
        for _f in self.actual_imp_df['feature'].unique():
            f_null_imps = self.null_imp_df.loc[self.null_imp_df['feature']
                                               == _f, 'importance_gain'].values
            f_act_imps = self.actual_imp_df.loc[self.actual_imp_df['feature']
                                                == _f, 'importance_gain'].values
            gain_score = 100 * \
                (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
            f_null_imps = self.null_imp_df.loc[self.null_imp_df['feature']
                                               == _f, 'importance_split'].values
            f_act_imps = self.actual_imp_df.loc[self.actual_imp_df['feature']
                                                == _f, 'importance_split'].values
            split_score = 100 * \
                (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
            correlation_scores.append((_f, split_score, gain_score))
        self.plot_correlation_scores(correlation_scores)
        return correlation_scores

    def plot_correlation_scores(self, correlation_scores):
        corr_scores_df = pd.DataFrame(correlation_scores, columns=[
            'feature', 'split_score', 'gain_score'])

        fig = plt.figure(figsize=(16, 16))
        gs = gridspec.GridSpec(1, 2)
        # Plot Split importances
        ax = plt.subplot(gs[0, 0])
        sns.barplot(x='split_score', y='feature', data=corr_scores_df.sort_values(
            'split_score', ascending=False).iloc[0:70], ax=ax)
        ax.set_title('Feature scores wrt split importances',
                     fontweight='bold', fontsize=14)
        # Plot Gain importances
        ax = plt.subplot(gs[0, 1])
        sns.barplot(x='gain_score', y='feature', data=corr_scores_df.sort_values(
            'gain_score', ascending=False).iloc[0:70], ax=ax)
        ax.set_title('Feature scores wrt gain importances',
                     fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.suptitle("Features' split and gain scores",
                     fontweight='bold', fontsize=16)
        fig.subplots_adjust(top=0.93)

    def score_feature_selection(self):
        results = []
        for threshold in tqdm([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]):
            split_feats, split_cat_feats = self.feature_select_by_threshold(
                threshold, 'split')
            gain_feats, gain_cat_feats = self.feature_select_by_threshold(
                threshold, 'gain')

            split_results = self.test_cv(
                train_features=split_feats, train_categoricals=split_cat_feats)
            gain_results = self.test_cv(
                train_features=gain_feats, train_categoricals=gain_cat_feats)
            results.append([threshold, split_results, 'split'])
            results.append([threshold, gain_results, 'gain'])
        selection_results_df = pd.DataFrame(
            results, columns=['threshold', 'score', 'type'])
        self.selection_results = selection_results_df
        self.plot_selection_score()
        condition = selection_results_df.sort_values(
            'score', ascending=False).iloc[0][['threshold', 'type']]
        return self.feature_select_by_threshold(condition['threshold'], condition['type'])

    def feature_select_by_threshold(self, threshold, type):
        if type == 'split':
            feats = [_f for _f, _score,
                     _ in self.correlation_scores if _score >= threshold]
            cat_feats = [_f for _f in feats if _f in self.categoricals]
        else:
            feats = [_f for _f, _,
                     _score in self.correlation_scores if _score >= threshold]
            cat_feats = [_f for _f in feats if _f in self.categoricals]
        return feats, cat_feats

    def test_cv(self, train_features, train_categoricals):
        lgb_train = lgb.Dataset(
            self.train[train_features], self.train[self.target], free_raw_data=False, silent=True)
        hist = lgb.cv(
            params=self.set_params(rf=False),
            train_set=lgb_train,
            num_boost_round=2000,
            categorical_feature=train_categoricals,
            nfold=5,
            stratified=True,
            shuffle=True,
            early_stopping_rounds=50,
            verbose_eval=0,
            seed=17
        )
        return hist['rmse-mean'][-1]

    def plot_selection_score(self):
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=self.selection_results,
                     x='threshold', y='score', hue='type')
        plt.title('Feature selection progress')

    def set_params(self, rf=True):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'rf',
            'subsample': 0.75,
            'colsample_bytree': 0.7,
            'num_leaves': 100,
            'max_depth': 8,
            'seed': 42,
            'bagging_freq': 1,
            'n_jobs': 4}
        if not rf:
            params['boosting_type'] = 'gbdt'
        return params


class OptimizedKappaRounder():
    # TODO: seed averagingしたいかも

    def __init__(self, train_predict_value, train_target_value, label):
        self.train_predict_value = train_predict_value
        self.train_target_value = train_target_value
        self.label = label
        self.coef = self.fit()

    def _kappa_loss(self, coef, X, y):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) +
                       [np.inf], labels=self.label)
        return -cohen_kappa_score(y, preds, weights='quadratic')

    def fit(self):
        loss_partial = partial(
            self._kappa_loss, X=self.train_predict_value, y=self.train_target_value)
        initial_coef = [i + 0.5for i in self.label][:-1]
        return sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X):
        preds = pd.cut(X, [-np.inf] + list(np.sort(self.coef['x'])) +
                       [np.inf], labels=self.label)
        return preds


class Adversarial_validator:

    def __init__(self, train, test, features, categoricals):
        self.train = train
        self.test = test
        self.features = features
        self.categoricals = categoricals
        self.union_df = self.train_test_union(self.train, self.test)
        self.cv = self.get_cv()
        self.models = []
        self.oof_pred = self.fit()
        self.report_plot()

    def fit(self):
        oof_pred = np.zeros((len(self.union_df), ))

        for fold, (train_idx, val_idx) in enumerate(self.cv):
            x_train, x_predict = self.union_df[self.features].iloc[
                train_idx], self.union_df[self.features].iloc[val_idx]
            y_train = self.union_df['target'][train_idx]
            train_set = self.convert_dataset(x_train, y_train)
            model = self.train_model(train_set)
            self.models.append(model)

            oof_pred[val_idx] = model.predict(
                x_predict).reshape(oof_pred[val_idx].shape)
        self.union_df['prediction'] = oof_pred
        return oof_pred

    def train_test_union(self, train, test):
        train['target'] = 0
        test['target'] = 1
        return pd.concat([train, test], axis=0).reset_index(drop=True)

    def get_cv(self):
        cv = StratifiedKFold(n_splits=5,
                             shuffle=True, random_state=42)

        return cv.split(self.union_df, self.union_df['target'])

    def convert_dataset(self, X, y):
        return lgb.Dataset(X, label=y, categorical_feature=self.categoricals)

    def train_model(self, train_set):
        return lgb.train(self.get_params(), train_set)

    def get_params(self):
        param = {'num_leaves': 50,
                 'num_round': 100,
                 'min_data_in_leaf': 30,
                 'objective': 'binary',
                 'max_depth': 5,
                 'learning_rate': 0.2,
                 'min_child_samples': 20,
                 'boosting': 'gbdt',
                 'feature_fraction': 0.9,
                 'bagging_freq': 1,
                 'bagging_fraction': 0.9,
                 'bagging_seed': 44,
                 'verbose_eval': 50,
                 'metric': 'auc',
                 'verbosity': -1}
        return param

    def report_plot(self):
        fig, ax = plt.subplots(figsize=(16, 12))
        plt.subplot(2, 2, 1)
        self.plot_feature_importance()
        plt.subplot(2, 2, 2)
        self.plot_roc_curve()
        plt.subplot(2, 2, 3)
        plt.hist(self.union_df['target'] - self.oof_pred)
        plt.title('Distribution of errors')
        plt.subplot(2, 2, 4)
        plt.hist(np.random.choice(self.oof_pred, 1000, False))
        plt.title('Distribution of oof predictions')

    def get_feature_importance(self):
        n = len(self.models)
        feature_imp_df = pd.DataFrame()
        for i in range(n):
            tmp = pd.DataFrame(zip(self.models[i].feature_importance(
            ), self.features), columns=['Value', 'Feature'])
            tmp['n_models'] = i
            feature_imp_df = pd.concat([feature_imp_df, tmp])
            del tmp
        self.feature_importance = feature_imp_df
        return feature_imp_df

    def plot_feature_importance(self, n=20):
        imp_df = self.get_feature_importance().groupby(
            ['Feature'])[['Value']].mean().reset_index(False)
        imp_top_df = imp_df.sort_values('Value', ascending=False).head(n)
        sns.barplot(data=imp_top_df, x='Value', y='Feature', orient='h')
        plt.title('Feature importances')

    def plot_roc_curve(self):
        fpr, tpr, thresholds = metrics.roc_curve(
            self.union_df['target'], self.oof_pred)
        auc = metrics.auc(fpr, tpr)

        plt.plot(fpr, tpr, label='ROC curve (area = %.2f)' % auc)
        plt.legend()
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')