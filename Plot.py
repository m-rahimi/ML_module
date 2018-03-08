import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
color = sns.color_palette()

class Plot(object):
    def __init__(self, font=1.5, style="darkgrid", **kwargs):
        sns.set()
        sns.set(font_scale = font)
        sns.set_style(style)

    def plot_help(self):
        print("distro(self, x, nbins=100, kde=True, xlim=None, xlabel=None)")
        print("count(self, df, feature, xlabel='', ylabel='Counts')")
        print("count_bar(self, df, feature, xlabel='', ylabel='Counts')")
        print("aveg(self, df, x, y, xlabel='', ylabel='Mean')")
        print("boxplot(self, df, x, y, xlabel='', ylabel='')")
        print("point(self, x, y, xlabel='', ylabel='', xlim=None, ylim=None)")


    def distro(self, x, nbins=100, kde=True, xlim=None, xlabel=None):
        plt.figure(figsize=(8,6))
        sns.distplot(x, kde=kde, bins=100, color=sns.xkcd_rgb["blue"])
        plt.ticklabel_format(style = 'sci', scilimits = (0,0))
        if xlim:
            plt.xlim(xlim)
        if xlabel:
            plt.xlabel(xlabel, fontsize=18)
        plt.show

    def count(self, df, feature, xlabel='', ylabel='Counts'):
        cnt = df[feature].value_counts()
        plt.figure(figsize=(8,6))
        sns.barplot(cnt.index, cnt.values, alpha=0.8, color='b')
        plt.xticks(rotation='vertical')
        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        sns.despine()
        plt.show()

    def count_bar(self, x, xlabel='', ylabel='Counts'):
        plt.subplots(figsize=(8,6))
        sns.countplot(x=x)
        plt.xticks(rotation='vertical')
        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        sns.despine()
        plt.show()

    def aveg(self, x, y, xlabel='', ylabel='Mean'):
        plt.figure(figsize=(8,6))
        sns.pointplot(x=x, y=y, errwidth=3.0, scale=.8, color=sns.xkcd_rgb["blue"])
        plt.xticks(rotation='vertical')
        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        sns.despine()
        plt.show()

    def boxplot(self, x, y, xlabel='', ylabel=''):
        plt.subplots(figsize=(8,6))
        sns.boxplot(x=x, y=y)
        plt.xticks(rotation='vertical')
        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        sns.despine()
        plt.show()

    def point(self, x, y, xlabel='', ylabel='', xlim=None, ylim=None):
        plt.subplots(figsize=(8,6))
        plt.plot(x, y, 'bo')

        if xlim:
            plt.xlim(xlim[0], xlim[1])

        if ylim:
            plt.ylim(ylim[0], ylim[1])

        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        plt.ticklabel_format(style = 'sci', scilimits = (0,0))

    def missing(self, df, features=None):
        if features:
            missing_df = df[features].isnull().sum(axis=0).reset_index()
        else:
            missing_df = df.isnull().sum(axis=0).reset_index()

        missing_df.columns = ['column_name', 'missing_count']
        missing_df = missing_df[missing_df['missing_count']>0]
        missing_df = missing_df.sort_values(by='missing_count')
        missing_df['missing_count'] = (missing_df['missing_count'] / df.shape[0]) * 100

        nfeatures = np.arange(missing_df.shape[0])
        plt.subplots(figsize=(12,missing_df.shape[0]/2))
        plt.barh(nfeatures, missing_df.missing_count.values, color='blue')
        plt.yticks(nfeatures, missing_df.column_name.values)
        plt.xlabel("Number of missing values (%)")
        plt.show()

    def correlation(self, df, features):

        corr = df[features].corr()

        corr.drop(features[0], axis=0, inplace=True)
        corr.drop(features[-1], axis=1, inplace=True)

        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask, 1)] = True

        fig = plt.figure(figsize=(10, 8))
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(0.0)
        plt.rcParams['xtick.labelsize'] = 18
        plt.rcParams['ytick.labelsize'] = 18

        with sns.axes_style("white"):
            sns.heatmap(abs(corr), mask=mask, annot=True, annot_kws={"size":20}, cmap='RdBu', fmt='+.2f', cbar=False)
