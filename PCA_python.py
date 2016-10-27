# -*- coding: utf-8 -*-
import pandas as pd
df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', 
    header=None, 
    sep=',')
    
df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']

# check corelation bwtween the variables
pd.scatter_matrix(df, alpha = 0.3, figsize = (14,8), diagonal = 'kde')


#*******************************************************************************
#********************************************************************************
# split data table into data X and class labels y

X = df.ix[:,0:4].values
y = df.ix[:,4].values


#EDA
import plotly.plotly as py
from plotly.graph_objs import *
import plotly.tools as tls

# plotting histograms

from matplotlib import pyplot as plt
import numpy as np
import math

label_dict = {1: 'Iris-Setosa',
              2: 'Iris-Versicolor',
              3: 'Iris-Virgnica'}

feature_dict = {0: 'sepal length [cm]',
                1: 'sepal width [cm]',
                2: 'petal length [cm]',
                3: 'petal width [cm]'}

with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(8, 6))
    for cnt in range(4):
        plt.subplot(2, 2, cnt+1)
        for lab in ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'):
            plt.hist(X[y==lab, cnt],
                     label=lab,
                     bins=10,
                     alpha=0.3,)
        plt.xlabel(feature_dict[cnt])
    plt.legend(loc='upper right', fancybox=True, fontsize=8)

    plt.tight_layout()
    plt.show()

#STANDARISING
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)


#Eigendecomposition of the standardized data based on the correlation matrix:
cor_mat2 = np.corrcoef(X.T)
eig_vals, eig_vecs = np.linalg.eig(cor_mat2)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

#Sorting Eigenpairs
# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()


# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

#Explained Variance
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(4), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
    plt.step(range(4), cum_var_exp, where='mid',
             label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
#**********************************

#Projection Matrix
matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1),
                      eig_pairs[1][1].reshape(4,1)))

print('Matrix W:\n', matrix_w)


#projection to new feature space
Y = X_std.dot(matrix_w)

with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                        ('blue', 'red', 'green')):
        plt.scatter(Y[y==lab, 0],
                    Y[y==lab, 1],
                    label=lab,
                    c=col)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='lower center')
    plt.tight_layout()
    plt.show()
#**********************************************************************
#************************************************************************
# PCA in SKLEARN

X_variables=df[['sepal_len','sepal_wid','petal_len','petal_wid']]
Y_variable=df['class']

from sklearn.preprocessing import StandardScaler
X_std1= StandardScaler().fit_transform(X_variables)


from sklearn.decomposition import PCA
sklearn_pca = PCA(n_components=4)
Y_sklearn = sklearn_pca.fit_transform(X_std1)

import renders as rs
# Generate PCA results plot

X_df=pd.DataFrame(X_std1,columns=['sepal_len','sepal_wid','petal_len','petal_wid'])
pca_results = rs.pca_results(X_df, sklearn_pca)
#since X_std1 is a np array but the above commmand takes only a df

print pca_results['Explained Variance'].cumsum()

# so only first 2 are enough to expalin most of the variance
#DIMESION REDUCTION

sklearn_pca = PCA(n_components=2)
Y_sklearn = sklearn_pca.fit_transform(X_std1)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(Y_sklearn, columns = ['Dimension 1', 'Dimension 2'])

reduced_data['class']=Y_variable

import matplotlib.pyplot as plt
import seaborn

#TO PLOT 

fg = seaborn.FacetGrid(data=reduced_data, hue='class',aspect=2.00)
fg.map(plt.scatter, 'Dimension 1', 'Dimension 2').add_legend()

