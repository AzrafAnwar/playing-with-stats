# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 22:11:51 2017

@author: azraf
"""
##PART 1.1: PRE-PROCESSING##
##Response to 1.1##

import numpy as np
from sklearn.preprocessing import StandardScaler as ss
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.neighbors import KNeighborsClassifier
import plotly as plotly
import plotly.plotly as py
from plotly.graph_objs import *
import plotly.tools as tls
plotly.tools.set_credentials_file(username='engleberry', api_key='JHK7zCU6FCcYgg6LcjaO')

import matplotlib.pyplot as plt
from scipy.signal import correlate




d_c = read_clean_dataset(summary=True) #Clean data set
c_d = read_corrupted_dataset(summary=True) #corrupted data set


#Initially import all necessary modules and functions. Please note that this
#api key for plotly is from an account I made specifically for this project
#so you can access the generated plots at username: engleberry, 
#password: Droice1212. There should be only 2 graphs in the account to view for PCA.

std = ss().fit_transform(d_c[0]) #Standardization

#Covariance matrix from standardized data.
cov_mat = np.cov(std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

#Correlation matrix from raw data.
cor_mat = np.corrcoef(d_c[0].T)
eig_vals2, eig_vecs2 = np.linalg.eig(cor_mat)

#Correlation matrix from standardized data.
cor_mat2 = np.corrcoef(std.T)
eig_vals3, eig_vecs3 = np.linalg.eig(cor_mat2)

#SVD could also be used to find the eigenvectors. 
#eig_vec4,s,v = np.linalg.svd(std.T)

#I continue the rest of the code with eigenvalues and vectors from method 1.

# Make a list of (eigenvalue, eigenvector) tuples to sort vectors based on values.
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples
eig_pairs.sort()
eig_pairs.reverse()

#This line is to showthe eigenvalues generated thus far
print('Eigenvalues for data set:')
for i in eig_pairs:
    print(i[0])

#The eigenvalues were then summed and then used to demonstrate which components
#carry what percent of the total information within the data set.

tot = sum(eig_vals) #Summing eigenvalues
#The cumulative variance across the data set was characterized and visualized.
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

#The following code generates a graph in plotly showing the distribution of 
#values across the components and which components carry what percentage of the 
#variance. The bars show the variance in each component and the cumulative
#variance curve above it shows how much of the variance is accounted for by the 
#sum of components up to that point. 
trace1 = Bar(
        x=['PC %s' %i for i in range(1,457)],
        y=var_exp,
        showlegend=False)

trace2 = Scatter(
        x=['PC %s' %i for i in range(1,457)], 
        y=cum_var_exp,
        name='cumulative explained variance')

data = Data([trace1, trace2])

layout=Layout(
        yaxis=YAxis(title='Variance in percent'),
        title='Variance by different principal components')

fig = Figure(data=data, layout=layout)
py.iplot(fig)

#Based on the graphs, 95.25 percent of data contained in first 169 PCs. 
#99.028 percent of the data is contained in first 329 PCs. 

#95 percent was chosen as the threshold to minimize the loss of information in
#dimension reduction while retaining the utility of reducing dimensions.

#With the component cut-off identified at 169, the PCA function in sklearn was
#used to carry out PCA.
sklearn_pca1 = sklearnPCA(n_components= 169)
Y_sklearn = sklearn_pca1.fit_transform(std) 

#These 169 principal components will be used for the remainder of the code.


##PART 1.2: CLASSIFIER TRAINING##
##Response to 1.2##
#A K-nearest neighbor classifier was chosen and trained from the scikit module.
classifier = KNeighborsClassifier(n_neighbors=10)
classifier.fit(Y_sklearn,d_c[1][0:25000])
subset_1 = d_c[1][25001:30000]
subset_2 = Y_sklearn[25001:30000]
score = classifier.score(subset_2,subset_1)
print(score)

##PART 2.1: CORRUPTED DATA LABELING##
##Response to 2.1##
#The classifier trained earlier was used to label the corrupted data set.
#In order to keep the dimensions of the training data and the data to be labeled
#consistent, PCA and pre-processing was performed on the corrupted data as well
#using the previously determined principal components. 

std2 = ss().fit_transform(c_d[0])

corrupted_pca1 = sklearnPCA(n_components= 169)
corrupted = corrupted_pca1.fit_transform(std2)

neighbors = classifier.kneighbors(corrupted)
predictions = classifier.predict(corrupted) 

##PART 2.2: OPTIMAL DATA ALIGNMENT##
##Response to 2.2##
#The scipy module's signal.correlate function was used to calculate the 
#cross correlation of a corrupted univariate time series with its nearest
#neighbor predicted from the classifier. The point of maximal correlation was
#determined from this cross-correlation array and used as the centering point
#to optimally align the data. 

closest_n = neighbors[1][:,0]
z = 0
alignment = np.zeros((30000,2))
for i in range(len(closest_n)):
    b_sig = c_d[0][i]
    a_sig = d_c[0][closest_n[i]]
    xcorr = correlate(a_sig, b_sig)
    lag = abs(np.argmax(xcorr))
    delay = abs(lag - len(a_sig)) #Since signal.correlate uses full cross 
                                  #correlation and len(xcorr) = 2*457 - 1
    end = delay + c_d[1][i]
    if end > 457:
        end = 457
    alignment[i] =  [delay, end]
 