# Cason Wight
# Example of Neural Nets, regression trees, and 
# 3/25/2020

# Load required packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.metrics import roc_curve, auc


# Read in the data
os.chdir('C:/Users/cason/Desktop/Classes/Assignments/Stat 536/Homework 6')
credit = pd.read_csv('creditcard.csv')

# Split the endogenous and exogenous variables
X = credit.iloc[:,:-1]
Y = credit["Class"]


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


# Rescale the exogenous variables
scaler = StandardScaler()
scaler.fit(X_train)
X = scaler.transform(X)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#### Neural Net Analysis

# Fit a Neural Net
mlp = MLPClassifier(max_iter=200, 
                    hidden_layer_sizes = [30,30,30],
                    activation = 'relu',
                    solver = 'adam',
                    alpha = .01,
                    learning_rate = 'constant')
mlp.fit(X,Y)


# Fitted values, sensitivities and specificities
fitted_valsNN = mlp.predict_proba(X)[:,1]
pred_valsNN = mlp.predict_proba(X_test)[:,1]
fprNN, tprNN, thresholdNN = roc_curve(Y, fitted_valsNN)

# Best cutoff value
Possible_Cutoffs = np.arange(0,1,.0001)
Truths = []
for cutoff in Possible_Cutoffs:
    Truths = Truths + [np.sum((fitted_valsNN > cutoff) == Y)]

best_cutoffNN = Possible_Cutoffs[Truths.index(np.max(Truths))]

# AUC for the model
NN_auc = auc(fprNN,tprNN)

#### Classification Tree Analysis

# Fit a Classification Tree (tuned to the optimal alpha level)
clf_sens = tree.DecisionTreeClassifier(ccp_alpha = 7.686e-05)
clf = clf_sens.fit(X_train, Y_train)

# Fitted values, sensitivities and specificities
fitted_valsCT = clf.predict_proba(X)[:,1]
pred_valsCT = clf.predict_proba(X_test)[:,1]
fprCT, tprCT, thresholdCT = roc_curve(Y, fitted_valsCT)

# Plot the decisions of the tree
tree.plot_tree(clf)
plt.show()

# Best cutoff value
Truths = []
for cutoff in Possible_Cutoffs:
    Truths = Truths + [np.sum((fitted_valsCT > cutoff) == Y)]

best_cutoffCT = Possible_Cutoffs[Truths.index(np.max(Truths))]

# AUC for the model
CT_auc = auc(fprCT,tprCT)


#### K nearest Neighbor Analysis (All done from R, fitted values pulled in)

# Fitted values, sensitivities and specificities
fitted_valsKNN = pd.read_csv('probs_knn.csv')['x']
fprKNN, tprKNN, thresholdKNN = roc_curve(Y, fitted_valsKNN)

# Best cutoff value
Truths = []
for cutoff in Possible_Cutoffs:
    Truths = Truths + [np.sum((fitted_valsKNN > cutoff) == Y)]

best_cutoffKNN = Possible_Cutoffs[Truths.index(np.max(Truths))]

# AUC for the model
KNN_auc = auc(fprKNN,tprKNN)

### Model Fit

# ROC Curve for the different Methods
plt.figure(figsize = (16,10))
plt.plot([0,1], [0,1], lw = 2, linestyle = "--")
plt.plot(fprNN, tprNN, label='Neural Net (AUC = %0.4f)' % NN_auc)
plt.plot(fprCT, tprCT, label='Classification Tree (AUC = %0.4f)' % CT_auc)
plt.plot(fprKNN, tprKNN, label='K-Nearest Neighbor (AUC = %0.4f)' % KNN_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig("ROC.png")

# Specificty, Sensitivity, Precision, and Negative Predicted Value
def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

fitted_predsNN = 1*(fitted_valsNN>best_cutoffNN)
fitted_predsCT = 1*(fitted_valsCT>best_cutoffCT)
fitted_predsKNN = 1*(fitted_valsKNN>best_cutoffKNN)

# True/false positives/negatives
TP_NN, FP_NN, TN_NN, FN_NN = perf_measure(Y,fitted_predsNN)
TP_CT, FP_CT, TN_CT, FN_CT = perf_measure(Y,fitted_predsCT)
TP_KNN, FP_KNN, TN_KNN, FN_KNN = perf_measure(Y,fitted_predsKNN)

# Sensitivities
Sens_NN = TP_NN/(TP_NN+FN_NN)
Sens_CT = TP_CT/(TP_CT+FN_CT)
Sens_KNN = TP_KNN/(TP_KNN+FN_KNN)

# Specificities
Spec_NN = TN_NN/(TN_NN+FP_NN)
Spec_CT = TN_CT/(TN_CT+FP_CT)
Spec_KNN = TN_KNN/(TN_KNN+FP_KNN)

# Precisions
Prec_NN = TP_NN/(TP_NN+FP_NN)
Prec_CT = TP_CT/(TP_CT+FP_CT)
Prec_KNN = TP_KNN/(TP_KNN+FP_KNN)

# Negative Predicted Values
NPred_NN = TN_NN/(TN_NN+FN_NN)
NPred_CT = TN_CT/(TN_CT+FN_CT)
NPred_KNN = TN_KNN/(TN_KNN+FN_KNN)

# F scores
F_NN = 2 * (Prec_NN * Sens_NN) / (Prec_NN + Sens_NN)
F_CT = 2 * (Prec_CT * Sens_CT) / (Prec_CT + Sens_CT)
F_SVM = 2 * (Prec_KNN * Sens_KNN) / (Prec_KNN + Sens_KNN)


## Predictions

fitted_predsNN = 1*(pred_valsNN>best_cutoffNN)
fitted_predsCT = 1*(pred_valsCT>best_cutoffCT)

TP_NN, FP_NN, TN_NN, FN_NN = perf_measure(Y_test.to_numpy(),fitted_predsNN)
TP_CT, FP_CT, TN_CT, FN_CT = perf_measure(Y_test.to_numpy(),fitted_predsCT)

# Sensitivities
Sens_NN = TP_NN/(TP_NN+FN_NN)
Sens_CT = TP_CT/(TP_CT+FN_CT)

# Specificities
Spec_NN = TN_NN/(TN_NN+FP_NN)
Spec_CT = TN_CT/(TN_CT+FP_CT)

# Precisions
Prec_NN = TP_NN/(TP_NN+FP_NN)
Prec_CT = TP_CT/(TP_CT+FP_CT)

# Negative Predicted Values
NPred_NN = TN_NN/(TN_NN+FN_NN)
NPred_CT = TN_CT/(TN_CT+FN_CT)

# F scores
F_NN = 2 * (Prec_NN * Sens_NN) / (Prec_NN + Sens_NN)
F_CT = 2 * (Prec_CT * Sens_CT) / (Prec_CT + Sens_CT)

