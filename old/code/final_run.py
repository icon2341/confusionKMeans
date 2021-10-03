#Runs all models to generate all visualizationos
#Camille Mince and Skye Rhomberg

import sys
import numpy as np
import os
import base85 as b
import hmm_tk as ht
import hmm_init as h
from hmmlearn.hmm import GaussianHMM
from hmmlearn.hmm import GMMHMM
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import masterprocess as m
import consensus_classifier as c
import visualize_modes as v


####################### RUN GAUSSIAN MODELS ############################
#run consensus classifier and store accuracies
res = c.run_all_consensus_classifier("G")
consensus_acc = res[0]
consensus_actuals = res[1]
consensus_preds = res[2]

#generate binned and un-binned confusion matrices for consensus classifier
x = ht.accuracy_confusion(np.array(consensus_actuals),np.array(consensus_preds),binned=True)
df_cm = pd.DataFrame(x[0], range(2), range(2))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, fmt='g') # font size
plt.title("Binned Consensus Classifier Confusion Matrix (G)")
plt.show()

x = ht.accuracy_confusion(np.array(consensus_actuals),np.array(consensus_preds),binned=False)
df_cm = pd.DataFrame(x[0], range(4), range(4))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, fmt='g') # font size
plt.title("Consensus Classifier Confusion Matrix (G)")
plt.show()

#initialize Gaussian HMM configs
model_configs = {
"Gaussian,Fixed,Viterbi": [0, 0, 0],
"Gaussian,Prior,Viterbi": [0, 1, 0],
"Gaussian,Learned,Viterbi":[0, 2, 0],
"Gaussian,Fixed,Map": [0,0,1],
"Gaussian,Prior,Map": [0,1,1],
"Gaussian,Learned,Map": [0,2,1],}

#run all Gaussian configs 
res = m.run_all(model_configs)
accs = res[0]
actuals = res[1]
preds = res[2]
means = res[0]
covars = res[1]
transmat = res[2]
xs = res[3]

#generate feature importance graph
lines, means = ht.feature_importance_polyfit(xs, np.array(means), np.array(covars))
ht.feature_slopes(lines[0], means)

#generate config and modality accuracy graphs
v.model_config_accuracies(accs, "G", consensus_acc)
v.modality_accuracies(np.load('master.npy'), "G")

########################## RUN GMM MODELS ##############################
#run GMM consensus classifier and store accuracies
res = c.run_all_consensus_classifier("GMM")
consensus_acc = res[0]
consensus_actuals = res[1]
consensus_preds = res[2]

#generate binned and un-binned confusion matrices
x = ht.accuracy_confusion(np.array(consensus_actuals),np.array(consensus_preds),binned=True)
df_cm = pd.DataFrame(x[0], range(2), range(2))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, fmt='g') # font size
plt.title("Binned Consensus Classifier Confusion Matrix (GMM)")
plt.show()

x = ht.accuracy_confusion(np.array(consensus_actuals),np.array(consensus_preds),binned=False)
df_cm = pd.DataFrame(x[0], range(4), range(4))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, fmt='g') # font size
plt.title("Consensus Classifier Confusion Matrix (GMM)")
plt.show()

#initialize GMM model configs
model_configs = {
"GMM,Fixed,Viterbi": [1, 0, 0],
"GMM,Prior,Viterbi": [1, 1, 0],
"GMM,Learned,Viterbi":[1, 2, 0],
"GMM,Fixed,Map": [1,0,1],
"GMM,Prior,Map": [1,1,1],
"GMM,Learned,Map": [1,2,1],}

#generate config and modality accuracy graphs
v.model_config_accuracies(np.load('GMMmaster.npy'), "GMM", consensus_acc)
v.modality_accuracies(np.load('GMMmaster.npy'), "GMM")