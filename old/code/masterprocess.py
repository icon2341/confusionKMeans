#Master Process to initialize all HMM models and run
#Millie Mince and Skye Rhomberg

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


model_types = ["Gaussian", "GMM"]
transmat_types = ["Fixed", "Prior", "Learned"]
decoding_algs = ["viterbi", "map"]

"""
model_configs = {
"GMM,Fixed,Viterbi": [1, 0, 0],
"GMM,Prior,Viterbi": [1, 1, 0],
"GMM,Learned,Viterbi":[1, 2, 0],
"GMM,Fixed,Map": [1,0,1],
"GMM,Prior,Map": [1,1,1],
"GMM,Learned,Map": [1,2,1],}
#"Gaussian,Fixed,Viterbi": [0, 0, 0],
#"Gaussian,Prior,Viterbi": [0, 1, 0],
#"Gaussian,Learned,Viterbi":[0, 2, 0],
#"Gaussian,Fixed,Map": [0,0,1],
#"Gaussian,Prior,Map": [0,1,1],
#"Gaussian,Learned,Map": [0,2,1],}
"""

MODES = {"Text": [2,10], "Audio": [10,42], "Video": [42,59], "Bimodal-AV": [10,59], "Trimodal": [2,59]}

PARTICIPANTS = [833, 194, 622, 305, 425, 913, 670, 391, 998, 139, 291, 556, 177, 731, 843, 279, 668, 903, 149, 811]

def model_init(m_type, t_type, d_type):
    """
    m_type: model type (Gaussian or GMM)
    t_type: transmat type (fixed, prior, or learned)
    d_type: decoding algorithm type (viertbi or map)
    returns HMM initialized to input specifications
    """
    #init_params overwritten
    if m_type == 0:
        #Gaussian
        if t_type == 0:
            #fixed transmat
            if d_type == 0:
                model = GaussianHMM(n_components=4, covariance_type="diag", n_iter=100, init_params="mc", params="st", algorithm="viterbi", verbose=True, random_state=47)
            else:
                model = GaussianHMM(n_components=4, covariance_type="diag", n_iter=100, init_params="mc", params="st", algorithm="map", verbose=True, random_state=47)
        elif t_type == 1:
            #transmat prior input
            if d_type == 0:
                model = GaussianHMM(n_components=4, covariance_type="diag", n_iter=100, init_params="mct", params="s", verbose=True, algorithm="viterbi", random_state=47)
            else: 
                model = GaussianHMM(n_components=4, covariance_type="diag", n_iter=100, init_params="mct", params="s", verbose=True, algorithm="map", random_state=47)
        else:
            #completely learn transmat 
            if d_type == 0:
                model = GaussianHMM(n_components=4, covariance_type="diag", n_iter=100, init_params="mct", params="s", algorithm="viterbi", verbose=True, random_state=47)
            else: 
                model = GaussianHMM(n_components=4, covariance_type="diag", n_iter=100, init_params="mct", params="s", algorithm="map", verbose=True, random_state=47)
    else:
        #Gaussian Mixture Model
        if t_type == 0:
            #fixed transmat
            if d_type == 0:
                model = GMMHMM(n_components=4, n_mix=3, covariance_type="diag", min_covar=0.002, n_iter=100, init_params="mc", params="st", algorithm="viterbi", verbose=True, random_state=47)
            else:
                model = GMMHMM(n_components=4, n_mix=3, covariance_type="diag", min_covar=0.002, n_iter=100, init_params="mc", params="st", algorithm="map", verbose=True, random_state=47)
        elif t_type == 1:
            #transmat prior input
            if d_type == 0:
                model = GMMHMM(n_components=4, n_mix=3, covariance_type="diag", min_covar=0.002, n_iter=100, init_params="mct", params="s", verbose=True, algorithm="viterbi", random_state=47)
            else: 
                model = GMMHMM(n_components=4, n_mix=3, covariance_type="diag", min_covar=0.002, n_iter=100, init_params="mct", params="s", verbose=True, algorithm="map", random_state=47)
        else:
            #completely learn transmat 
            if d_type == 0:
                model = GMMHMM(n_components=4, n_mix=3, covariance_type="diag", min_covar=0.002, n_iter=100, init_params="mct", params="s", algorithm="viterbi", verbose=True, random_state=47)
            else: 
                model = GMMHMM(n_components=4, n_mix=3, covariance_type="diag", min_covar=0.002, n_iter=100, init_params="mct", params="s", algorithm="map", verbose=True, random_state=47)
    
    return model


def determine_accuracies(actual, pred):
    """
    determines a variety of accuracies of classifier
    Input:
    actual: actual confusion labels given by participant
    pred: consensus classifier prediction
    Output:
    np.array(6,): [label 0 acc, label 1 acc, label 2 acc, label 3 acc, overall acc, binned acc]
    """
    np_pred = np.array(pred)
    np_actual = np.array(actual)
    label_accs = [np.sum((np_pred==i) * (np_actual==i))/np.sum(np_actual==i) for i in range(4)]
    overall_accuracy = np.sum(np_pred==np_actual)/len(actual)
    binned_accuracy = np.sum((np_pred<=1) * (np_actual<=1) + (np_pred>=2) * (np_actual>=2))/len(actual)
    return label_accs + [overall_accuracy] + [binned_accuracy]


def init_all_models(model_configs):
    """
    initializes all models specified 
    Input:
    model_configs: Dict, {[Gaussian, Fixed, Viterbi]: [0,0,0], ...}
    Output: Dict, {[Gaussian, Fixed, Viterbi]: initialized model with specifications, ...}
    """
    models = {}
    features = np.load('avt_0728.npz')
    for config in model_configs:
        types = model_configs[config]
        model_type, transmat_type, decoding_type = types[0], types[1], types[0]
        print("generating model configuration: " + config)
        model = model_init(model_type, transmat_type, decoding_type)
        models[config] = model
    print(len(models))
    return models


def run_all(model_configs):
    """
    runs all models specified
    Input:
    model_configs: Dict, {[Gaussian, Fixed, Viterbi]: [0,0,0], ...}
    Output: np.array(6 configs,5 modalities,20 participants,6 accuracy types)
    """
    features = np.load('avt_0728.npz')
    master_accuracies = []
    master_preds = []
    master_actuals = []
    models = init_all_models(model_configs)
    preds = []
    actuals = []
    for i, model in enumerate(models):
        print("testing model: " + model)
        mode_accuracies = []
        mod = models[model]
        for mode in MODES:
            print("mode: " + mode)
            preds = []
            actuals = []
            #set model n_features and starting probability 
            mod.n_features = MODES[mode][1] - MODES[mode][0]
            mod.startprob_ = [1,0,0,0]
            participant_accuracies = []
            for part in PARTICIPANTS:
                x, part_lengths, transmat = h.hmm_init(features, [str(part)])
                sliced = x[:, MODES[mode][0]:MODES[mode][1]]
                if (i % 3 == 1):
                    #set transmat_prior
                    mod.transmat_prior = transmat
                else:
                    mod.transmat_ = transmat
                #fit model
                mod.fit(sliced, [a for (t, a) in part_lengths])

                #separate tests and determine accuracy on test subject
                tests = [(t, features[t]) for t in features if any(p in t for p in [str(part)])]
                test_accs = []
                for (t, xs) in tests:
                    print("testing: " + str(t))
                    pred = mod.predict(xs[:, MODES[mode][0]:MODES[mode][1]])
                    actual = features[t][:, -1]
                    preds.extend(pred)
                    actuals.extend(actual)

                    accuracies = determine_accuracies(actual, pred)
                    test_accs.append(accuracies)
                participant_accuracies.append(np.mean(np.array(test_accs), axis=0))
            
            #uncomment to generate confusion matrices 
            """
            x = ht.accuracy_confusion(np.array(actuals),np.array(preds),binned=True)
            df_cm = pd.DataFrame(x[0], range(2), range(2))
            sn.set(font_scale=1.4) # for label size
            sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, fmt='g') # font size
            plt.title(mode + " Binned Confusion Matrix (GMM)")
            plt.show()

            x = ht.accuracy_confusion(np.array(actuals),np.array(preds),binned=False)
            df_cm = pd.DataFrame(x[0], range(4), range(4))
            sn.set(font_scale=1.4) # for label size
            sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, fmt='g') # font size
            plt.title(mode + " Confusion Matrix (GMM)")
            plt.show()
            """
            mode_accuracies.append(participant_accuracies)
        master_accuracies.append(mode_accuracies)
        master_preds.append(preds)
        master_actuals.append(actual)

    #save np_array to master.npy in working dir
    np.save("master.npy", np.array(master_accuracies))
    
    #return (np.array(master_accuracies), np.array(master_actuals), np.array(master_preds))
    return (mod.means_, mod.covars_, mod.transmat_, x[:, 2:59])