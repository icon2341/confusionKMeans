#Consensus Classifier 
#Skye Rhomberg and Millie Mince

import sys
import numpy as np
import os
import base85 as b
import hmm_tk as ht
import hmm_init as h
from hmmlearn.hmm import GaussianHMM
from hmmlearn.hmm import GMMHMM
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

participants = ["833", "194", "622", "305", "425", "913", "670", "391", "998", "139", "291", "556", "177", "731", "843", "279", "668", "903", "149", "811"]

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

def consensus_classifier(test, model_type):
    """
    runs consensus classifier, which runs text, audio, and video unimodal models
    then takes the consensus of each unimodal model to make prediction
    Input:
    test: String, participant ID to use as test subject
    model_type: String, either "G" for Gaussian or "GMM" for Gaussian Mixture Model
    Output:
    np.array(3,): accuracy, actual labels, predicted labels
    """
    features = np.load('avt_0728.npz')
    predictions = {}
    #Text Classifier - Learned Transmat, MAP decoding 
    if model_type == "G":
        model = GaussianHMM(n_components=4, covariance_type="diag", n_iter=100, init_params="mct", params="s", algorithm="map", verbose=True, random_state=47)
    else:
        model = GMMHMM(n_components=4, n_mix=3, min_covar=0.002, covariance_type="diag", n_iter=100, init_params="mct", params="s", algorithm="map", verbose=True, random_state=47)
    model.n_features = 8
    model.startprob_ = [1,0,0,0]
    x, part_lengths, transmat = h.hmm_init(features, [test])
    txt_features = x[:, 2:10]
    model.fit(txt_features, [a for (t, a) in part_lengths])
    tests = [(t, features[t]) for t in features if any(p in t for p in [str(test)])]
    for (t, xs) in tests:
        print("testing: " + str(t))
        pred = model.predict(xs[:, 2:10])
        predictions[str(t) + "T"] = pred
    #Audio Classifier - Learned Transmat, Viterbi decoding
    model = GaussianHMM(n_components=4, covariance_type="diag", n_iter=100, init_params="mct", params="s", algorithm="viterbi", verbose=True)
    model.n_features = 32
    model.startprob_ = [1,0,0,0]
    x, part_lengths, transmat = h.hmm_init(features, [test])
    aud_features = x[:, 10:42]
    model.fit(aud_features, [a for (t, a) in part_lengths])
    tests = [(t, features[t]) for t in features if any(p in t for p in [str(test)])]
    for (t, xs) in tests:
        print("testing: " + str(t))
        pred = model.predict(xs[:, 10:42])
        predictions[str(t) + "A"] = pred
    #Video Classifier - Learned Transmat, Viterbi decoding
    model = GaussianHMM(n_components=4, covariance_type="diag", n_iter=100, init_params="mct", params="s", algorithm="viterbi", verbose=True)
    model.n_features = 17
    model.startprob_ = [1,0,0,0]
    x, part_lengths, transmat = h.hmm_init(features, [test])
    vid_features = x[:, 42:59]
    model.fit(vid_features, [a for (t, a) in part_lengths])
    tests = [(t, features[t]) for t in features if any(p in t for p in [str(test)])]
    for (t, xs) in tests:
        print("testing: " + str(t))
        pred = model.predict(xs[:, 42:59])
        predictions[str(t) + "V"] = pred
    preds = []
    actuals = []

    #check whether test participant has data for task1
    if (test + "t1T") in predictions:
        t1_preds = []
        for mode in ["T", "A", "V"]:
            t1_preds.append(predictions[test + "t1" + mode])
        np_preds = np.array(t1_preds)
        pred = np.median(np_preds,axis=0)
        preds.extend(pred)
        actual = features[test + "t1"][:,-1]
        actuals.extend(actual)
        accs = determine_accuracies(actual, pred)[-1]
        t1_exists = True
    else:
        t1_exists = False

    #get task2 accuracies 
    t2_preds = []
    for mode in ["T", "A", "V"]:
        t2_preds.append(predictions[test + "t2" + mode])
    np_preds = np.array(t2_preds)
    pred = np.median(np_preds,axis=0)
    preds.extend(pred)
    actual = features[test + "t2"][:,-1]
    actuals.extend(actual)
    accs_2 = determine_accuracies(actual, pred)[-1]

    #get task3 accuracies
    t3_preds = []
    for mode in ["T", "A", "V"]:
        t3_preds.append(predictions[test + "t3" + mode])
    np_preds = np.array(t3_preds)
    pred = np.median(np_preds,axis=0)
    preds.extend(pred)
    actual = features[test + "t3"][:,-1]
    actuals.extend(actual)
    accs_3 = determine_accuracies(actual, pred)[-1]

    if t1_exists:
        total_accs = [accs, accs_2, accs_3]
    else:
        total_accs = [accs_2, accs_3]
    overall = np.mean(total_accs)
    return (overall, preds, actuals)

def run_all_consensus_classifier(model_type):
    """ 
    runs consensus classifier with each participant as a test subject
    Input:
    model_type: String, "G" or "GMM"
    Output:
    np.array(3,): average accuracy across all participants,  
    concatenated actual labels, concatenated predicted labels
    """
    accs = []
    preds = []
    actuals = []
    for part in participants:
        result = consensus_classifier(part, model_type)
        acc = result[0]
        pred = result[1]
        actual = result[2]
        accs.append(acc)
        preds.extend(pred)
        actuals.extend(actual)
    print(np.average(np.array(accs)), np.array(actuals), np.array(preds))
    return (np.average(np.array(accs)), np.array(actuals), np.array(preds))

    #generate bar graph of consensus classifier accuracy among participants
    plt.bar(participants, height=accs)
    plt.title("Consensus Classifier Accuracy for Each Participant")
    plt.show()

    #generate confusion matrix for consensus classifier
    x = ht.accuracy_confusion(np.array(actuals),np.array(preds),binned=True)
    df_cm = pd.DataFrame(x[0], range(2), range(2))
    sn.set(font_scale=1.4) 
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, fmt='g') 
    plt.title("Confusion Matrix for Consensus Classifier")

    plt.show()


if __name__ == "__main__":
    run_all_consensus_classifier("G")