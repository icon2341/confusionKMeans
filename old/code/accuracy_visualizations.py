#Visualizations of Accuracy by Model and Modality
#Camille Mince and Skye Rhomberg

import matplotlib.pyplot as plt
import numpy as np
import hmm_tk as ht
import sys

def models_and_confusion_levels(data):
    """
    data: np.array: (6,5,20,6) output of masterproces.py
    (6 configs, 5 modalities, 20 participants, 6 accuracy types)

    generates graph that shows accuracy of predicting labels
    0, 1, 2, and 3 for each model configuation
    """
    labels = ["F,V", "P,V", "L,V", "F,M", "P,M", "L,M"]

    x = np.arange(len(labels))  # the label locations
    width = 0.125

    #data = np.load('GMMmaster.npy')
    
    accs_0 = np.mean(np.nanmean(data[:,:,:,0],axis=2),axis=1)
    accs_1 = np.mean(np.nanmean(data[:,:,:,1],axis=2),axis=1)
    accs_2 = np.mean(np.nanmean(data[:,:,:,2],axis=2),axis=1)
    accs_3 = np.mean(np.nanmean(data[:,:,:,3],axis=2),axis=1)

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, accs_0, width, label='0')
    rects2 = ax.bar(x - 2*width, accs_1, width, label='1')
    rects3 = ax.bar(x, accs_2, width, label='2')
    rects4 = ax.bar(x + width, accs_3, width, label='4')


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracies')
    ax.set_title('Accuracies By Model and Confusion Level (GMM)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    ax.bar_label(rects4, padding=3)
    #ax.bar_label(rects5, padding=3)

    fig.tight_layout()

    plt.show()


def models_and_modalities(data):
    """
    data: np.array: (6,5,20,6) output of masterproces.py
    (6 configs, 5 modalities, 20 participants, 6 accuracy types)
    
    generates graph that shows accuracy of each modality
    for each model configuation
    """
    labels = ["F,V", "P,V", "L,V", "F,M", "P,M", "L,M"]

    x = np.arange(len(labels))  # the label locations
    width = 0.125
    #data = np.load('GMMmaster.npy')
    accs = np.mean(data[:,:,:,-1],axis=2)
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, accs[:,0], width, label='T')
    rects2 = ax.bar(x - 2*width, accs[:,1], width, label='A')
    rects3 = ax.bar(x, accs[:,2], width, label='V')
    rects4 = ax.bar(x + width, accs[:,3], width, label='AV')
    rects5 = ax.bar(x + 2*width, accs[:,4], width, label='TAV')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracies')
    ax.set_title('Accuracies By Model and Modalities (GMM)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    ax.bar_label(rects4, padding=3)
    #ax.bar_label(rects5, padding=3)

    fig.tight_layout()

    plt.show()

def model_config_accuracies(data, model_type, consensus_acc):
    """
    data: np.array: (6,5,20,6) output of masterproces.py
    (6 configs, 5 modalities, 20 participants, 6 accuracy types)
    model_type: string, either "G" for Gaussian or "GMM" for 
    Gaussian Mixture Model
    consensus_acc: accuracy of consensus classifier to be graphed
    against other model configurations
    
    generates graph that shows binned accuracy of each model 
    configuration, including consensus classifier
    """
    #data = np.load('GMMmaster.npy')
    accs = np.mean(data[:,:,:,-1],axis=2)
    tav_accs = accs[:,-1]
    accs = np.append(tav_accs, consensus_acc)
    np_accs = np.array(accs)
    plt.bar(["Fixed,Viterbi", "Prior,Viterbi", "Learned,Viterbi", "Fixed,MAP", "Prior,MAP", "Learned,MAP", "Consensus"], height=np_accs)
    plt.title("Multimodal Accuracy for Each Model Configuration" + " (" + model_type + ")")
    plt.show()

def modality_accuracies(data, model_type):
    """
    data: np.array: (6,5,20,6) output of masterproces.py
    (6 configs, 5 modalities, 20 participants, 6 accuracy types)
    model_type: string, "G" or "GMM"
    
    generates graph that shows accuracy of each unimodal model,
    bimodal-AV, and trimodal model across all model configs
    """
    #data = np.load('GMMmaster.npy')
    accs = np.mean(data[:,:,:,-1],axis=2)
    modal_accs = np.mean(accs,axis=0)
    plt.bar(["Text", "Audio", "Video", "Audio/Video", "Text/Audio/Video"], height=modal_accs)
    plt.title("Accuracy by Modality" + " (" + model_type + ")")
    plt.show()

if __name__ == "__main__":
    data = sys.argv[0]
    models_and_confusion_levels(data)
    models_and_modalities(data)
    modality_accuracies(data)
    model_config_accuracies(data, "GMM", 0.54)