# Master HMM Initialization
# Skye Rhomberg, Camille Mince
# 7/27/21

import numpy as np
import pandas as pd
import hmm_tk as ht
import os
import re

#########################################################################################
# Feature Processing

def proc_txt(f_in):
    '''
    Process text feature CSV file, produce numpy output
    Cut punctuation rows
    Input:
    f_in: string. full path to csv file from root of directory walking
    Output:
    np array (n_tokens,n_txt_features) properly formatted array of text features
    '''
    # On punctuation lines, empty times get replaced with nan
    arr = pd.read_csv(f_in,header=1,\
            usecols=np.arange(1,11),delimiter=',',na_values=np.nan)
    # Remove any rows with nans, thereby eliminating punctuation rows
    return ht.fuck_the_nans_harder(np.array(arr))

def proc_av(av,times):
    '''
    Process the av feature numpy array for merging with text
    Collapse the frames into bins by text token, represented by times
    Average over the bins and cat
    Input:
    av: np array (n_frames,n_av_features+1) av processed array (last col is LABELS)
    times: np array (n_tokens,2) start_time, end_time columns of txt feature array
    Output:
    np array (n_tokens,n_av_features) av features formatted for merging
    '''
    # Generate masking array of which rows are in a given range from times
    in_range = lambda i: (av[:,0]>= times[i,0]) * (av[:,0] <= times[i,1])
    # Mode of a 1d array
    _m = lambda a: np.bincount(a.astype(int)).argmax()
    mode = lambda a: _m(a) if a.size > 0 else np.array([])
    # Take the mean of all columns except last, take mode instead
    # Handles labels as last col
    collapse = lambda a,i:\
            np.hstack((np.mean(a[in_range(i),:-1],axis=0),mode(a[in_range(i),-1])))
    return np.array([collapse(av[:, 1:],i) for i in range(times.shape[0])])

def proc(f_in,av):
    '''
    Master process: given txt csv file, process it and appropriate av,
    get corresponding labels, and merge
    Input:
    f_in: string. full path to csv file from root of directory walking
    av: np array (n_frames,n_av_features+1) av processed array
    Output:
    np array (n_tokens,n_txt_features+n_av_features+1)
    Combined features for each txt token, including label as last column
    '''
    # Process Text Features from csv
    p_txt = proc_txt(f_in)
    # Get associated confusion labels from dict
    labels = ht.CONF_LABELS[f_in.split('/')[-1][:-4]+'t'+f_in.split('/')[-2][-1]]
    # Align labels to av features
    p_lb = ht.gen_aligned_labels(av,labels)[:,None]
    # Put labels as last col of av and process it to txt features
    p_av = proc_av(np.hstack((av,p_lb)),p_txt[:,:2])
    # Merge into one array start_time,end_time,[txt_features],[av_features],labels
    return np.hstack((p_txt,p_av))

#########################################################################################
# Directory Walking

def hmm_proc_all(txt_dir,av_dir,out):
    '''
    Walk the directory structure of txt_dir
    For each csv file (txt features for a participant-task),
    load in the the appropriate av features np array from av_dir
    add labels to av_dir, then proc both and merge
    save all the participant files in a single npz
    Input:
    '''
    # All data arrays to output
    arrs = {}
    for (root,dirs,files) in os.walk(txt_dir):
        for f in files:
            if f[-3:] == 'csv':
                f_in = root + '/' + f
                # Participant-Task Name e.g. task1/668.csv --> 668t1
                pt_name = f_in.split('/')[-1][:-4]+'t'+f_in.split('/')[-2][-1]
                # Dialogue Name taken from path
                d_name = re.search('(d\d{2})',f_in).group(1)
                # Load Correct AV features 
                av = np.load(f'{av_dir}/{d_name}.npz')[pt_name]
                arrs.update({pt_name:proc(f_in,av)})
    print([f for f in arrs])
    np.savez(out,**arrs)

#########################################################################################
# HMM Initialization

def hmm_init(xs,test_set):
    '''
    Generate combined array, length, and transmat for hmm from given data
    Cat everything in xs unless it matches test_set
    Input:
    xs: dict of np array {pt_name:[arr]} features for each participant-task
    test_set: list of str. participant ids to leave out for test set
    Output:
    x: concatenated array from xs -- (n_tokens,txt+aud+vid+label)
    lengths: lengths of each array catted into x in order they appear
    transmat: transition matrix from labels in xs (last column)
    '''
    # List of arrays to cat -- exclude test_set
    x_list = [(t, xs[t]) for t in xs if not any(p in t for p in test_set)]
    # Stack all arrays to make xs
    xs = np.vstack([x[1] for x in x_list])
    # Normalize xs, return all their row counts for lengths,
    # Make transition matrix from labels (last column)
    lengths = [(t,a.shape[0]) for (t,a) in x_list]
    return ht.normalize(xs),lengths,ht.make_transmat(xs[:,-1],[a for (t,a) in lengths])

########################################################################################
# Main Code

if __name__ == "__main__":
	hmm_proc_all("txt", "av_0724_1", "avt_0728.npz")

