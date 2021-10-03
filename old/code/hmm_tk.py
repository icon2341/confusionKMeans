# HMM Init and Analysis Helper Functions
# Skye Rhomberg, Camille Mince

import numpy as np
import base85 as b
import math
########################################################################################
# Constants

# Confusion Labels
CONF_LABELS = {"903t1": "j+8KD6la:fa3?39", 
"668t1": "j.uQ81e2)Y2>evw5IK*05][W{7gX.t", 
"149t1": "v3<lf0m.u}2]%sE3Cv5B63g$A", 
"811t1": "v3&7M4dK^67:f-v",
"556t1": "yLFa^1L<s94C}T#4UbdR6E1l#9A:L2a$>fy",
"291t1": "yLLp+2vN{(3/}*.4OWHd5&7fE6>5058*rxbaUL5ybA-3b",
"391t1": "y+lZ+0cFKx2[WRk4t+!06[4078QP!Db(4Au",
"670t1": "y+kl00]TB#4a#E]",
"913t1": "FPfUW0e*&&0W!5?3qQT4",
"425t1": "O}{M0V.B}49n}Q4kwCu",
"305t1": "G[aeB1&QXI2Lwwp349gd4nEvN4Czr{5BHco6?e%39Q$nfa$0u<",
"622t1": "G[a=j2aVbu4+hgM8qckY",
"194t1": "HfkH=1fz:p5IjUA",
"833t1": "Hfh)>0bI^H1-aT44.@)e7TKv]bn}^!buK{C",
"903t2": "j-9^*62+KB6[3{n97.CGdA9??fV=n7j4RSYk=sU.",
"668t2": "j.*}m1doH/4fmcD6AG4#8c[c-8Tek/cdsmjdu2)Ah/Z}>i:a:{kH@9U",
"279t2": "j<{971?s}C34SJA4{oK[6)yMW9wF&0dWI^}h90=O",
"843t2": "j<(#{0{G>v2eZV)2t2i-3?T#W4{xGT5HXlN63.iy6#Z@k8F6<m8{YH#9vS2%9*X0&a)Qu%cs57Xer@pb",
"149t2": "v4q1Q0zSVf0+8nx1.4Xj33D:o4hGGU6snND7!#1V9wex(9^P4)b*4GOej7{kgtT}BhSl*2",
"811t2": "v4o$f2ak.O30vS03/ILx4(B(f5FL&&6r&g&caUX>gRbHgh*du>",
"731t2": "yzTs%0r&OZ2GWym7T<)D",
"177t2": "yzX@-15PgV27l=O5jmBXb-%}e",
"556t2": "yL{sc0qV.e0@h#}1-BT}32f<U4Utl15rNT%7dn#28wsabacE=<bDJehb@l$@e1!PvgQ68&", 
"291t2": "yM0Yo0@SA$3uDXC6QxVPbloj{f0)p+gDmy{jtn]&k6f0I",
"139t2": "yX6n}1gY1%a&bSZ",
"998t2": "yX9::0nW^l2F7gr3LA(A80S0z8ShL(bKcZ{ey(z5g7=oV",
"391t2": "y+XrD3+WqS8*9krax{4)b?jp0dUPCkfKlCWg[uXEjK<Ai",
"670t2": "y+V5n6rSff8myE-fYt7vjEl3F",
"913t2": "FPxCe0Gvr64UB-Y",
"425t2": "FPf7}19>ac4Ej:77GRGK8.SpVaU+fkcIF}BdU/PNg.hSJi$RC)",
"305t2": "G[JK00o11)1-jE52pfTl3aHri3Kvdw3{UB%6b?JC6V.g08X%xB9Gqg$b10LPbATv{dn.-Fhnq5@jOnWO",
"622t2": "G[La<0rSUJ4lUlo6djL98}=HBdOzO]hoN(e",
"194t2": "HfTX:0Hj5T5G8sf7@mJ$dS)Fi",
"833t2": "HfP#V0Ypej3bm0+5l6n26C}Lz7nz!79RyLfaVQ/Eb/}x<dz)R{fnqemf!29sjcY^C",
"903t3": "j-W$#24m&>5N=rF6A]iG7v-h]895TX8]S[^aO^tMcOW9RfI:M$jF$z&kSnVy",
"668t3": "j-?Td5Tr#p8D!gZeibEzg/l9DhsS^UlnnDu",
"279t3": "j>Qdv21Y4[2#Is}7e2IH95wUUcFH0=f>SpaiC*&Akd%he",
"843t3": "j>N4f1dYQZ362sP4NRdL8!Qwlblxc>b[.ItjRx]ak6xe)lzsCW",
"731t3": "yAn[30Hi}B5UQ<AfxAfl",
"177t3": "yAu=H0Hi$C7PwN1bv8=}",
"556t3": "yMR#f16L(X2n)+W3FbW-4ycGw5J{:r7cJC:7LAWe8w0<.9m#B.betS=dF>QBe9[*sf-9OAgdTk9hiG>#hYT([kF/Vql5U^z",
"291t3": "yM.<u2H%eJ3zOsQ6&=9Gc@zJRgvWp6h]dJTi*&jJ",
"139t3": "yX[6-0cFpE3*!u57y.z8ddYC2",
"998t3": "yXY&L0uzwR2yM{Z3AT/p7d/o(9%A?}a@@2cdzWGsd-<s?ff7!7gw9B.g.z&lj-7f0",
"391t3": "y=ysx1c0No2#.i&5cS.d8Ejwr9Tr^cfNU+Jkfi^8",
"670t3": "y=vCC1yqpV6=4!<9&HjKcosFo",
"913t3": "FQcHh0I6TJ6R=)Ycq2U5emFOV",
"425t3": "FP>2^0NRau3ayK+66f}n9S)F*aa>=mbx1>zc?O}VedTpveIO&Se-OF=",
"305t3": "G]j>I0pZgq2/Nuw7{s>UaZ36gdub&keSP(i",
"622t3": "G]jTd0iVyB0>eT(4>Ow*6LOQ3ab7/0bc&Gce$%#3f^O::h}m)d",
"194t3": "HgrK82%/mDb^/D]",
"833t3": "Hgn&50!*+{2ZW9oa3hjc",
"149t3": "^KD3/2gTqF7ekVK8o%oTc(aHhhqg4*kh2#p",
"811t3": "^JYT/0X<[R3rW}.5}7zJ6E1qi8A5Kubuuv-e^0)Bhm>]^jnw<R",}

# Feature Names
TXT_HEADERS = ['start_time','end_time','is_question','is_pause','curr_sentence_length','speech_rate','is_edit_word','is_reparandum','is_interregnum','is_repair']
AUD_HEADERS = ['pcm_RMSenergy_sma', 'pcm_fftMag_mfcc_sma[1]', 'pcm_fftMag_mfcc_sma[2]', 'pcm_fftMag_mfcc_sma[3]', 'pcm_fftMag_mfcc_sma[4]', 'pcm_fftMag_mfcc_sma[5]', 'pcm_fftMag_mfcc_sma[6]', 'pcm_fftMag_mfcc_sma[7]', 'pcm_fftMag_mfcc_sma[8]', 'pcm_fftMag_mfcc_sma[9]', 'pcm_fftMag_mfcc_sma[10]', 'pcm_fftMag_mfcc_sma[11]', 'pcm_fftMag_mfcc_sma[12]', 'pcm_zcr_sma', 'voiceProb_sma', 'F0_sma', 'pcm_RMSenergy_sma_de', 'pcm_fftMag_mfcc_sma_de[1]', 'pcm_fftMag_mfcc_sma_de[2]', 'pcm_fftMag_mfcc_sma_de[3]', 'pcm_fftMag_mfcc_sma_de[4]', 'pcm_fftMag_mfcc_sma_de[5]', 'pcm_fftMag_mfcc_sma_de[6]', 'pcm_fftMag_mfcc_sma_de[7]', 'pcm_fftMag_mfcc_sma_de[8]', 'pcm_fftMag_mfcc_sma_de[9]', 'pcm_fftMag_mfcc_sma_de[10]', 'pcm_fftMag_mfcc_sma_de[11]', 'pcm_fftMag_mfcc_sma_de[12]', 'pcm_zcr_sma_de', 'voiceProb_sma_de', 'F0_sma_de']
VID_HEADERS = ['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r','AU01_c','AU02_c','AU04_c','AU05_c','AU06_c','AU07_c','AU09_c','AU10_c','AU12_c','AU14_c','AU15_c','AU17_c','AU20_c','AU23_c','AU25_c','AU26_c','AU28_c','AU45_c']

########################################################################################
# Pre-Process & Initialization

def normalize(xs):
    '''
    Normalize columns of xs to between 0 and 1 SEPARATELY
    '''
    return (xs - np.min(xs,axis=0))/(np.max(xs,axis=0)-np.min(xs,axis=0))

def make_transmat(labels,lengths=None):
    '''
    Make trans_mat for HMM init
    Input:
    labels: (n_labels,) array (dtype=int). confusion labels by frame 
    (could be cat of multiple subject-tasks)
    lengths: list of int. lengths of the subject-tasks in frames
    (for splitting the cat)
    Output:
    start_probs: (4,) array. proportions of confusion labels 0-4 in labels
    trans_mat: (4,4) array. normalized transition matrix between labels in labels
    '''
    # Transition Frequencies: transition a-->b encoded in 4bits aabb
    # Bincount 0-16 reshaped into 4x4 matrix
    get_tf = lambda l: np.bincount(4*l[:-1]+l[1:],minlength=16).reshape((4,4))
    # Start indices of each of the subject-tasks (inferred from lengths)
    _s = [sum(lengths[:i]) for i in range(len(lengths))]
    # Get transition frequencies for each subject-task and sum
    _tf = np.sum(np.dstack([get_tf(a.astype(int)) for a in np.split(labels,_s)]),axis=2)
    # Normalize each so row(s) sum to 1
    return _tf/np.linalg.norm(_tf,axis=1,ord=1)[:,None]

def fuck_the_nans(arr,amt):
    '''
    FUCK THE NANS AAAAAAAAAAAAAAAAAAAAA
    (Actually fights nans with nans lmao it adds nans to remove them)
    Input:
    arr: np array. taken to be 1 subject-task, shape=(n_frames,n_features)
    amt: int. number of rows to offset in EACH direction to get averages
    AMOUNT BY WHICH TO FUCK THE NANS
    Output:
    np array. shape=(n_frames,n_features) nans averaged out

    Probably make amt 2-3 times number of nan rows on end of input array
    '''
    # Array of nans : shape (offset, n_features)
    _n = np.empty( (amt, arr.shape[1]) ) * np.nan
    # Stack nans above and below array
    _p = np.vstack( (_n,arr,_n) )
    # Stack copies of nan-padded array, each offset by 1 (original in middle)
    # Cut off ends so middle layer is original (unpadded) array
    _t = np.dstack( [_p[i:_p.shape[0] + i - 2*amt] for i in range(2*amt + 1)] )
    # Average across the layers, ignoring nans
    # This smooths each entry as an average of its {amt} neighbors on either side
    # Near the ends, this is unidirectional, with end rows smoothed by {amt} neighbors
    _s = np.nansum(_t, axis=2) / np.sum(_t==_t, axis=2)
    # Replace all nans in original array with averaged values from smooth array _s
    return np.nansum(np.dstack( (arr, _s*(arr!=arr)) ), axis=2)

def fuck_the_nans_harder(arr):
    '''
    When u just gotta fucking slam those nans into oblivion
    Input:
    arr: np array (n_frames,n_features) array with nans
    Output:
    arr with all rows containing a nan gone
    '''
    return arr[~np.isnan(arr).any(axis=1),:]

def gen_aligned_labels(arr,code,framerate=0.04):
    '''
    Generate a label vector aligned to array arr
    Account for missing data
    Inputs:
    arr: np array (n_frames,n_features) a participant-task
    code: str. annotation code in Z85
    framerate: float. number of seconds per frame
    Output:
    np array (n_frames,) of int. labels aligned to arr
    '''
    # Convert deciseconds to frame number, assume frame = 0.04 sec
    to_frame = lambda x: [(math.ceil(t/(10*framerate)),v) for (t,v) in x]
    # List frames beginning with (0,0)
    # Set -> list removes any trailing (0,0) left over from decoding
    l = sorted(list(set([(0,0)]+to_frame(b.time16_decode(code)[1:]))))
    # For each frame_time in the array, take the last label whose time is before
    return np.array([max([s for s in l if t>=s[0]])[1] for t in arr[:,0]],dtype=np.int64)

########################################################################################
# Post-Process & Analysis

def feature_likelihoods(x,means_,covars_):
    '''
    Compute likelihood over time of the all the features with each confusion state
    Input:
    x: np array (n_frames,n_features) a participant-task data array
    means_: np array (n_components,n_features) means from hmm
    covars_: np array (n_components,n_features,n_features) variances from hmm
    Output:
    np array (n_frames,n_features,n_components)
    likelihood of each feature in each frame given each confusion state
    '''
    # Take diagonals and stack means or covars to make (n_components,n_features)
    diag = lambda l: np.vstack([np.diag(a)[None,:] for a in l])
    # Std_dev is sqrt of variance
    sigma = np.sqrt(diag(covars_)).T[None,:,:]
    mu = means_.T[None,:,:]
    # Use a Gaussian
    return np.exp(-0.5*((x[:,:,None]-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi))

def avg_feature_importance(xs,means_,covars_):
    '''
    Compute overall feature importance via averaging likelihoods across all participants
    Input:
    xs: list of np array (n_frames,n_features) all participant-tasks to sample
    means_: np array (n_components,n_features) means from hmm
    covars_: np array (n_components,n_features,n_features) variances from hmm
    Output:
    means: np array (n_features,n_components) overall mean likelihood for each state
    std_devs: np array (n_features,n_components) overall std_dev likelihood for each state
    '''
    # Stack all arrays, compute feature likelihoods
    _f = feature_likelihoods(np.vstack(xs))
    return np.mean(_f,axis=0),np.std(_f,axis=0)

def get_features(arr,feat_names,modes='tavl'):
    '''
    Return participant-task array col-sliced by feature name
    Input:
    arr: np array (n_frames,n_features)
    feat_names: list of str. names of features in TXT_, VID_ -- or AUD_FEATURES
    modes: which modalities are included in dataset
    Output:
    np array (n_frames,len(feat_names)) sliced to columns IN ORDER SPECIFIED in FEAT_NAMES
    '''
    features = TXT_FEATURES * ('t' in modes) + AUD_FEATURES * ('a' in modes)\
            + VID_FEATURES * ('v' in modes) + ['labels'] * ('l' in modes)
    return arr[:,[features.index(f) for f in feat_names]]

########################################################################################
# Visualization