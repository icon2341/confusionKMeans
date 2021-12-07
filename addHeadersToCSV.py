import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import sklearn
from joblib.numpy_pickle_utils import xrange
from kneed import KneeLocator
from plotly.validators.box.marker import SymbolValidator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly as pgo
import plotly.io as pio
import glob
from random import seed
import plotly.graph_objects as go
from random import random
from old.code import base85 as encode
import csv



# Feature Names for conversion
TXT_HEADERS = ['start_time','end_time','is_question','is_pause','curr_sentence_length','speech_rate','is_edit_word','is_reparandum','is_interregnum','is_repair']
AUD_HEADERS = ['pcm_RMSenergy_sma', 'pcm_fftMag_mfcc_sma[1]', 'pcm_fftMag_mfcc_sma[2]', 'pcm_fftMag_mfcc_sma[3]', 'pcm_fftMag_mfcc_sma[4]', 'pcm_fftMag_mfcc_sma[5]', 'pcm_fftMag_mfcc_sma[6]', 'pcm_fftMag_mfcc_sma[7]', 'pcm_fftMag_mfcc_sma[8]', 'pcm_fftMag_mfcc_sma[9]', 'pcm_fftMag_mfcc_sma[10]', 'pcm_fftMag_mfcc_sma[11]', 'pcm_fftMag_mfcc_sma[12]', 'pcm_zcr_sma', 'voiceProb_sma', 'F0_sma', 'pcm_RMSenergy_sma_de', 'pcm_fftMag_mfcc_sma_de[1]', 'pcm_fftMag_mfcc_sma_de[2]', 'pcm_fftMag_mfcc_sma_de[3]', 'pcm_fftMag_mfcc_sma_de[4]', 'pcm_fftMag_mfcc_sma_de[5]', 'pcm_fftMag_mfcc_sma_de[6]', 'pcm_fftMag_mfcc_sma_de[7]', 'pcm_fftMag_mfcc_sma_de[8]', 'pcm_fftMag_mfcc_sma_de[9]', 'pcm_fftMag_mfcc_sma_de[10]', 'pcm_fftMag_mfcc_sma_de[11]', 'pcm_fftMag_mfcc_sma_de[12]', 'pcm_zcr_sma_de', 'voiceProb_sma_de', 'F0_sma_de']
VID_HEADERS = ['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r','AU01_c','AU02_c','AU04_c','AU05_c','AU06_c','AU07_c','AU09_c','AU10_c','AU12_c','AU14_c','AU15_c','AU17_c','AU20_c','AU23_c','AU25_c','AU26_c','AU28_c','AU45_c']
ALL_HEADERS = TXT_HEADERS + AUD_HEADERS + VID_HEADERS

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # buildHeaderLine
    headerLine = ""
    for header in ALL_HEADERS:
        headerLine += header
        headerLine += ", "

    print(headerLine)

    with open("data/output.csv", "r") as f:
        contents = f.readlines()

    contents.insert(0, headerLine)


    with open("data/output.csv", "w") as f:
        contents2 = "".join(contents)
        print(contents2)
        f.write(contents2)









