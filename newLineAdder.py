# This is a sample Python script.
#File Name: ConversionScript.py
#File Desc: The script converts the old subject data files with time aligned video/text/audio features into a better
#formatted csv since nunpy does not natively output into reasonably formatted csv
#@author Ashar Rahman


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import csv
import os


# Feature Names for conversion
TXT_HEADERS = ['start_time','end_time','is_question','is_pause','curr_sentence_length','speech_rate','is_edit_word','is_reparandum','is_interregnum','is_repair']
AUD_HEADERS = ['pcm_RMSenergy_sma', 'pcm_fftMag_mfcc_sma[1]', 'pcm_fftMag_mfcc_sma[2]', 'pcm_fftMag_mfcc_sma[3]', 'pcm_fftMag_mfcc_sma[4]', 'pcm_fftMag_mfcc_sma[5]', 'pcm_fftMag_mfcc_sma[6]', 'pcm_fftMag_mfcc_sma[7]', 'pcm_fftMag_mfcc_sma[8]', 'pcm_fftMag_mfcc_sma[9]', 'pcm_fftMag_mfcc_sma[10]', 'pcm_fftMag_mfcc_sma[11]', 'pcm_fftMag_mfcc_sma[12]', 'pcm_zcr_sma', 'voiceProb_sma', 'F0_sma', 'pcm_RMSenergy_sma_de', 'pcm_fftMag_mfcc_sma_de[1]', 'pcm_fftMag_mfcc_sma_de[2]', 'pcm_fftMag_mfcc_sma_de[3]', 'pcm_fftMag_mfcc_sma_de[4]', 'pcm_fftMag_mfcc_sma_de[5]', 'pcm_fftMag_mfcc_sma_de[6]', 'pcm_fftMag_mfcc_sma_de[7]', 'pcm_fftMag_mfcc_sma_de[8]', 'pcm_fftMag_mfcc_sma_de[9]', 'pcm_fftMag_mfcc_sma_de[10]', 'pcm_fftMag_mfcc_sma_de[11]', 'pcm_fftMag_mfcc_sma_de[12]', 'pcm_zcr_sma_de', 'voiceProb_sma_de', 'F0_sma_de']
VID_HEADERS = ['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r','AU01_c','AU02_c','AU04_c','AU05_c','AU06_c','AU07_c','AU09_c','AU10_c','AU12_c','AU14_c','AU15_c','AU17_c','AU20_c','AU23_c','AU25_c','AU26_c','AU28_c','AU45_c']
ALL_HEADERS = TXT_HEADERS + AUD_HEADERS + VID_HEADERS

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #loop through each file in in the directory to reformat
    for file in os.listdir("data"):
        filename = os.fsdecode(file)
        print("\n\n" + filename + "\n\n")
        #open the file
        f = open("data/" + filename,"a")

        f.write('\n')


        f.close()
