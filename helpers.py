from IPython.display import display
from matplotlib import pyplot as plt
import numpy as np
from statsmodels.graphics import tsaplots
from scipy.signal import butter, filtfilt
import scipy
import statsmodels.api as sm
import heartpy as hp


def get_DCT(signal, No_sampels):
    AC = sm.tsa.acf(signal, nlags=No_sampels)
    sig = AC[0:300]
    DCT = scipy.fftpack.dct(sig, type=2)
    return DCT


def butter_band_pass_filter(input_signal, low_cutoff, high_cutoff, sampling_rate, order):
    nyqRate = 0.5 * sampling_rate
    low = low_cutoff / nyqRate
    high = high_cutoff / nyqRate
    numerator, denominator = butter(order, [low, high], btype='band', output='ba', analog=False, fs=None)
    input_signal = input_signal.reshape((-1,))
    filtered = filtfilt(numerator, denominator, input_signal)
    return filtered


def data_preprocessing(signals, low_cutoff, high_cutoff, sampling_rate, order):
    preprocessed = []
    for signal in signals:
        # band pass filter
        preSig = butter_band_pass_filter(signal, low_cutoff, high_cutoff, sampling_rate, order)

        # preprocessed using smooth_signal
        preSig = hp.smooth_signal(preSig, sampling_rate)

        # # preprocessed using enhance_ecg_peaks
        # preSig = hp.enhance_ecg_peaks(preSig, sampling_rate,iterations=5)

        preprocessed.append(preSig)
    return preprocessed


def extract_features(signals, No_of_sampels):
    feature_extracted = []
    for signal in signals:
        feature_extracted.append(get_DCT(signal, No_of_sampels))
    return feature_extracted
