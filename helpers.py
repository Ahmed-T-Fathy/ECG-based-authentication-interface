from IPython.display import display
from matplotlib import pyplot as plt
import numpy as np
from statsmodels.graphics import tsaplots
from scipy.signal import butter, filtfilt
import scipy
import statsmodels.api as sm
import heartpy as hp
import biosppy.signals.ecg as ecg


def get_DCT(signal, No_sampels):
    AC = sm.tsa.acf(signal, nlags=No_sampels)
    sig = AC[0:300]
    DCT = scipy.fftpack.dct(sig, type=2)
    return DCT[0:100]


def butter_band_pass_filter(input_signal, low_cutoff, high_cutoff, sampling_rate, order):
    nyqRate = 0.5 * sampling_rate
    low = low_cutoff / nyqRate
    high = high_cutoff / nyqRate
    numerator, denominator = butter(order, [low, high], btype='band', output='ba', analog=False, fs=None)
    input_signal = input_signal.reshape((-1,))
    filtered = filtfilt(numerator, denominator, input_signal)
    return filtered

# def butter_lowpass_filter(input_signal, low_cutoff, sampling_rate, order):
#     nyqRate = 0.5 * sampling_rate
#     normal_cutoff = low_cutoff / nyqRate
#     # Get the filter coefficients
#     numerator, denominator = butter(order, normal_cutoff, btype='low', analog=False)
#     y = filtfilt(numerator, denominator, input_signal)
#     return y
#
# def butter_highpass_filter(input_signal, high_cutoff, sampling_rate, order):
#     nyqRate = 0.5 * sampling_rate
#     normal_cutoff = high_cutoff / nyqRate
#     # Get the filter coefficients
#     numerator, denominator = butter(order, normal_cutoff, btype='high', analog=False)
#     y = filtfilt(numerator, denominator, input_signal)
#     return y

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

def winIntegration(signal,samplingRate):
    window=int(0.08*int(samplingRate))
    integWindow=np.ones(window)/window
    signal=signal.reshape((-1,))
    integratedSignal=np.convolve(signal,integWindow,mode='same')
    return integratedSignal

def QRS_Features(signal):
    lowCut=1.0
    highCut=40.0
    samplingRate=500.0

    #1- Band Pass Filter
    lowHighPass_Signal=butter_band_pass_filter(signal,lowCut,highCut,samplingRate,6)
    #2- Differentation
    differentation_Signal=np.diff(lowHighPass_Signal)
    #3- Squaring
    squaring_Signal=differentation_Signal**2
    #4- Moving Window integration
    intergratedSingal=winIntegration(squaring_Signal, samplingRate)
    #5- Trsehsold
    threshold=0.3*np.max(intergratedSingal)
    listThreshold=np.where(intergratedSingal>threshold)[0]

    R=[]
    windowPeriod=int(0.2*samplingRate)
    for i in listThreshold:
        if(intergratedSingal[i]>intergratedSingal[i-1] and intergratedSingal[i]>intergratedSingal[i+1]):
            if(len(R)==0 or (i-R[-1]>windowPeriod)):
                R.append(i+1)

    qrs_result = ecg.christov_segmenter(signal=lowHighPass_Signal, sampling_rate=samplingRate)

    #Plotting
    # qrs_result=np.array(qrs_result).reshape(-1,)
    # time=np.arange(len(lowHighPass_Signal))/samplingRate
    # plt.figure(figsize=(12, 6))
    # plt.subplot(121)
    # plt.title("QRS Algorithm - Khaled")
    # plt.plot(time, lowHighPass_Signal, 'b', label='ECG Signal')
    # plt.plot(time[R], lowHighPass_Signal[R], 'ro', label='R Peaks')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.grid(True)
    # plt.subplot(122)
    # plt.title("QRS Algorithm - Library")
    # plt.plot(time, lowHighPass_Signal, 'b', label='ECG Signal')
    # plt.plot(time[qrs_result], lowHighPass_Signal[qrs_result], 'ro', label='R Peaks')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    RR_Features=[]
    for i in range(len(qrs_result)-1):
        RR_Features.append(qrs_result[i+1]-qrs_result[i])
    RR_Features=np.array(RR_Features)
    return RR_Features

def extract_features(signals, No_of_sampels,type):
    feature_extracted = []
    if type==1:
        for signal in signals:
            feature_extracted.append(get_DCT(signal, No_of_sampels))
    elif type==2:
        for signal in signals:
            feature_extracted.append(QRS_Features(signal))
    return feature_extracted
