import biosppy
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


def winIntegration(signal, samplingRate):
    window = int(0.079 * int(samplingRate))
    integWindow = np.ones(window) / window
    integratedSignal = np.convolve(signal, integWindow, mode='same')
    return integratedSignal


def QRS_Features(signal):
    lowCut = 1.0
    highCut = 40.0
    samplingRate = 500.0

    signal = (signal - np.mean(signal)) / np.std(signal)
    # 1- Band Pass Filter
    lowHighPass_Signal = butter_band_pass_filter(signal, lowCut, highCut, samplingRate, 1)
    # 2- Differentation
    differentation_Signal = np.gradient(lowHighPass_Signal)
    # 3- Squaring
    squaring_Signal = np.square(differentation_Signal)
    # 4- Moving Window integration
    intergratedSingal = winIntegration(squaring_Signal, samplingRate)
    # 5- Trsehsold
    R = scipy.signal.find_peaks(intergratedSingal, height=np.mean(intergratedSingal), distance=0.2 * samplingRate)[0]

    def R_correction(signal, peaks):

        peaks_corrected_list = []
        for index in range(peaks.shape[0]):
            peak = peaks[index]  # Peak
            if peak - 1 < 0:
                break
            if signal[peak] < signal[peak - 1]:
                while signal[peak] < signal[peak - 1]:
                    peak -= 1
                    if peak < 0:
                        break
            elif signal[peak] < signal[peak + 1]:
                while signal[peak] < signal[peak + 1]:
                    peak += 1
                    if peak < 0:
                        break
            peaks_corrected_list.append(peak)
        return np.array(peaks_corrected_list)

    R = R_correction(lowHighPass_Signal, R)
    # listY=[]
    # for r in R:
    #     listY.append(lowHighPass_Signal[r])
    # listY=sorted(listY,reverse=True)
    # listY=listY[:3]
    # correctR=[]
    # sig=lowHighPass_Signal.tolist()
    # for y in listY:
    #     correctR.append(sig.index(y))
    # R=correctR


    # qrs_result = ecg.christov_segmenter(signal=lowHighPass_Signal, sampling_rate=samplingRate)

    # find q and s peaks by R peaks
    q_r_s_peaks,Features = extract_Q_S_peaks(lowHighPass_Signal, R, 50)


    # Plotting
    time = np.arange(len(lowHighPass_Signal)) / samplingRate
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title("11 Points")
    plt.plot(time, lowHighPass_Signal, 'b')
    plt.plot(time[q_r_s_peaks], lowHighPass_Signal[q_r_s_peaks], 'ro')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.subplot(122)
    plt.title("R Peak")
    plt.plot(time, lowHighPass_Signal, 'b', label='ECG Signal')
    plt.plot(time[R], lowHighPass_Signal[R], 'ro', label='R Peaks')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

    RR_Features = []
    for i in range(len(R) - 1):
        RR_Features.append(R[i + 1] - R[i])
    RR_Features = np.array(RR_Features)


    return Features


def extract_features(signals, No_of_sampels, type):
    feature_extracted = []
    if type == 1:
        for signal in signals:
            feature_extracted.append(get_DCT(signal, No_of_sampels))
    elif type == 2:
        for signal in signals:
            feature_extracted.append(QRS_Features(signal))
    return feature_extracted


def extract_Q_S_peaks(signal, R, range):
    q_r_s_peaks = []
    signal = signal.tolist()
    Features=[]
    p = q = s = t = p_on = t_on = p_off = t_off = qrs_on = qrs_off = -1
    for r in R:
        if ((r - range) > 0):
            q = signal.index(min(signal[(r - range):r]))
            q_r_s_peaks.append(q)
        if ((r - 120) > 0):
            p = signal.index(max(signal[(r - 120):r - range]))
            q_r_s_peaks.append(p)

            # get p on
            if ((r - 160) > 0):
                p_on = get_point_with_max_area(signal, p - 40, p)
                q_r_s_peaks.append(p_on)

        if ((r + range) < len(signal)):
            s = signal.index(min(signal[r:(r + range)]))
            q_r_s_peaks.append(s)
        if ((r + 120) < len(signal)):
            t = signal.index(max(signal[r + range:(r + 120)]))

            # get t_off
            if ((r + 200) < len(signal)):
                t_off = get_point_with_max_area(signal, t , t+ 100)
                q_r_s_peaks.append(t_off)
            q_r_s_peaks.append(t)

        # get qrs_on and p_off
        if(p!=-1 and q!=-1):
            qrs_on=get_point_with_max_area(signal, p, q)
            p_off=get_point_with_max_area(signal, qrs_on, q)
            q_r_s_peaks.append(qrs_on)
            q_r_s_peaks.append(p_off)
        # get qrs_off and t_on
        if (s != -1 and t != -1):
            qrs_off = get_point_with_max_area(signal, s, t)
            t_on = get_point_with_max_area(signal, qrs_off, t)
            q_r_s_peaks.append(qrs_off)
            q_r_s_peaks.append(t_on)

        q_r_s_peaks.append(r)
        if(p!=-1 and q!=-1 and  s!=-1 and  t !=-1 and  p_on !=-1 and  t_on !=-1 and  p_off !=-1 and  t_off !=-1 and  qrs_on !=-1 and  qrs_off!=-1):
            Features.append(r-q) #RQ
            Features.append(s-r) #SR
            Features.append(p_off-p_on) #P-INTERVAL
            Features.append(t_off-t_on) #T-INTERVAL
            Features.append(q-p) #QP
            Features.append(s-q) #SQ
            Features.append(t-qrs_off) #TQRSOFF
            Features.append(r-p_off) #RPOFF
            Features.append(t_on-r) #TONR
            Features.append(r-p) #RP
            Features.append(t-r) #TR
            Features.append(s-p) #SP
            Features.append(t-q) #TQ
            Features.append(r-p_on) #RPON
            Features.append(t_off-r) #TOFFR
            Features.append(q-p_on) #QPONN
            Features.append(t_off-s) #TOFFS
            Features.append(qrs_on-p_off) #QRSONPOFF
            Features.append(t_on-qrs_off) #TONQRSOFF
            Features.append(t_off-p_off) #TOFFPOFF
            Features.append(t-p) #TP
            #21Features
    return q_r_s_peaks,Features


def get_point_with_max_area(signal, p1, p2):
    max_area = -1
    max_point = -1
    for i in range(p2 - p1 - 1):
        p3 = i + p1
        area = abs(get_area(signal, p1, p2, p3))
        if (area > max_area):
            max_area = area
            max_point = p3
    return max_point


def get_area(signal, p1, p2, p3):
    x = [p1, p2, p3]
    y = []
    y.append(signal[int(p1)])
    y.append(signal[int(p2)])
    y.append(signal[int(p3)])
    area = 0.5 * (x[0] * (y[1] - y[2]) + x[1] * (y[2] - y[0]) + x[2]
                  * (y[0] - y[1]))
    return int(area * 10000)
