from IPython.display import display
from matplotlib import pyplot as plt
import numpy as np
from statsmodels.graphics import tsaplots
import wfdb
import glob


def read_data(path, no_of_samples):
    persons = glob.glob(path)
    persons = [person for person in persons if person.__contains__("Person")]
    signals = []
    labels = []
    for person in persons:
        recs = glob.glob(person + "/*")
        NoOfRec = int(len(recs)/3)
        for i in range(NoOfRec):
            recPath=person + './rec_' + str((i + 1))
            signal, fields = wfdb.rdsamp(recPath, sampfrom=0, sampto=no_of_samples,
                                         channels=[0, 1])
            signal = np.array(signal)
            un_filtered, filtered = np.split(signal, 2, axis=1)
            signals.append(un_filtered)
            absPath=person.split('\\')
            labels.append(absPath[-1])
    return signals ,labels

# read_data("./Data sets/*",1500)

# record = wfdb.rdrecord('./ecg-id-database-1.0.0\Person_01/rec_1', sampfrom=0, sampto=1000, channels=[0, 1])
# display(record.__dict__)
# # wfdb.plot_wfdb(record=record,title="the record")
#
#
# signal, fields = wfdb.rdsamp('./ecg-id-database-1.0.0\Person_02/rec_2', sampfrom=0, sampto=1500, channels=[0, 1])
# seg1 = np.array(signal)
# un_filtered, filtered = np.split(seg1, 2, axis=1)
#
# butter_filtered = butter_band_pass_filter(un_filtered, low_cutoff=1.0, high_cutoff=40.0, sampling_rate=500, order=4)
# DCT=get_DCT(filtered,1500)
# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# plt.plot(filtered)
# plt.subplot(122)
# plt.plot(DCT)
# plt.show()
# print(signal)
