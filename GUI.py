import glob
import math
import tkinter as tk
from tkinter import *
from tkinter import ttk

import wfdb

import helpers
import main
from main import *

def testData(path):
    persons = glob.glob(path)
    labels=[]
    persons = [person for person in persons if person.__contains__("Person")]
    for person in persons:
        absPath = person.split('\\')
        labels.append(absPath[-1])
    print(labels)
    return labels

def change_label(label):
    color=""
    if label=="Authorized..!":
        color = "green"
    elif label=="Not Authorized..!":
        color="red"
    accPerc.config(text=label, fg=color)

testPredict=-1
def runTest():
    if signalCombo.get()!='' and featuresCombo.get()!='':
        type=-1
        personID = signalCombo.get()
        signalTestPath = "./DataGUITest/" + personID + './rec_1'
        signal, fields = wfdb.rdsamp(signalTestPath, sampfrom=0, sampto=1500,
                                     channels=[0, 1])
        signal = np.array(signal)
        un_filtered, filtered = np.split(signal, 2, axis=1)
        Features=[]
        if featuresCombo.get()=='Fiducial':
            type=2
            QRS=helpers.QRS_Features(un_filtered)
            Features =QRS[0]
            Features=Features[0:42]
            helpers.plotSignal(QRSPOINTS=QRS[1],signal=QRS[2],type=type)
        elif featuresCombo.get()=='non-fiducial':
            type=1
            preProcSignal=helpers.data_preprocessing([un_filtered],1.0,40.0,500,4)
            Features = helpers.get_DCT(preProcSignal[0], 1500)
            helpers.plotSignal(signal=un_filtered,preProc=preProcSignal[0],DCT=Features,type=type)
        testPredict=main.testFromMain(type,np.array(Features))
        change_label(testPredict)
    else:
        print("YOU MUST CHOOSE FROM COMBOBOX........!")

mainWindow = tk.Tk()
mainWindow.title("HCI - ECG Based authentication Interface")
mainWindow.geometry("650x300")

mainTitle=Label(mainWindow,text="ECG Based authentication Interface - SC 14",font=("Arial", 19))
mainTitle.place(x=80,y=50)

signalLabel=Label(mainWindow,text="Signal of Person No:")
signalLabel.place(x=170,y=130)

signalCombo=ttk.Combobox(mainWindow)
signalCombo['values']=testData("./DataGUITest/*")   #Edit List
signalCombo.place(x=320, y=130)

featuresLabel=Label(mainWindow,text="Type of Features:")
featuresLabel.place(x=40,y=200)

featuresCombo=ttk.Combobox(mainWindow)
featuresCombo['values']=['Fiducial','non-fiducial']   #Edit List
featuresCombo.place(x=170, y=200)

# Button
runButton = Button(mainWindow, text="Test Authentication",command=runTest) #command=dataFromGUI
runButton.place(x=420, y=195,width=150)


accPerc=Label(mainWindow,text="PREDICT TEXT..!",font=("Arial", 16), fg= "green")
accPerc.place(x=250,y=250)

mainWindow.mainloop()