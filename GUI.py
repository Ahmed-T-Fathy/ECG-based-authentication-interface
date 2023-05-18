import glob
import math
import tkinter as tk
from tkinter import *
from tkinter import ttk

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


mainWindow = tk.Tk()
mainWindow.title("HCI - ECG Based authentication Interface")
mainWindow.geometry("680x350")

mainTitle=Label(mainWindow,text="ECG Based authentication Interface - SC 14",font=("Arial", 19))
mainTitle.place(x=20,y=50)

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
runButton = Button(mainWindow, text="Test Authentication",) #command=dataFromGUI
runButton.place(x=420, y=195,width=150)

accLabel=Label(mainWindow,text="Accuracy: ",font=("Arial", 16))
accLabel.place(x=40,y=250)

accPerc=Label(mainWindow,text=f"{main.accuracy*100:.2f}%",font=("Arial", 16))
accPerc.place(x=170,y=250)

accPerc=Label(mainWindow,text="Authenticated..!",font=("Arial", 16), fg= "green")
accPerc.place(x=420,y=250)

mainWindow.mainloop()