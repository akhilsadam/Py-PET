import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy.optimize import curve_fit
from scipy import stats

#The required names for the photon data columns are : EventID, CopyNumber, Time

def generateTextY(num,max):
    y = [max - 0.04*(i+1)*max for i in range(0,num+1)]
    return y

class Sim_Data:
    def __init__(self, PhotonData, GammaData, ElectronData):
        self.PhotonData = PhotonData
        self.GammaData = GammaData
        self.ElectronData = ElectronData

    def ToSingleChannel(self, allDets=False, DR=False, det_sep = 0):
        self.PhotonData = self.PhotonData.sort_values(by="Time")
        events = self.PhotonData.EventID.unique()
        groups = self.PhotonData.groupby("EventID")
        Det_Data = pd.DataFrame(columns=["EventID", "CopyNumber", "Counts", "PhotTime"])
        for group in groups:
            det = group[1].CopyNumber.value_counts().reset_index()
            det.columns=["Copynumber","Counts"]
            for i in range(len(det.Counts)):
                Time_Data = group[1][group[1]["CopyNumber"] == det.Copynumber[i]].reset_index()
                Time_Data = Time_Data.sort_values(by="Time").reset_index()
                if Time_Data.Time.count() >= 5:
                    nth = Time_Data.Time[4]
                else:
                    nth = 0
                Det_Data = Det_Data.append({"EventID":group[0], "CopyNumber":det.Copynumber[i], "Counts":det.Counts[i], "PhotTime":nth}, ignore_index=True)
        if DR == False:
            if allDets == True:
                return Det_Data
            elif allDets == False:
                return Det_Data.groupby("EventID").first().reset_index()
        elif DR == True and det_sep != 0:
            if allDets == True:
                return Det_Data
            elif allDets == False:
                Det_diff = pd.DataFrame(columns=["EventID","SC_Det_diff", "SC_Det_Total", "CopyNumber", "SC_Time_In", "SC_Time_Out"])
                events = Det_Data.EventID.unique()
                for event in events:
                    test = Det_Data[Det_Data["EventID"] == event].reset_index()
                    SC = test.CopyNumber[0]
                    for i in range(len(test.CopyNumber)):
                        if (SC - test.CopyNumber[i] == det_sep) or (SC - test.CopyNumber[i] == -1*det_sep):
                            if (SC > test.CopyNumber[i]):
                                Det_diff = Det_diff.append({"EventID":event,"Det_diff":(test.Counts[0]-test.Counts[i]), "Det_Total":(test.Counts[0]+test.Counts[i]),"CopyNumber":SC, "Time_In":test.PhotTime[i], "Time_Out":test.PhotTime[0]}, ignore_index=True)
                            elif (SC < test.CopyNumber[i]):
                                Det_diff = Det_diff.append({"EventID":event,"Det_diff":(test.Counts[i]-test.Counts[0]), "Det_Total":(test.Counts[0]+test.Counts[i]),"CopyNumber":SC, "Time_In":test.PhotTime[0], "Time_Out":test.PhotTime[i]}, ignore_index=True)
                self.PhotonData = Det_diff
        elif DR == True and det_sep == 0:
            print("You must specify a nonzero detector seperation to calculate paired detectors.")
            return 0

    def ToMultiChannel(self, DR=False, det_sep=0):
        self.PhotonData = self.PhotonData.sort_values(by='Time')

        if DR == True and det_sep != 0:
            df_out1 = self.PhotonData[self.PhotonData["CopyNumber"] >= det_sep]
            df_in1 = self.PhotonData[self.PhotonData["CopyNumber"] < det_sep]#%1000 < 100

            df_out = df_out1.EventID.value_counts().reset_index()
            df_out.columns=["EventID", "Det_Out"]
            df_in = df_in1.EventID.value_counts().reset_index()
            df_in.columns=["EventID", "Det_In"]
            df_counts = pd.merge(df_out, df_in, on="EventID").reset_index()
            df_counts = df_counts.drop(columns=["index"])

            out_time = df_out1.groupby("EventID").Time.nth(4).reset_index()
            out_time.columns=["EventID", "Time_Out"]
            in_time = df_in1.groupby("EventID").Time.nth(4).reset_index()
            in_time.columns=["EventID", "Time_In"]
            time = pd.merge(out_time,in_time, on="EventID")
            DATA = pd.merge(df_counts, time, on="EventID")
            DATA["Det_Total"] = DATA["Det_Out"] + DATA["Det_In"]
            DATA["Det_Diff"] = DATA["Det_Out"] - DATA["Det_In"]
            self.PhotonData = DATA
        elif DR == True and det_sep == 0:
            print("You must specify a nonzero detector seperation to calculate paired detectors.")
            return 0
        elif DR == False:
            df = self.PhotonData.EventID.value_counts().reset_index()
            df.columns=["EventID", "Det_Total"]
            time = self.PhotonData.groupby("EventID").Time.nth(4).reset_index()
            time.columns=["EventID", "Time_Out"]
            DATA = pd.merge(df, time, on="EventID")
            self.PhotonData = DATA

    def ToCoincidence(self, display=False):
        self.PhotonData = self.PhotonData.sort_values(by="Time")
        events = self.PhotonData.EventID.unique()
        groups = self.PhotonData.groupby("EventID")
        Det_Data = pd.DataFrame(columns=["EventID", "CopyNumber", "Counts", "PhotTime"])
        for group in groups:
            det = group[1].CopyNumber.value_counts().reset_index()
            det.columns=["Copynumber","Counts"]
            for i in range(len(det.Counts)):
                Time_Data = group[1][group[1]["CopyNumber"] == det.Copynumber[i]].reset_index()

                Time_Data = Time_Data.sort_values(by="Time").reset_index()
                if Time_Data.Time.count() >= 5:
                    nth = Time_Data.Time[4]
                else:
                    nth = 0
                Det_Data = Det_Data.append({"EventID":group[0], "CopyNumber":det.Copynumber[i], "Counts":det.Counts[i], "PhotTime":nth}, ignore_index=True)

        Det_Data = Det_Data.groupby("EventID").first().reset_index()

        #Legacy code from expirimental ananlysis, used to convert number of detected photons to ratio w/ 511. This will be the energy metric used to compare expiriment and simulation
        fig = plt.figure()
        values,bins,presenters = plt.hist(Det_Data.Counts, bins = np.linspace(0,Det_Data.Counts.max()+100,80), fill=True)
        bin_centers = [(bins[q] + bins[q+1])/2 for q in range(0,len(bins)-1)]
        values = values.tolist()
        #photopeak finding algortihm. Potential area for improvment. At present only about 95% accurate
        for i in range(1, len(values)-4):
            if values[-1*i] > values[-1*(i+1)] and values[-1*i] > values[-1*(i+2)] and values[-1*i] > values[-1*(i+3)] and values[-1*(i+1)] != 0 and values[-1*i] >= 30:
                photopeak = values.index(values[-1*i],-1*i-1)
                break
        Ex_p = [0,1,1]
        fit_x = bin_centers[photopeak-7:photopeak+8]
        fit_y = values[photopeak-7:photopeak+8]

        if len(fit_y) != 0 and len(fit_x) != 0:
            try:
                Ex_p, Ex_co = curve_fit(gauss, fit_x, fit_y, p0=[values[photopeak], fit_x[7], 0.5*(bin_centers[photopeak+7] - bin_centers[photopeak-7])])
            except RuntimeError:
                print("Error - curve_fit failed")
        if display == True:
            x_f = np.linspace(0,Det_Data.Counts.max()+100,200)
            plt.plot(x_f, gauss(x_f, *Ex_p), color='k')
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()
        Det_Data["Counts"] = (Det_Data["Counts"] - Ex_p[1])/Ex_p[2]
        Compressed = pd.DataFrame(columns=["ChannelIDL", "ChargeL","TimeL", "ChannelIDR", "ChargeR", "TimeR"])
        for i in range(Det_Data.Counts.count() // 2):
            Compressed = Compressed.append({"ChannelIDL":Det_Data.iloc[2*i][1],"ChargeL":Det_Data.iloc[2*i][2],"TimeL":Det_Data.iloc[2*i][3],"ChannelIDR":Det_Data.iloc[2*i+1][1],"ChargeR":Det_Data.iloc[2*i+1][2],"TimeR":Det_Data.iloc[2*i+1][3]}, ignore_index=True)
        self.PhotonData = Compressed


    def SetPhotonData(self, NewData):
        self.PhotonData = NewData

    def SetGammaData(self, NewData):
        self.GammaData = NewData

    def SetElectronData(self, NewData):
        self.ElectronData = NewData

    def GetPhotonData(self, NewData):
        return self.PhotonData

    def GetGammaData(self, NewData):
        return self.GammaData

    def GetElectronData(self, NewData):
        return self.ElectronData

class Plot:
    @staticmethod
    def CTRHist(dataClass, xlim=[0,2], ylim=[0,100], Fit=fit):
        dataClass.PhotonData["Time_diff"] = dataClass.PhotonData["TimeL"] - dataClass.PhotonData["TimeR"]

        CTRs = data.CTR
        CTRs = CTRs.abs()
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111,xlim=(0,0.9), ylim=(0,300),xlabel="CTR in ns",ylabel="Counts",title="CTR of most active Channel Pairs \n ")

        low = 0
        high = low +1 #high = low * -1
        num_bins = 100
        bins = np.linspace(low,high,num_bins)
        x_p = np.linspace(low+((high-low)/(2*num_bins)),high-((high-low)/(2*num_bins)),400)
        T_x = 0.6
        T_y = generateTextY(10,300)

        ax1.hist(CTRs, bins=bins,color='b')
        ax1.text(T_x,T_y[0],"Data Stats:")
        ax1.text(T_x,T_y[1],"mean: "+format(CTRs.mean(),"7.4f"))
        ax1.text(T_x,T_y[2],"std: "+format(CTRs.std(),"7.4f"))
        ax1.text(T_x,T_y[3],"count: "+str(len(CTRs)))
        ax1.text(T_x,T_y[4],"Fit Paramters:")
        ax1.text(T_x,T_y[5],"mu: "+format(x1_p[1],"7.4f"))
        ax1.text(T_x,T_y[6],"sigma: "+format(x1_p[2],"7.4f"))
        ax1.text(T_x,T_y[7],"A: "+format(x1_p[0],"5.2f"))
        ax1.text(T_x,T_y[8],"FWHM: "+format(2.3548* x1_p[2],"7.4f"))
        ax1.text(T_x-0.05,T_y[9]-5,"CTR Mean: "+format(x1_p[1],"7.3f"), fontsize='x-large')
        ax1.set_xlim(xlim[0], xlim[1])
        ax1.set_ylim(ylim[0], ylim[1])
        plt.show()

    def DOIHist(dataClass, xlim=[0,31], ylim=[-600,600], Fit=fit):
        
