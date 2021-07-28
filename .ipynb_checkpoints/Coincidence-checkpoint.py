#Will Matava, Kyle Klein

#Takes in Coincidence Data for MiniPET with 2 PCBs connected; outputs 2 plots showing charge spectra for each channel

#NOTE ALL RUNS TAKEN BEFORE 4-20 HAVE LEFT AND RIGHT SWAPPED

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy.optimize import curve_fit
import seaborn as sns
from Bad_Pix import Run17_5, Run17_6, Run18_5, Run18_6
import statistics as stat
from tqdm import tqdm


pd.set_option("display.max_columns",100)

class Channel_Pair:
    def __init__(self, ChannelPairData):
        self.df = ChannelPairData
        valuesR,binsR,presentersR = plt.hist(self.df.ChargeR, bins = np.linspace(0,29,160), ec='r',label='PCB 5', fill=False)
        valuesL,binsL,presentersL = plt.hist(self.df.ChargeL, bins = np.linspace(0,29,160), ec='b',label='PCB 6', fill=False)
        bin_centers = [(binsR[q] + binsR[q+1])/2 for q in range(0,len(binsR)-1)]
        self.EHistR_y = valuesR.tolist()
        self.EHistR_x = bin_centers
        self.EHistL_y = valuesL.tolist()
        self.EHistL_x = bin_centers
        self.PhotopeakR = 0
        self.PhotopeakL = 0
        self.EResR = 0
        self.EResL = 0
        self.PP_Count = 0
        self.EFitParam = 0
        self.CTRFitParam = 0
        plt.clf()
        plt.close()

    def findPhotopeak(self):
        #find the peak within 3 bins starting from the RHS of the graph
        for i in range(1, len(self.EHistR_y)-4):
            if self.EHistR_y[-1*i] > self.EHistR_y[-1*(i+1)] and self.EHistR_y[-1*i] > self.EHistR_y[-1*(i+2)] and self.EHistR_y[-1*i] > self.EHistR_y[-1*(i+3)] and self.EHistR_y[-1*(i+1)] != 0 and self.EHistR_y[-1*i] >= 30:
                self.PhotopeakR = self.EHistR_y.index(self.EHistR_y[-1*i],-1*i-1)
                break
        for i in range(1, len(self.EHistL_y)-4):
            if self.EHistL_y[-1*i] > self.EHistL_y[-1*(i+1)] and self.EHistL_y[-1*i] > self.EHistL_y[-1*(i+2)] and self.EHistL_y[-1*i] > self.EHistL_y[-1*(i+3)]and self.EHistL_y[-1*(i+1)] != 0 and self.EHistL_y[-1*i] >= 30:
                self.PhotopeakL = self.EHistL_y.index(self.EHistL_y[-1*i],-1*i-1)
                break

    def getQINTs(self):
        # numpy requires the arguments y before x
        INTR = np.trapz(self.EHistR_y,self.EHistR_x)
        INTL = np.trapz(self.EHistL_y,self.EHistL_x)
        return [INTR, INTL]

    def PPFit(self):
        ExR_p = [0,1,1]
        ExL_p = [0,1,1]

        fitR_x = self.EHistR_x[self.PhotopeakR-7:self.PhotopeakR+8]
        fitL_x = self.EHistL_x[self.PhotopeakL-7:self.PhotopeakL+8]
        fitR_y = self.EHistR_y[self.PhotopeakR-7:self.PhotopeakR+8]
        fitL_y = self.EHistL_y[self.PhotopeakL-7:self.PhotopeakL+8]

        #fitting said data and recording photpeak energy cuts
        if len(fitR_y) != 0 and len(fitR_x) != 0:
            try:
                ExR_p, ExR_co = curve_fit(gauss, fitR_x, fitR_y, p0=[100, fitR_x[7], 2])
            except RuntimeError:
                print("Error - curve_fit failed on Right PCB Failed")

        if len(fitL_x) != 0 and len(fitL_y) != 0:
            try:
                ExL_p, ExL_co = curve_fit(gauss, fitL_x, fitL_y,p0=[200,fitL_x[7],2])
            except RuntimeError:
                print("Error - curve_fit failed on Left PCB Failed")

        return [ExR_p, ExL_p]

    def setERes(self):
        self.EResR = abs(100*(2.3548*abs(self.EFitParam[0][2]))/self.EFitParam[0][1])
        self.EResL = abs(100*(2.3548*abs(self.EFitParam[1][2]))/self.EFitParam[1][1])

    def SetEnergyParameters(self, run_param):
        self.EFitParam = run_param

    def CutOnEnergy(self):
        R_cut = self.EFitParam[0][1] - 2.5*self.EFitParam[0][2]
        L_cut = self.EFitParam[1][1] - 2.5*self.EFitParam[1][2]
        self.df = self.df[self.df["ChargeR"] >= R_cut]
        self.df = self.df[self.df["ChargeL"] >= L_cut].reset_index(drop=True)
        self.df["Time_diff"] = 10**-3 * (self.df["TimeL"] - self.df["TimeR"])

    def EnergyCut(self):
        if self.EFitParam[0][1] > 10.5 and self.EFitParam[0][1] > 10.5:
            return True
        else:
            return False

    def PPCut(self):
        if self.df["Time_diff"].count() < 100:
            return False
        else:
            return True

    def SetCTRParameters(self, CTR_param):
        self.CTRFitParam = CTR_param

    def CTRFit(self):
        TRes5_UN = self.df["Time_diff"]

        x1_p = [0,1,1]
        low = -2
        high = low * -1
        num_bins = 150
        bins = np.linspace(low,high,num_bins)
        x = np.linspace(low+((high-low)/(2*num_bins)),high-((high-low)/(2*num_bins)),num_bins-1)
        y1 = TRes5_UN.value_counts(bins=bins,sort=False).tolist()
        peak = y1.index(max(y1))
        bin_centers = [(bins[i] + bins[i+1])/2 for i in range(0,len(bins)-1)]
        fit_x = bin_centers[peak-12:peak+13]
        fit_y = y1[peak-12:peak+13]
        try:
            x1_p, x1_co = curve_fit(gauss,fit_x,fit_y, p0=[75,TRes5_UN.mean(),TRes5_UN.std()])
        except RuntimeError:
            print("Error - curve_fit failed")

        return x1_p


    def PlotEnergySpectrum(self, display=False):
        global LChannel
        global RChannel
        # plotting every energy spectrum
        x_f = np.linspace(0,29,300)
        plt.step(ChannelPair.EHistR_x, ChannelPair.EHistR_y, color='r',label='PCB 5')
        plt.step(ChannelPair.EHistL_x, ChannelPair.EHistL_y, color='b',label='PCB 6')
        plt.plot(x_f, gauss(x_f,*self.EFitParam[0]), color='k')
        plt.plot(x_f, gauss(x_f,*self.EFitParam[1]), color='k')
        t_y = generateTextY(10,400)
        t_x = 1
        plt.text(t_x,t_y[0], "PCB 5:")
        plt.text(t_x,t_y[1], "mean: "+format(self.EFitParam[0][1], "5.2f"))
        plt.text(t_x,t_y[2], "std: "+format(abs(self.EFitParam[0][2]), "5.2f"))
        plt.text(t_x,t_y[3], "count: "+format(ChannelPair.df.ChargeR.count(), "5"))
        plt.text(t_x,t_y[4], "ERes: "+format(self.EResR, "5.2f")+"%")
        plt.text(t_x,t_y[5], "PCB 6:")
        plt.text(t_x,t_y[6], "mean: "+format(self.EFitParam[1][1], "5.2f"))
        plt.text(t_x,t_y[7], "std: "+format(abs(self.EFitParam[1][2]), "5.2f"))
        plt.text(t_x,t_y[8], "count: "+format(ChannelPair.df.ChargeL.count(), "5"))
        plt.text(t_x,t_y[9], "ERes: "+format(self.EResL, "5.2f")+"%")
        plt.ylim(0,400)
        plt.title("Energy Spectrum, Pair {}, {}".format(toGeo(LChannel), toGeo(RChannel)))
        plt.legend()
        plt.savefig("Energy Spectrum, Pair {}, {}".format(toGeo(LChannel), toGeo(RChannel)))
        if display == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()

    def PlotCTR(self, display=False):
        global LChannel
        global RChannel
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111,xlim=(-2,2), ylim=(0,100),xlabel="Time in ns",ylabel="Counts",title="CTR: Channel Pair {}, {}".format(toGeo(LChannel),toGeo(RChannel)))
        low = -2
        high = low * -1
        num_bins = 150
        bins = np.linspace(low,high,num_bins)
        x_p = np.linspace(low+((high-low)/(2*num_bins)),high-((high-low)/(2*num_bins)),400)
        T_x = -1.9
        T_y = generateTextY(9,100)
        TRes5_UN = self.df["Time_diff"]
        ax1.hist(TRes5_UN, bins=bins,color='b')
        ax1.plot(x_p,gauss(x_p,*self.CTRFitParam), color='k')
        ax1.text(T_x,T_y[0],"Data Stats:")
        ax1.text(T_x,T_y[1],"mean: "+format(TRes5_UN.mean(),"5.2f"))
        ax1.text(T_x,T_y[2],"std: "+format(TRes5_UN.std(),"7.4f"))
        ax1.text(T_x,T_y[3],"count: "+str(len(TRes5_UN)))
        ax1.text(T_x,T_y[4],"Fit Paramters:")
        ax1.text(T_x,T_y[5],"mu: "+format(self.CTRFitParam[1],"5.2f"))
        ax1.text(T_x,T_y[6],"sigma: "+format(self.CTRFitParam[2],"7.4f"))
        ax1.text(T_x,T_y[7],"A: "+format(self.CTRFitParam[0],"5.2f"))
        ax1.text(T_x,T_y[8],"FWHM: "+format(2.3548* self.CTRFitParam[2],"7.4f"))
        plt.savefig("CTR, Pair {}, {}".format(toGeo(LChannel), toGeo(RChannel)))
        if display == True:
            plt.show()
            plt.clf()
            plt.close()
        else:
            plt.clf()
            plt.close()


def generateTextY(num,max):
    y = [max - 0.04*(i+1)*max for i in range(0,num+1)]
    return y

def gauss(x,A,mu,sigma):
    y = A*np.exp(-(x - mu)**2 / (2 * sigma**2))
    return y

#converts PETSys ID to geometric ID
def toGeo(x):
    y = 8*indices.get(x)[0] + indices.get(x)[1]
    return y


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

dir = "/home/kilroy101/path/Geant4/Expirimental_Data/"
run_num = 41

indices = {
      0 : (4,7-7),
      1 : (4,7-6),
      2 : (7,7-5),
      3 : (5,7-7),
      4 : (5,7-4),
      5 : (5,7-5),
      6 : (4,7-4),
      7 : (7,7-7),
      8 : (6,7-6),
      9 : (7,7-4),
      10 : (5,7-6),
      11 : (6,7-4),
      12 : (4,7-5),
      13 : (6,7-5),
      14 : (6,7-7),
      15 : (7,7-6),
      16 : (3,7-7),
      17 : (3,7-6),
      18 : (2,7-7),
      19 : (2,7-6),
      20 : (0,7-7),
      21 : (1,7-7),
      22 : (0,7-6),
      23 : (1,7-6),
      24 : (3,7-5),
      25 : (1,7-5),
      26 : (2,7-5),
      27 : (4,7-3),
      28 : (0,7-5),
      29 : (3,7-4),
      30 : (0,7-4),
      31 : (1,7-4),
      32 : (2,7-4),
      33 : (3,7-3),
      34 : (2,7-3),
      35 : (0,7-3),
      36 : (1,7-3),
      37 : (0,7-2),
      38 : (5,7-3),
      39 : (1,7-2),
      40 : (2,7-2),
      41 : (3,7-2),
      42 : (1,7-1),
      43 : (0,7-1),
      44 : (0,7-0),
      45 : (3,7-1),
      46 : (1,7-0),
      47 : (2,7-1),
      48 : (3,7-0),
      49 : (2,7-0),
      50 : (6,7-2),
      51 : (6,7-1),
      52 : (7,7-1),
      53 : (4,7-1),
      54 : (5,7-1),
      55 : (6,7-0),
      56 : (7,7-0),
      57 : (7,7-2),
      58 : (7,7-3),
      59 : (4,7-2),
      60 : (5,7-0),
      61 : (5,7-2),
      62 : (6,7-3),
      63 : (4,7-0),
      64:(3+8,7),
      65:(3+8,6),
      66:(2+8,4),
      67:(2+8,6),
      68:(3+8,4),
      69:(1+8,7),
      70:(1+8,5),
      71:(0+8,7),
      72:(1+8,6),
      73:(3+8,3),
      74:(2+8,7),
      75:(2+8,3),
      76:(3+8,5),
      77:(0+8,5),
      78:(2+8,5),
      79:(0+8,6),
      80:(4+8,7),
      81:(6+8,7),
      82:(5+8,7),
      83:(7+8,7),
      84:(5+8,6),
      85:(4+8,6),
      86:(6+8,6),
      87:(7+8,6),
      88:(4+8,5),
      89:(6+8,5),
      90:(5+8,5),
      91:(1+8,4),
      92:(7+8,5),
      93:(7+8,4),
      94:(6+8,4),
      95:(4+8,4),
      96:(5+8,4),
      97:(5+8,3),
      98:(6+8,3),
      99:(4+8,3),
      100:(7+8,3),
      101:(7+8,2),
      102:(0+8,4),
      103:(6+8,2),
      104:(7+8,1),
      105:(5+8,2),
      106:(6+8,1),
      107:(4+8,2),
      108:(7+8,0),
      109:(5+8,1),
      110:(6+8,0),
      111:(4+8,1),
      112:(5+8,0),
      113:(4+8,0),
      114:(0+8,2),
      115:(2+8,1),
      116:(0+8,1),
      117:(3+8,1),
      118:(1+8,1),
      119:(1+8,0),
      120:(0+8,0),
      121:(1+8,2),
      122:(1+8,3),
      123:(3+8,2),
      124:(2+8,0),
      125:(2+8,2),
      126:(0+8,3),
      127:(3+8,0)}

df1 = pd.read_csv(dir+"Run{}_coinc.dat".format(run_num), sep="\t", header=None, low_memory=False)
df1.columns = ["A", "B", "TimeL", "ChargeL", "ChannelIDL", "C", "D", "TimeR", "ChargeR", "ChannelIDR"]
df1["Left"] = df1.ChannelIDL % 128
df1["Right"] = df1.ChannelIDR % 128
df1.ChannelIDR = df1.Right
df1.ChannelIDL = df1.Left

# fig1, axs1 = plt.subplots(nrows = 16,ncols = 8, figsize = [6,12], sharex = True, sharey= True)
# fig2, axs2 = plt.subplots(nrows = 16,ncols = 8, figsize = [6,12], sharex = True, sharey= True)

#making the almighty dataframe which will store everything we need, praise be
DATA = pd.DataFrame(columns=["ChannelIDL", "ChannelIDR", "PP_Count", "PeakL", "PeakR", "E_ResL", "E_ResR", "QINTL", "QINTR", "CTR", "E_Cut", "PP_Cut"])

pairs = df1.groupby(["ChannelIDL", "ChannelIDR"]).size().reset_index(name="Counts").sort_values(by='Counts', ascending=False).reset_index()

num_photons = 300
valid_channels = pairs[pairs["Counts"] >= num_photons].reset_index()
for j in tqdm(range(valid_channels.ChannelIDL.count())):


    test = df1[df1["ChannelIDL"] == valid_channels.ChannelIDL[j]]
    test = test[test["ChannelIDR"] == valid_channels.ChannelIDR[j]].reset_index()
    RChannel = valid_channels.ChannelIDR[j]
    LChannel = valid_channels.ChannelIDL[j]

    ChannelPair = Channel_Pair(test)
    ChannelPair.findPhotopeak()
    ChannelPair.SetEnergyParameters(ChannelPair.PPFit())
    QInts = ChannelPair.getQINTs()
    ChannelPair.setERes()
    ChannelPair.PlotEnergySpectrum(display=True)


    if ChannelPair.EnergyCut() == True:
        ChannelPair.CutOnEnergy()
        if ChannelPair.PPCut() == True:
            ChannelPair.SetCTRParameters(ChannelPair.CTRFit())
            ChannelPair.PlotCTR(display=True)
            DATA = DATA.append({"ChannelIDL":LChannel, "ChannelIDR":RChannel, "PP_Count":ChannelPair.PP_Count, "PeakL":ChannelPair.PhotopeakL, "PeakR":ChannelPair.PhotopeakR, "E_ResL":ChannelPair.EResL, "E_ResR":ChannelPair.EResR,"QINTL":QInts[1], "QINTR":QInts[0],"CTR":abs(2.3548* ChannelPair.CTRFitParam[2]), "E_Cut":1, "PP_Cut":1},ignore_index=True)
        elif ChannelPair.PPCut() == False:
            DATA = DATA.append({"ChannelIDL":LChannel, "ChannelIDR":RChannel, "PP_Count":ChannelPair.PP_Count, "PeakL":ChannelPair.PhotopeakL, "PeakR":ChannelPair.PhotopeakR, "E_ResL":ChannelPair.EResL, "E_ResR":ChannelPair.EResR,"QINTL":QInts[1], "QINTR":QInts[0],"CTR":0, "E_Cut":1, "PP_Cut":0},ignore_index=True)
    elif ChannelPair.EnergyCut() == False:
        DATA = DATA.append({"ChannelIDL":LChannel, "ChannelIDR":RChannel, "PP_Count":0, "PeakL":ChannelPair.PhotopeakL, "PeakR":ChannelPair.PhotopeakL, "E_ResL":ChannelPair.EResL, "E_ResR":ChannelPair.EResR,"QINTL":QInts[1], "QINTR":QInts[0],"CTR":0, "E_Cut":0, "PP_Cut":0},ignore_index=True)

DATA.to_csv(dir+"Run{}_DATA.csv".format(run_num))


# df = pd.read_csv(dir+"Run{}_DATA.csv".format(run_num), sep=",", low_memory=False)
# # This is pretty dumb, but I can't directly hash the dictionary with the dataframe column, so it's pandas fault.
# Geo_L = []
# Geo_R = []
# for i in range(df.ChannelIDL.count()):
#     Geo_L.append(toGeo(df["ChannelIDL"][i]))
#     Geo_R.append(toGeo(df["ChannelIDR"][i]))
# df["GeoChannelIDL"] = Geo_L
# df["GeoChannelIDR"] = Geo_R
# df["CTR"] = df["CTR"].abs()
#
# #only keeping the data that passed the cuts
# data = df[df["E_Cut"] == 1].reset_index(drop=True)
# data = data[data["PP_Cut"] == 1].reset_index(drop=True)
#
#
# # #cutting on array pairs
# # data = data[data["GeoChannelIDR"] < 64]
# # data = data[data["GeoChannelIDL"] > 63]
#
#
# cmap = plt.get_cmap('binary')
# cmap = truncate_colormap(cmap, 0.2, 0.8)


# # visual crystal quality code
# LQ = pd.read_csv(dir+"PCB6CrystPic.txt", sep="\t", low_memory=False)
# LQ = LQ.reset_index()
# LQ["GeoChannelIDL"] = LQ["index"]
# LQ = LQ.drop(axis=1,labels=["Crystal ID","index"])
#
# RQ = pd.read_csv(dir+"PCB5CrystPic.txt", sep="\t", low_memory=False)
# RQ = RQ.reset_index()
# RQ["GeoChannelIDR"] = RQ["index"]
# RQ = RQ.drop(axis=1,labels=["Crystal ID","index"])
#
#
# LQA = LQ[LQ["Quality"] =="A"].reset_index(drop=True)
# LQB = LQ[LQ["Quality"] =="B"].reset_index(drop=True)
# LQC = LQ[LQ["Quality"] =="C"].reset_index(drop=True)
# LQD = LQ[LQ["Quality"] =="D"].reset_index(drop=True)
#
# LQCD = pd.concat([LQC, LQD], ignore_index=True)
#
# RQA = RQ[RQ["Quality"] =="A"].reset_index(drop=True)
# RQB = RQ[RQ["Quality"] =="B"].reset_index(drop=True)
# RQC = RQ[RQ["Quality"] =="C"].reset_index(drop=True)
# RQD = RQ[RQ["Quality"] =="D"].reset_index(drop=True)
#
# RQCD = pd.concat([RQC, RQD], ignore_index=True)
#
# LQA = pd.merge(data, LQA, on='GeoChannelIDL').reset_index(drop=True)
# RQA = pd.merge(data, RQA, on='GeoChannelIDR').reset_index(drop=True)
# LQCD = pd.merge(data, LQCD, on='GeoChannelIDL').reset_index(drop=True)
# RQCD = pd.merge(data, RQCD, on='GeoChannelIDR').reset_index(drop=True)
#
# data = pd.concat([LQCD, RQCD], ignore_index=True)



# # #plotting all the CTRS
# CTRs = data.CTR
# CTRs = CTRs.abs()
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111,xlim=(0,0.9), ylim=(0,50),xlabel="CTR in ns",ylabel="Counts",title="CTR of most active Channel Pairs \n ")
#
# low = 0
# high = low +1 #high = low * -1
# num_bins = 100
# bins = np.linspace(low,high,num_bins)
# x = np.linspace(low+((high-low)/(2*num_bins)),high-((high-low)/(2*num_bins)),num_bins-1)
# y1 = CTRs.value_counts(bins=bins,sort=False).tolist()
# x1_p, x1_co = curve_fit(gauss,x,y1, p0=[75,CTRs.mean(),CTRs.std()])
#
#
# x_p = np.linspace(low+((high-low)/(2*num_bins)),high-((high-low)/(2*num_bins)),400)
# T_x = 0.6
# T_y = generateTextY(9,50)
#
# ax1.hist(CTRs, bins=bins,color='b')
# ax1.plot(x_p,gauss(x_p,x1_p[0],x1_p[1],x1_p[2]), color='k')
# ax1.text(T_x,T_y[0],"Data Stats:")
# ax1.text(T_x,T_y[1],"mean: "+format(CTRs.mean(),"7.4f"))
# ax1.text(T_x,T_y[2],"std: "+format(CTRs.std(),"7.4f"))
# ax1.text(T_x,T_y[3],"count: "+str(len(CTRs)))
# ax1.text(T_x,T_y[4],"Fit Paramters:")
# ax1.text(T_x,T_y[5],"mu: "+format(x1_p[1],"7.4f"))
# ax1.text(T_x,T_y[6],"sigma: "+format(x1_p[2],"7.4f"))
# ax1.text(T_x,T_y[7],"A: "+format(x1_p[0],"5.2f"))
# ax1.text(T_x,T_y[8],"FWHM: "+format(2.3548* x1_p[2],"7.4f"))
# plt.show()

# # All CTRS plotted by array
# CTRL = data.sort_values(by="ChannelIDL")
# CTRR = data.sort_values(by="ChannelIDR")
#
# CTRRT = CTRR[CTRR["ChannelIDR"] < 64].reset_index(drop=True)
# CTRRB = CTRR[CTRR["ChannelIDR"] >= 64].reset_index(drop=True)
# CTRLT = CTRL[CTRL["ChannelIDL"] < 64].reset_index(drop=True)
# CTRLB = CTRL[CTRL["ChannelIDL"] >= 64].reset_index(drop=True)
#
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(221,xlim=(0,0.9), ylim=(0,50),xlabel="CTR in ns",ylabel="Counts",title="Best CTR PCB 6")
# ax2 = fig1.add_subplot(222,xlim=(0,0.9), ylim=(0,50),xlabel="CTR in ns",ylabel="Counts",title="Best CTR PCB 8")
# ax3 = fig1.add_subplot(223,xlim=(0,0.9), ylim=(0,50),xlabel="CTR in ns",ylabel="Counts")
# ax4 = fig1.add_subplot(224,xlim=(0,0.9), ylim=(0,50),xlabel="CTR in ns",ylabel="Counts")
#
# low = 0
# high = low +1 #high = low * -1
# num_bins = 100
# bins = np.linspace(low,high,num_bins)
# x = np.linspace(low+((high-low)/(2*num_bins)),high-((high-low)/(2*num_bins)),num_bins-1)
# y1 = CTRLT.CTR.value_counts(bins=bins,sort=False).tolist()
# y2 = CTRRT.CTR.value_counts(bins=bins,sort=False).tolist()
# y3 = CTRLB.CTR.value_counts(bins=bins,sort=False).tolist()
# y4 = CTRRB.CTR.value_counts(bins=bins,sort=False).tolist()
# x1_p, x1_co = curve_fit(gauss,x,y1, p0=[75,CTRLT.CTR.mean(),CTRLT.CTR.std()])
# x2_p, x2_co = curve_fit(gauss,x,y2, p0=[75,CTRRT.CTR.mean(),CTRRT.CTR.std()])
# x3_p, x3_co = curve_fit(gauss,x,y3, p0=[75,CTRLB.CTR.mean(),CTRLB.CTR.std()])
# x4_p, x4_co = curve_fit(gauss,x,y4, p0=[75,CTRRB.CTR.mean(),CTRRB.CTR.std()])
#
#
# x_p = np.linspace(low+((high-low)/(2*num_bins)),high-((high-low)/(2*num_bins)),400)
# T_x = 0.6
# T_y = generateTextY(9,50)
#
# ax1.hist(CTRLT.CTR, bins=bins,color='b')
# ax1.plot(x_p,gauss(x_p,x1_p[0],x1_p[1],x1_p[2]), color='k')
# ax1.text(T_x,T_y[0],"Data Stats:")
# ax1.text(T_x,T_y[1],"mean: "+format(CTRLT.CTR.mean(),"7.4f"))
# ax1.text(T_x,T_y[2],"std: "+format(CTRLT.CTR.std(),"7.4f"))
# ax1.text(T_x,T_y[3],"count: "+str(CTRLT.CTR.count()))
# ax1.text(T_x,T_y[4],"Fit Paramters:")
# ax1.text(T_x,T_y[5],"mu: "+format(x1_p[1],"7.4f"))
# ax1.text(T_x,T_y[6],"sigma: "+format(x1_p[2],"7.4f"))
# ax1.text(T_x,T_y[7],"A: "+format(x1_p[0],"5.2f"))
# ax1.text(T_x,T_y[8],"FWHM: "+format(2.3548* x1_p[2],"7.4f"))
# ax1.text(T_x-0.02,T_y[8]-5,"CTR Mean: "+format(x1_p[1],"7.3f"), fontsize='x-large')
#
#
# ax2.hist(CTRRT.CTR, bins=bins,color='b')
# ax2.plot(x_p,gauss(x_p,x2_p[0],x2_p[1],x2_p[2]), color='k')
# ax2.text(T_x,T_y[0],"Data Stats:")
# ax2.text(T_x,T_y[1],"mean: "+format(CTRRT.CTR.mean(),"7.4f"))
# ax2.text(T_x,T_y[2],"std: "+format(CTRRT.CTR.std(),"7.4f"))
# ax2.text(T_x,T_y[3],"count: "+str(CTRRT.CTR.count()))
# ax2.text(T_x,T_y[4],"Fit Paramters:")
# ax2.text(T_x,T_y[5],"mu: "+format(x2_p[1],"7.4f"))
# ax2.text(T_x,T_y[6],"sigma: "+format(x2_p[2],"7.4f"))
# ax2.text(T_x,T_y[7],"A: "+format(x2_p[0],"5.2f"))
# ax2.text(T_x,T_y[8],"FWHM: "+format(2.3548* x2_p[2],"7.4f"))
# ax2.text(T_x-0.02,T_y[8]-5,"CTR Mean: "+format(x2_p[1],"7.3f"), fontsize='x-large')
#
# ax3.hist(CTRLB.CTR, bins=bins,color='b')
# ax3.plot(x_p,gauss(x_p,x3_p[0],x3_p[1],x3_p[2]), color='k')
# ax3.text(T_x,T_y[0],"Data Stats:")
# ax3.text(T_x,T_y[1],"mean: "+format(CTRLT.CTR.mean(),"7.4f"))
# ax3.text(T_x,T_y[2],"std: "+format(CTRLT.CTR.std(),"7.4f"))
# ax3.text(T_x,T_y[3],"count: "+str(CTRLT.CTR.count()))
# ax3.text(T_x,T_y[4],"Fit Paramters:")
# ax3.text(T_x,T_y[5],"mu: "+format(x3_p[1],"7.4f"))
# ax3.text(T_x,T_y[6],"sigma: "+format(x3_p[2],"7.4f"))
# ax3.text(T_x,T_y[7],"A: "+format(x3_p[0],"5.2f"))
# ax3.text(T_x,T_y[8],"FWHM: "+format(2.3548* x3_p[2],"7.4f"))
# ax3.text(T_x-0.02,T_y[8]-5,"CTR Mean: "+format(x3_p[1],"7.3f"), fontsize='x-large')
#
# ax4.hist(CTRRB.CTR, bins=bins,color='b')
# ax4.plot(x_p,gauss(x_p,x4_p[0],x4_p[1],x4_p[2]), color='k')
# ax4.text(T_x,T_y[0],"Data Stats:")
# ax4.text(T_x,T_y[1],"mean: "+format(CTRLT.CTR.mean(),"7.4f"))
# ax4.text(T_x,T_y[2],"std: "+format(CTRLT.CTR.std(),"7.4f"))
# ax4.text(T_x,T_y[3],"count: "+str(CTRLT.CTR.count()))
# ax4.text(T_x,T_y[4],"Fit Paramters:")
# ax4.text(T_x,T_y[5],"mu: "+format(x4_p[1],"7.4f"))
# ax4.text(T_x,T_y[6],"sigma: "+format(x4_p[2],"7.4f"))
# ax4.text(T_x,T_y[7],"A: "+format(x4_p[0],"5.2f"))
# ax4.text(T_x,T_y[8],"FWHM: "+format(2.3548* x4_p[2],"7.4f"))
# ax4.text(T_x-0.02,T_y[8]-5,"CTR Mean: "+format(x4_p[1],"7.3f"), fontsize='x-large')
#
#
# plt.show()


# # Best CTR by pixel
# CTRL = data.groupby("GeoChannelIDL").CTR.min().reset_index()
# CTRL.columns = ["ChannelIDL", "CTR"]
#
# #I know there is a better way to do this but this is just a quick method, feel free to suggest an improvment
#
# # #Uncomment for heatmap
# # CTRL["CTR"] = 1000*CTRL["CTR"]
# # for i in range(0,128):
# #     run = CTRL[CTRL["ChannelIDL"] == i]
# #     if run.ChannelIDL.count() == 0:
# #         CTRL = CTRL.append({"ChannelIDL":i, "CTR":0},ignore_index=True)
#
# CTRR = data.groupby("GeoChannelIDR").CTR.min().reset_index()
# CTRR.columns = ["ChannelIDR", "CTR"]
#
# # #Uncomment for heatmap
# # CTRR["CTR"] = 1000*CTRR["CTR"]
# # for i in range(0,128):
# #     run = CTRR[CTRR["ChannelIDR"] == i]
# #     if run.ChannelIDR.count() == 0:
# #         CTRR = CTRR.append({"ChannelIDR":i, "CTR":0},ignore_index=True)
#
# CTRL = CTRL.sort_values(by="ChannelIDL")
# CTRR = CTRR.sort_values(by="ChannelIDR")
#
# CTRRT = CTRR[CTRR["ChannelIDR"] < 64].reset_index(drop=True)
# CTRRB = CTRR[CTRR["ChannelIDR"] >= 64].reset_index(drop=True)
# CTRLT = CTRL[CTRL["ChannelIDL"] < 64].reset_index(drop=True)
# CTRLB = CTRL[CTRL["ChannelIDL"] >= 64].reset_index(drop=True)
#
# # on a histogram for each array
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(221,xlim=(0,0.9), ylim=(0,50),xlabel="CTR in ns",ylabel="Counts",title="Best CTR PCB 6")
# ax2 = fig1.add_subplot(222,xlim=(0,0.9), ylim=(0,50),xlabel="CTR in ns",ylabel="Counts",title="Best CTR PCB 8")
# ax3 = fig1.add_subplot(223,xlim=(0,0.9), ylim=(0,50),xlabel="CTR in ns",ylabel="Counts")
# ax4 = fig1.add_subplot(224,xlim=(0,0.9), ylim=(0,50),xlabel="CTR in ns",ylabel="Counts")
#
# low = 0
# high = low +1 #high = low * -1
# num_bins = 100
# bins = np.linspace(low,high,num_bins)
# x = np.linspace(low+((high-low)/(2*num_bins)),high-((high-low)/(2*num_bins)),num_bins-1)
# y1 = CTRLT.CTR.value_counts(bins=bins,sort=False).tolist()
# y2 = CTRRT.CTR.value_counts(bins=bins,sort=False).tolist()
# y3 = CTRLB.CTR.value_counts(bins=bins,sort=False).tolist()
# y4 = CTRRB.CTR.value_counts(bins=bins,sort=False).tolist()
# x1_p, x1_co = curve_fit(gauss,x,y1, p0=[75,CTRLT.CTR.mean(),CTRLT.CTR.std()])
# x2_p, x2_co = curve_fit(gauss,x,y2, p0=[75,CTRRT.CTR.mean(),CTRRT.CTR.std()])
# x3_p, x3_co = curve_fit(gauss,x,y3, p0=[75,CTRLB.CTR.mean(),CTRLB.CTR.std()])
# x4_p, x4_co = curve_fit(gauss,x,y4, p0=[75,CTRRB.CTR.mean(),CTRRB.CTR.std()])
#
#
# x_p = np.linspace(low+((high-low)/(2*num_bins)),high-((high-low)/(2*num_bins)),400)
# T_x = 0.6
# T_y = generateTextY(9,50)
#
# ax1.hist(CTRLT.CTR, bins=bins,color='b')
# ax1.plot(x_p,gauss(x_p,x1_p[0],x1_p[1],x1_p[2]), color='k')
# ax1.text(T_x,T_y[0],"Data Stats:")
# ax1.text(T_x,T_y[1],"mean: "+format(CTRLT.CTR.mean(),"7.4f"))
# ax1.text(T_x,T_y[2],"std: "+format(CTRLT.CTR.std(),"7.4f"))
# ax1.text(T_x,T_y[3],"count: "+str(CTRLT.CTR.count()))
# ax1.text(T_x,T_y[4],"Fit Paramters:")
# ax1.text(T_x,T_y[5],"mu: "+format(x1_p[1],"7.4f"))
# ax1.text(T_x,T_y[6],"sigma: "+format(x1_p[2],"7.4f"))
# ax1.text(T_x,T_y[7],"A: "+format(x1_p[0],"5.2f"))
# ax1.text(T_x,T_y[8],"FWHM: "+format(2.3548* x1_p[2],"7.4f"))
# ax1.text(T_x-0.02,T_y[8]-5,"CTR Mean: "+format(x1_p[1],"7.3f"), fontsize='x-large')
#
# ax2.hist(CTRRT.CTR, bins=bins,color='b')
# ax2.plot(x_p,gauss(x_p,x2_p[0],x2_p[1],x2_p[2]), color='k')
# ax2.text(T_x,T_y[0],"Data Stats:")
# ax2.text(T_x,T_y[1],"mean: "+format(CTRRT.CTR.mean(),"7.4f"))
# ax2.text(T_x,T_y[2],"std: "+format(CTRRT.CTR.std(),"7.4f"))
# ax2.text(T_x,T_y[3],"count: "+str(CTRRT.CTR.count()))
# ax2.text(T_x,T_y[4],"Fit Paramters:")
# ax2.text(T_x,T_y[5],"mu: "+format(x2_p[1],"7.4f"))
# ax2.text(T_x,T_y[6],"sigma: "+format(x2_p[2],"7.4f"))
# ax2.text(T_x,T_y[7],"A: "+format(x2_p[0],"5.2f"))
# ax2.text(T_x,T_y[8],"FWHM: "+format(2.3548* x2_p[2],"7.4f"))
# ax2.text(T_x-0.02,T_y[8]-5,"CTR Mean: "+format(x2_p[1],"7.3f"), fontsize='x-large')
#
# ax3.hist(CTRLB.CTR, bins=bins,color='b')
# ax3.plot(x_p,gauss(x_p,x3_p[0],x3_p[1],x3_p[2]), color='k')
# ax3.text(T_x,T_y[0],"Data Stats:")
# ax3.text(T_x,T_y[1],"mean: "+format(CTRLT.CTR.mean(),"7.4f"))
# ax3.text(T_x,T_y[2],"std: "+format(CTRLT.CTR.std(),"7.4f"))
# ax3.text(T_x,T_y[3],"count: "+str(CTRLT.CTR.count()))
# ax3.text(T_x,T_y[4],"Fit Paramters:")
# ax3.text(T_x,T_y[5],"mu: "+format(x3_p[1],"7.4f"))
# ax3.text(T_x,T_y[6],"sigma: "+format(x3_p[2],"7.4f"))
# ax3.text(T_x,T_y[7],"A: "+format(x3_p[0],"5.2f"))
# ax3.text(T_x,T_y[8],"FWHM: "+format(2.3548* x3_p[2],"7.4f"))
# ax3.text(T_x-0.02,T_y[8]-5,"CTR Mean: "+format(x3_p[1],"7.3f"), fontsize='x-large')
#
# ax4.hist(CTRRB.CTR, bins=bins,color='b')
# ax4.plot(x_p,gauss(x_p,x4_p[0],x4_p[1],x4_p[2]), color='k')
# ax4.text(T_x,T_y[0],"Data Stats:")
# ax4.text(T_x,T_y[1],"mean: "+format(CTRLT.CTR.mean(),"7.4f"))
# ax4.text(T_x,T_y[2],"std: "+format(CTRLT.CTR.std(),"7.4f"))
# ax4.text(T_x,T_y[3],"count: "+str(CTRLT.CTR.count()))
# ax4.text(T_x,T_y[4],"Fit Paramters:")
# ax4.text(T_x,T_y[5],"mu: "+format(x4_p[1],"7.4f"))
# ax4.text(T_x,T_y[6],"sigma: "+format(x4_p[2],"7.4f"))
# ax4.text(T_x,T_y[7],"A: "+format(x4_p[0],"5.2f"))
# ax4.text(T_x,T_y[8],"FWHM: "+format(2.3548* x4_p[2],"7.4f"))
# ax4.text(T_x-0.02,T_y[8]-5,"CTR Mean: "+format(x4_p[1],"7.3f"), fontsize='x-large')
#
#
# plt.show()
#
# # on a single histogram
# oof = pd.concat([CTRLT, CTRLB, CTRRT, CTRRB], ignore_index=True)
# CTRs = oof.CTR
# CTRs = CTRs.abs()
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111,xlim=(0,0.9), ylim=(0,50),xlabel="CTR in ns",ylabel="Counts",title="CTR of most active Channel Pairs \n Best CTRs per pixel")
#
# low = 0
# high = low +1 #high = low * -1
# num_bins = 100
# bins = np.linspace(low,high,num_bins)
# x = np.linspace(low+((high-low)/(2*num_bins)),high-((high-low)/(2*num_bins)),num_bins-1)
# y1 = CTRs.value_counts(bins=bins,sort=False).tolist()
# x1_p, x1_co = curve_fit(gauss,x,y1, p0=[75,CTRs.mean(),CTRs.std()])
#
#
# x_p = np.linspace(low+((high-low)/(2*num_bins)),high-((high-low)/(2*num_bins)),400)
# T_x = 0.6
# T_y = generateTextY(9,50)
#
# ax1.hist(CTRs, bins=bins,color='b')
# ax1.plot(x_p,gauss(x_p,x1_p[0],x1_p[1],x1_p[2]), color='k')
# ax1.text(T_x,T_y[0],"Data Stats:")
# ax1.text(T_x,T_y[1],"mean: "+format(CTRs.mean(),"7.4f"))
# ax1.text(T_x,T_y[2],"std: "+format(CTRs.std(),"7.4f"))
# ax1.text(T_x,T_y[3],"count: "+str(len(CTRs)))
# ax1.text(T_x,T_y[4],"Fit Paramters:")
# ax1.text(T_x,T_y[5],"mu: "+format(x1_p[1],"7.4f"))
# ax1.text(T_x,T_y[6],"sigma: "+format(x1_p[2],"7.4f"))
# ax1.text(T_x,T_y[7],"A: "+format(x1_p[0],"5.2f"))
# ax1.text(T_x,T_y[8],"FWHM: "+format(2.3548* x1_p[2],"7.4f"))
# plt.show()
#
# # # On a heatmap
# # CTRLT = np.reshape(np.array(CTRLT.CTR), (8,8))
# # CTRLB = np.reshape(np.array(CTRLB.CTR), (8,8))
# # CTRRT = np.reshape(np.array(CTRRT.CTR), (8,8))
# # CTRRB = np.reshape(np.array(CTRRB.CTR), (8,8))
# #
# #
# # plt.subplot(221)
# # plt.xticks(np.arange(8))
# # plt.yticks(np.arange(8))
# # for i in range(8):
# #     for j in range(8):
# #         plt.text(j, i, format(int(CTRLT[i][j]),"3"), ha="center", va="center", color="b")
# # plt.title("Best CTR by channel: \n PCB 6")
# # plt.imshow(CTRLT,cmap=cmap,vmin=150, vmax=320)
# #
# # plt.subplot(222)
# # plt.xticks(np.arange(8))
# # plt.yticks(np.arange(8))
# # for i in range(8):
# #     for j in range(8):
# #         plt.text(j, i, format(int(CTRRT[i][j]),"3"), ha="center", va="center", color="b")
# # plt.title("Best CTR by channel: \n PCB 5")
# # plt.imshow(CTRRT,cmap=cmap,vmin=150, vmax=320)
# #
# # plt.subplot(223)
# # plt.xticks(np.arange(8))
# # plt.yticks(np.arange(8))
# # for i in range(8):
# #     for j in range(8):
# #         plt.text(j, i, format(int(CTRLB[i][j]),"3"), ha="center", va="center", color="b")
# # plt.imshow(CTRLB,cmap=cmap,vmin=150, vmax=320)
# #
# # plt.subplot(224)
# # plt.xticks(np.arange(8))
# # plt.yticks(np.arange(8))
# # for i in range(8):
# #     for j in range(8):
# #         plt.text(j, i, format(int(CTRRB[i][j]),"3"), ha="center", va="center", color="b")
# # plt.imshow(CTRRB,cmap=cmap,vmin=150, vmax=320)
# #
# # plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9,left=0.2)
# # cax = plt.axes([0.85, 0.1, 0.025, 0.8])
# # plt.colorbar(cax=cax)
# # plt.show()



# # ERes Graph
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111,xlim=(0,15), ylim=(0,100),xlabel="Energy Resolution",ylabel="Counts",title="Energy Resolution PCB 8 \n")
#
# jazz = data.E_ResR
# low = 0
# high = low +15 #high = low * -1
# num_bins = 35
# bins = np.linspace(low,high,num_bins)
# x = np.linspace(low+((high-low)/(2*num_bins)),high-((high-low)/(2*num_bins)),num_bins-1)
# y1 = jazz.value_counts(bins=bins,sort=False).tolist()
# x1_p, x1_co = curve_fit(gauss,x,y1, p0=[75,7,1.4])
#
#
# x_p = np.linspace(low+((high-low)/(2*num_bins)),high-((high-low)/(2*num_bins)),400)
# T_x = 0.05
# T_y = generateTextY(9,100)
#
# ax1.hist(jazz, bins=bins,color='b')
# ax1.plot(x_p,gauss(x_p,x1_p[0],x1_p[1],x1_p[2]), color='k')
# ax1.text(T_x,T_y[0],"Data Stats:")
# ax1.text(T_x,T_y[1],"mean: "+format(jazz.mean(),"7.4f"))
# ax1.text(T_x,T_y[2],"std: "+format(jazz.std(),"7.4f"))
# ax1.text(T_x,T_y[3],"count: "+str(len(jazz)))
# ax1.text(T_x,T_y[4],"Fit Paramters:")
# ax1.text(T_x,T_y[5],"mu: "+format(x1_p[1],"5.2f"))
# ax1.text(T_x,T_y[6],"sigma: "+format(abs(x1_p[2]),"7.4f"))
# ax1.text(T_x,T_y[7],"A: "+format(x1_p[0],"5.2f"))
# ax1.text(T_x,T_y[8],"FWHM: "+format(abs(2.3548* x1_p[2]),"7.4f"))
# plt.show()



# # Best ERes by pixel on a heatmap
# EResL = data.groupby("GeoChannelIDL").E_ResL.min().reset_index()
# EResL.columns = ["ChannelIDL", "ERes"]
# #I know there is a better way to do this but this is just a quick method, feel free to suggest an improvment
#
# # #Uncomment for heatmap
# # for i in range(0,128):
# #     run = EResL[EResL["ChannelIDL"] == i]
# #     if run.ChannelIDL.count() == 0:
# #         EResL = EResL.append({"ChannelIDL":i, "ERes":0},ignore_index=True)
#
# EResR = data.groupby("GeoChannelIDR").E_ResR.min().reset_index()
# EResR.columns = ["ChannelIDR", "ERes"]
# # # Uncomment for heatmap
# # for i in range(0,128):
# #     run = EResR[EResR["ChannelIDR"] == i]
# #     if run.ChannelIDR.count() == 0:
# #         EResR = EResR.append({"ChannelIDR":i, "ERes":0},ignore_index=True)
#
# EResL = EResL.sort_values(by="ChannelIDL")
# EResR = EResR.sort_values(by="ChannelIDR")
#
# EResRT = EResR[EResR["ChannelIDR"] < 64].reset_index(drop=True)
# EResRB = EResR[EResR["ChannelIDR"] >= 64].reset_index(drop=True)
# EResLT = EResL[EResL["ChannelIDL"] < 64].reset_index(drop=True)
# EResLB = EResL[EResL["ChannelIDL"] >= 64].reset_index(drop=True)
#
# # on a single histogram
# oof = pd.concat([EResLT, EResLB, EResRT, EResRB], ignore_index=True)
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111,xlim=(0,15), ylim=(0,100),xlabel="Energy Resolution",ylabel="Counts",title="Energy Resolution: Best ERes per pixel")
#
# jazz = oof.ERes
# low = 0
# high = low +15 #high = low * -1
# num_bins = 35
# bins = np.linspace(low,high,num_bins)
# x = np.linspace(low+((high-low)/(2*num_bins)),high-((high-low)/(2*num_bins)),num_bins-1)
# y1 = jazz.value_counts(bins=bins,sort=False).tolist()
# x1_p, x1_co = curve_fit(gauss,x,y1, p0=[75,jazz.mean(),jazz.std()])
#
#
# x_p = np.linspace(low+((high-low)/(2*num_bins)),high-((high-low)/(2*num_bins)),400)
# T_x = 0.05
# T_y = generateTextY(9,100)
#
# ax1.hist(jazz, bins=bins,color='b')
# ax1.plot(x_p,gauss(x_p,x1_p[0],x1_p[1],x1_p[2]), color='k')
# ax1.text(T_x,T_y[0],"jazz Stats:")
# ax1.text(T_x,T_y[1],"mean: "+format(jazz.mean(),"7.4f"))
# ax1.text(T_x,T_y[2],"std: "+format(jazz.std(),"7.4f"))
# ax1.text(T_x,T_y[3],"count: "+str(len(jazz)))
# ax1.text(T_x,T_y[4],"Fit Paramters:")
# ax1.text(T_x,T_y[5],"mu: "+format(x1_p[1],"5.2f"))
# ax1.text(T_x,T_y[6],"sigma: "+format(x1_p[2],"7.4f"))
# ax1.text(T_x,T_y[7],"A: "+format(x1_p[0],"5.2f"))
# ax1.text(T_x,T_y[8],"FWHM: "+format(2.3548* x1_p[2],"7.4f"))
# plt.show()
#
#
# # on a histogram for each array
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(221,xlim=(0,15), ylim=(0,100),xlabel="ERes",ylabel="Counts",title="Best ERes PCB 6")
# ax2 = fig1.add_subplot(222,xlim=(0,15), ylim=(0,100),xlabel="ERes",ylabel="Counts",title="Best ERes PCB 8")
# ax3 = fig1.add_subplot(223,xlim=(0,15), ylim=(0,100),xlabel="ERes",ylabel="Counts")
# ax4 = fig1.add_subplot(224,xlim=(0,15), ylim=(0,100),xlabel="ERes",ylabel="Counts")
#
# low = 0
# high = low +15 #high = low * -1
# num_bins = 35
# bins = np.linspace(low,high,num_bins)
# x = np.linspace(low+((high-low)/(2*num_bins)),high-((high-low)/(2*num_bins)),num_bins-1)
# y1 = EResLT.ERes.value_counts(bins=bins,sort=False).tolist()
# y2 = EResRT.ERes.value_counts(bins=bins,sort=False).tolist()
# y3 = EResLB.ERes.value_counts(bins=bins,sort=False).tolist()
# y4 = EResRB.ERes.value_counts(bins=bins,sort=False).tolist()
# x1_p, x1_co = curve_fit(gauss,x,y1, p0=[75,EResLT.ERes.mean(),EResLT.ERes.std()])
# x2_p, x2_co = curve_fit(gauss,x,y2, p0=[75,EResRT.ERes.mean(),EResRT.ERes.std()])
# x3_p, x3_co = curve_fit(gauss,x,y3, p0=[75,EResLB.ERes.mean(),EResLB.ERes.std()])
# x4_p, x4_co = curve_fit(gauss,x,y4, p0=[75,EResRB.ERes.mean(),EResRB.ERes.std()])
#
#
# x_p = np.linspace(low+((high-low)/(2*num_bins)),high-((high-low)/(2*num_bins)),400)
# T_x = 0.6
# T_y = generateTextY(9,100)
#
# ax1.hist(EResLT.ERes, bins=bins,color='b')
# ax1.plot(x_p,gauss(x_p,x1_p[0],x1_p[1],x1_p[2]), color='k')
# ax1.text(T_x,T_y[0],"Data Stats:")
# ax1.text(T_x,T_y[1],"mean: "+format(EResLT.ERes.mean(),"7.4f"))
# ax1.text(T_x,T_y[2],"std: "+format(EResLT.ERes.std(),"7.4f"))
# ax1.text(T_x,T_y[3],"count: "+str(EResLT.ERes.count()))
# ax1.text(T_x,T_y[4],"Fit Paramters:")
# ax1.text(T_x,T_y[5],"mu: "+format(x1_p[1],"7.4f"))
# ax1.text(T_x,T_y[6],"sigma: "+format(x1_p[2],"7.4f"))
# ax1.text(T_x,T_y[7],"A: "+format(x1_p[0],"5.2f"))
# ax1.text(T_x,T_y[8],"FWHM: "+format(2.3548* x1_p[2],"7.4f"))
#
# ax2.hist(EResRT.ERes, bins=bins,color='b')
# ax2.plot(x_p,gauss(x_p,x2_p[0],x2_p[1],x2_p[2]), color='k')
# ax2.text(T_x,T_y[0],"Data Stats:")
# ax2.text(T_x,T_y[1],"mean: "+format(EResRT.ERes.mean(),"7.4f"))
# ax2.text(T_x,T_y[2],"std: "+format(EResRT.ERes.std(),"7.4f"))
# ax2.text(T_x,T_y[3],"count: "+str(EResRT.ERes.count()))
# ax2.text(T_x,T_y[4],"Fit Paramters:")
# ax2.text(T_x,T_y[5],"mu: "+format(x2_p[1],"7.4f"))
# ax2.text(T_x,T_y[6],"sigma: "+format(x2_p[2],"7.4f"))
# ax2.text(T_x,T_y[7],"A: "+format(x2_p[0],"5.2f"))
# ax2.text(T_x,T_y[8],"FWHM: "+format(2.3548* x2_p[2],"7.4f"))
#
# ax3.hist(EResLB.ERes, bins=bins,color='b')
# ax3.plot(x_p,gauss(x_p,x3_p[0],x3_p[1],x3_p[2]), color='k')
# ax3.text(T_x,T_y[0],"Data Stats:")
# ax3.text(T_x,T_y[1],"mean: "+format(EResLB.ERes.mean(),"7.4f"))
# ax3.text(T_x,T_y[2],"std: "+format(EResLB.ERes.std(),"7.4f"))
# ax3.text(T_x,T_y[3],"count: "+str(EResLB.ERes.count()))
# ax3.text(T_x,T_y[4],"Fit Paramters:")
# ax3.text(T_x,T_y[5],"mu: "+format(x3_p[1],"7.4f"))
# ax3.text(T_x,T_y[6],"sigma: "+format(x3_p[2],"7.4f"))
# ax3.text(T_x,T_y[7],"A: "+format(x3_p[0],"5.2f"))
# ax3.text(T_x,T_y[8],"FWHM: "+format(2.3548* x3_p[2],"7.4f"))
#
# ax4.hist(EResRB.ERes, bins=bins,color='b')
# ax4.plot(x_p,gauss(x_p,x4_p[0],x4_p[1],x4_p[2]), color='k')
# ax4.text(T_x,T_y[0],"Data Stats:")
# ax4.text(T_x,T_y[1],"mean: "+format(EResRB.ERes.mean(),"7.4f"))
# ax4.text(T_x,T_y[2],"std: "+format(EResRB.ERes.std(),"7.4f"))
# ax4.text(T_x,T_y[3],"count: "+str(EResRB.ERes.count()))
# ax4.text(T_x,T_y[4],"Fit Paramters:")
# ax4.text(T_x,T_y[5],"mu: "+format(x4_p[1],"7.4f"))
# ax4.text(T_x,T_y[6],"sigma: "+format(x4_p[2],"7.4f"))
# ax4.text(T_x,T_y[7],"A: "+format(x4_p[0],"5.2f"))
# ax4.text(T_x,T_y[8],"FWHM: "+format(2.3548* x4_p[2],"7.4f"))
#
# plt.show()
#
# # # as a heatmap
# # EResLT = np.reshape(np.array(EResLT.ERes), (8,8))
# # EResLB = np.reshape(np.array(EResLB.ERes), (8,8))
# # EResRT = np.reshape(np.array(EResRT.ERes), (8,8))
# # EResRB = np.reshape(np.array(EResRB.ERes), (8,8))
# #
# # plt.subplot(221)
# # plt.xticks(np.arange(8))
# # plt.yticks(np.arange(8))
# # for i in range(8):
# #     for j in range(8):
# #         plt.text(j, i, format(EResLT[i][j],".1f"), ha="center", va="center", color="b")
# # plt.title("Best ERes by channel: \n PCB 6")
# # plt.imshow(EResLT,cmap=cmap,vmin=0, vmax=18.0)
# #
# # plt.subplot(222)
# # plt.xticks(np.arange(8))
# # plt.yticks(np.arange(8))
# # for i in range(8):
# #     for j in range(8):
# #         plt.text(j, i, format(EResRT[i][j],".1f"), ha="center", va="center", color="b")
# # plt.title("Best ERes by channel: \n PCB 8")
# # plt.imshow(EResRT,cmap=cmap,vmin=0, vmax=18.0)
# #
# # plt.subplot(223)
# # plt.xticks(np.arange(8))
# # plt.yticks(np.arange(8))
# # for i in range(8):
# #     for j in range(8):
# #         plt.text(j, i, format(EResLB[i][j],".1f"), ha="center", va="center", color="b")
# # plt.imshow(EResLB,cmap=cmap,vmin=0, vmax=18.0)
# #
# # plt.subplot(224)
# # plt.xticks(np.arange(8))
# # plt.yticks(np.arange(8))
# # for i in range(8):
# #     for j in range(8):
# #         plt.text(j, i, format(EResRB[i][j],".1f"), ha="center", va="center", color="b")
# # plt.imshow(EResRB,cmap=cmap,vmin=0, vmax=18.0)
# #
# # plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9,left=0.2)
# # cax = plt.axes([0.85, 0.1, 0.025, 0.8])
# # plt.colorbar(cax=cax)
# # plt.show()



#
# # #Pixel map of coincident channel pairs
# # for i in range(len(a)):
# #     fig = plt.figure()
# #     ax = fig.add_subplot(221)
# #     ax1 = fig.add_subplot(222)
# #     ax2 = fig.add_subplot(223)
# #     ax3 = fig.add_subplot(224)
# #     ax.set_title("PCB 6")
# #     ax1.set_title("PCB 5")
# #     for k in range(0,9):
# #         ax.hlines(k,0,8)
# #     for j in range(0,9):
# #         ax.vlines(j,0,8)
# #     if a[i] < 64:
# #         ax.text(indices.get(a[i])[1]+0.05,8-indices.get(a[i])[0]-0.7, str(tot_phot[i]))
# #
# #     for k in range(0,9):
# #         ax2.hlines(k,0,8)
# #     for j in range(0,9):
# #         ax2.vlines(j,0,8)
# #     if a[i] >= 64:
# #         ax2.text(indices.get(a[i])[1]+0.05,16-indices.get(a[i])[0]-0.7, str(tot_phot[i]))
# #
# #     for k in range(0,9):
# #         ax1.hlines(k,0,8)
# #     for j in range(0,9):
# #         ax1.vlines(j,0,8)
# #     if b[i] < 64:
# #         ax1.text(indices.get(b[i])[1]+0.05,8-indices.get(b[i])[0]-0.7, str(tot_phot[i]))
# #
# #     for k in range(0,9):
# #         ax3.hlines(k,0,8)
# #     for j in range(0,9):
# #         ax3.vlines(j,0,8)
# #     if b[i] >= 64:
# #         ax3.text(indices.get(b[i])[1]+0.05,16-indices.get(b[i])[0]-0.7, str(tot_phot[i]))
# #     plt.savefig("Coincidence_Pair_{}.png".format(i))
# #     plt.clf()
# #     plt.close()



# #Coincidence Channel Sums
# PP_P = pd.DataFrame()
# # THIS IS NOT A MISTAKE THE TEXT PLACEMENT IS LEGACY CODE THAT REQUIRES PETSYS CHANNEL ID
# PP_P["ChannelIDL"] = data["ChannelIDL"]
# PP_P["ChannelIDR"] = data["ChannelIDR"]
# PP_P["Count"] = data["PP_Count"]
# PP_L = PP_P.groupby("ChannelIDL").Count.sum().reset_index()
# PP_L.columns = ["ChannelIDL", "Counts"]
# PP_R = PP_P.groupby("ChannelIDR").Count.sum().reset_index()
# PP_R.columns = ["ChannelIDR", "Counts"]
#
#
# fig = plt.figure()
# ax = fig.add_subplot(221)
# ax1 = fig.add_subplot(222)
# ax2 = fig.add_subplot(223)
# ax3 = fig.add_subplot(224)
# ax.set_title("PCB 6")
# ax1.set_title("PCB 5")
#
#
# for k in range(0,9):
#     ax.hlines(k,0,8)
#     ax1.hlines(k,0,8)
#     ax2.hlines(k,0,8)
#     ax3.hlines(k,0,8)
# for j in range(0,9):
#     ax.vlines(j,0,8)
#     ax1.vlines(j,0,8)
#     ax2.vlines(j,0,8)
#     ax3.vlines(j,0,8)
#
# for i in range(PP_L.ChannelIDL.count()):
#     if PP_L.ChannelIDL[i] < 64:
#         ax.text(indices.get(PP_L.ChannelIDL[i])[1]+0.1,8-indices.get(PP_L.ChannelIDL[i])[0]-0.6, str(PP_L.Counts[i]))
#     elif PP_L.ChannelIDL[i] >= 64:
#         ax2.text(indices.get(PP_L.ChannelIDL[i])[1]+0.1,16-indices.get(PP_L.ChannelIDL[i])[0]-0.6, str(PP_L.Counts[i]))
#
# for i in range(PP_R.ChannelIDR.count()):
#     if PP_R.ChannelIDR[i] < 64:
#         ax1.text(indices.get(PP_R.ChannelIDR[i])[1]+0.1,8-indices.get(PP_R.ChannelIDR[i])[0]-0.6, str(PP_R.Counts[i]))
#     if PP_R.ChannelIDR[i] >= 64:
#         ax3.text(indices.get(PP_R.ChannelIDR[i])[1]+0.1,16-indices.get(PP_R.ChannelIDR[i])[0]-0.6, str(PP_R.Counts[i]))
#
# for i in range(df.ChannelIDL.count()):
#     if df.E_Cut[i] == 0:
#         if df.ChannelIDL[i] < 64:
#             ax.plot(indices.get(df.ChannelIDL[i])[1]+0.1,8-indices.get(df.ChannelIDL[i])[0]-0.6,'bs')
#         elif df.ChannelIDL[i] >= 64:
#             ax2.plot(indices.get(df.ChannelIDL[i])[1]+0.1,16-indices.get(df.ChannelIDL[i])[0]-0.6,'bs')
#     if df.PP_Cut[i] == 0:
#         if df.ChannelIDL[i] < 64:
#             ax.plot(indices.get(df.ChannelIDL[i])[1]+0.1,8-indices.get(df.ChannelIDL[i])[0]-0.6,'rs')
#         elif df.ChannelIDL[i] >= 64:
#             ax2.plot(indices.get(df.ChannelIDL[i])[1]+0.1,16-indices.get(df.ChannelIDL[i])[0]-0.6,'rs')
# #its not finished yet,don't yell at me
# for i in range(df.ChannelIDL.count()):
#     if df.E_Cut[i] == 0:
#         if df.ChannelIDR[i] < 64:
#             ax1.plot(indices.get(df.ChannelIDR[i])[1]+0.1,8-indices.get(df.ChannelIDR[i])[0]-0.6,'bs')
#         elif df.ChannelIDR[i] >= 64:
#             ax3.plot(indices.get(df.ChannelIDR[i])[1]+0.1,16-indices.get(df.ChannelIDR[i])[0]-0.6,'bs')
#     if df.PP_Cut[i] == 0:
#         if df.ChannelIDR[i] < 64:
#             ax1.plot(indices.get(df.ChannelIDR[i])[1]+0.1,8-indices.get(df.ChannelIDR[i])[0]-0.6,'rs')
#         elif df.ChannelIDR[i] >= 64:
#             ax3.plot(indices.get(df.ChannelIDR[i])[1]+0.1,16-indices.get(df.ChannelIDR[i])[0]-0.6,'rs')
#
# plt.show()
# # plt.savefig("Coincidence_Sums.png")
# plt.clf()
# plt.close()
#

# #
# #Charge Integration
# fig = plt.figure(figsize=(12,8))
# ax = fig.add_subplot(211)
# ax1 = fig.add_subplot(212)
# ax.set_title("Run "+str(run_num)+": PCB 6 Charge Integral by Pixel")
# ax1.set_title("Run "+str(run_num)+": PCB 8 Charge Integral by Pixel")
# ax.set_ylim(0,6000)
# ax1.set_ylim(0,6000)
# #ax.scatter(E_Int.ChannelID, E_Int.Charge_INTR)
# #ax1.scatter(E_Int.ChannelID, E_Int.Charge_INTL)
# dataCI = df.groupby("GeoChannelIDL").QINTL.sum().reset_index()
# ax.step(dataCI["GeoChannelIDL"], dataCI["QINTL"], where='mid')
# dataCI = df.groupby("GeoChannelIDR").QINTR.sum().reset_index()
# ax1.step(dataCI["GeoChannelIDR"], dataCI["QINTR"], where='mid')
# plt.subplots_adjust(hspace=0.5)
# plt.show()


#
#
#
# # plotting the photopeak counts
# T_y = [9000, 6000,2000]
# plt.hist(df.PP_Count, bins=np.linspace(0,1000,50))
# plt.title("Number of Coincident Photoelectric Events events in a channel pair: \n ")
# plt.xlabel("Number of Coincident Events")
# plt.ylabel("Counts")
# plt.yscale("log")
# # plt.xscale("log")
# plt.xlim(0,1000)
# #first bin goes to 13500
# plt.ylim(1,10000)
# T_x = 600
# T_y = generateTextY(3,1000)
# plt.text(T_x,T_y[0], "Mean: "+format(df.PP_Count.mean(), "6.2f"))
# plt.text(T_x,T_y[1], "Std: "+format(df.PP_Count.std(), "5.2f"))
# plt.text(T_x,T_y[2], "Count: "+format(df.PP_Count.count(), "6"))
# plt.show()




# fig = plt.figure()
# ax = fig.add_subplot(221)
# ax1 = fig.add_subplot(222)
# ax2 = fig.add_subplot(223)
# ax3 = fig.add_subplot(224)
# ax.set_title("PCB 6")
# ax1.set_title("PCB 5")
#
# for k in range(0,9):
#     ax.hlines(k,0,8)
#     ax1.hlines(k,0,8)
#     ax2.hlines(k,0,8)
#     ax3.hlines(k,0,8)
# for j in range(0,9):
#     ax.vlines(j,0,8)
#     ax1.vlines(j,0,8)
#     ax2.vlines(j,0,8)
#     ax3.vlines(j,0,8)
#
# for i in range(LQ.Quality.count()):
#     if LQ.GeoChannelIDL[i] < 64:
#         # ax.text((LQ.GeoChannelIDL[i]%8)+0.2,8-(LQ.GeoChannelIDL[i]//8)-0.6, str(LQ.Quality[i]))
#         if LQ.Quality[i] == "A":
#             ax.plot((LQ.GeoChannelIDL[i]%8)+0.4,8-(LQ.GeoChannelIDL[i]//8)-0.6,'gs', ms=10)
#         elif LQ.Quality[i] == "B":
#             ax.plot((LQ.GeoChannelIDL[i]%8)+0.4,8-(LQ.GeoChannelIDL[i]//8)-0.6,'bs', ms=10)
#         elif LQ.Quality[i] == "C":
#             ax.plot((LQ.GeoChannelIDL[i]%8)+0.4,8-(LQ.GeoChannelIDL[i]//8)-0.6,'ys', ms=10)
#         elif LQ.Quality[i] == "D":
#             ax.plot((LQ.GeoChannelIDL[i]%8)+0.4,8-(LQ.GeoChannelIDL[i]//8)-0.6,'rs', ms=10)
#
#
#     elif LQ.GeoChannelIDL[i] >= 64:
#         # ax2.text((LQ.GeoChannelIDL[i]%8)+0.4,16-(LQ.GeoChannelIDL[i]//8)-0.6, str(LQ.Quality[i]))
#         if LQ.Quality[i] == "A":
#             ax2.plot((LQ.GeoChannelIDL[i]%8)+0.4,16-(LQ.GeoChannelIDL[i]//8)-0.6,'gs', ms=10)
#         elif LQ.Quality[i] == "B":
#             ax2.plot((LQ.GeoChannelIDL[i]%8)+0.4,16-(LQ.GeoChannelIDL[i]//8)-0.6,'bs', ms=10)
#         elif LQ.Quality[i] == "C":
#             ax2.plot((LQ.GeoChannelIDL[i]%8)+0.4,16-(LQ.GeoChannelIDL[i]//8)-0.6,'ys', ms=10)
#         elif LQ.Quality[i] == "D":
#             ax2.plot((LQ.GeoChannelIDL[i]%8)+0.4,16-(LQ.GeoChannelIDL[i]//8)-0.6,'rs', ms=10)
#
# for i in range(RQ.Quality.count()):
#     if RQ.GeoChannelIDR[i] < 64:
#         # ax1.text((RQ.GeoChannelIDR[i]%8)+0.4,8-(RQ.GeoChannelIDR[i]//8)-0.6, str(RQ.Quality[i]))
#         if RQ.Quality[i] == "A":
#             ax1.plot((RQ.GeoChannelIDR[i]%8)+0.4,8-(RQ.GeoChannelIDR[i]//8)-0.6,'gs', ms=10)
#         elif RQ.Quality[i] == "B":
#             ax1.plot((RQ.GeoChannelIDR[i]%8)+0.4,8-(RQ.GeoChannelIDR[i]//8)-0.6,'bs', ms=10)
#         elif RQ.Quality[i] == "C":
#             ax1.plot((RQ.GeoChannelIDR[i]%8)+0.4,8-(RQ.GeoChannelIDR[i]//8)-0.6,'ys', ms=10)
#         elif RQ.Quality[i] == "D":
#             ax1.plot((RQ.GeoChannelIDR[i]%8)+0.4,8-(RQ.GeoChannelIDR[i]//8)-0.6,'rs', ms=10)
#
#     elif RQ.GeoChannelIDR[i] >= 64:
#         # ax3.text((RQ.GeoChannelIDR[i]%8)+0.4,16-(RQ.GeoChannelIDR[i]//8)-0.6, str(RQ.Quality[i]))
#         if RQ.Quality[i] == "A":
#             ax3.plot((RQ.GeoChannelIDR[i]%8)+0.4,16-(RQ.GeoChannelIDR[i]//8)-0.6,'gs', ms=10)
#         elif RQ.Quality[i] == "B":
#             ax3.plot((RQ.GeoChannelIDR[i]%8)+0.4,16-(RQ.GeoChannelIDR[i]//8)-0.6,'bs', ms=10)
#         elif RQ.Quality[i] == "C":
#             ax3.plot((RQ.GeoChannelIDR[i]%8)+0.4,16-(RQ.GeoChannelIDR[i]//8)-0.6,'ys', ms=10)
#         elif RQ.Quality[i] == "D":
#             ax3.plot((RQ.GeoChannelIDR[i]%8)+0.4,16-(RQ.GeoChannelIDR[i]//8)-0.6,'rs', ms=10)
#
# plt.show()
# # plt.savefig("Coincidence_Sums.png")
# plt.clf()
# plt.close()

# plt.scatter(data.PP_Count,data.CTR)
# plt.title("CTR vs Number of Photopeak events in a Channel Pair")
# plt.ylabel("CTR in ns")
# plt.xlabel("Photopeak Counts")
# plt.show()
