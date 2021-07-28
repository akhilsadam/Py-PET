import numpy as np
import pandas as pd
import math
from scipy.optimize import curve_fit
import statistics as stat

class Fit:
    # default parameters
    binwidth_E = 0.5
    binwidth_T = 0.01
    limit = np.array([[0,15],[0,0.4],[0,1]])
    WTH_D = 1 # default peak-finding width
    LIN_P = [0,0.0001]
    LIN_BOUNDS = ([-np.inf,-np.inf],[np.inf,np.inf])
    STDY = 2
    MINLENGTH = 4
    
    @staticmethod
    def bins(x,binwidth):
        rangex = max(x)-min(x)
        if(rangex<binwidth):
            return 1
        return int(rangex/binwidth)
            
    
    @staticmethod
    def limitSize(limits,length):
        if(limits[0] < 0):
            limits[0]=0
        if(limits[1]>length):
            limits[1]=length
        return limits
    
    @staticmethod
    def limitRange(limits,values):
        for i in range(len(values)):
            if((limits[0][i]>values[i]) or (limits[1][i]<values[i])):
                return False
        return True
    
    @staticmethod
    def weight_stat(values, weights):
        """
        Return the weighted average and standard deviation.
        values, weights -- Numpy ndarrays with the same shape.
        """
        average = np.average(values, weights=weights)
        variance = np.average((values-average)**2, weights=weights)
        variance = variance*(np.sum(weights)/(np.sum(weights)-1)) # Bessel's correction (unbiased variance)
        return (average, math.sqrt(variance)) # note that this is still biased standard deviation.
    
    @staticmethod
    def RSQ(obs,exp):
        sig = stat.stdev(exp)**2
        return (np.sum((obs-exp)**2)/sig)

    @staticmethod
    def line(x, *param):
        p = [1,0] # slope, intercept
        p[0:len(param)] = param
        return np.array(x*p[0]+p[1])
    
    @staticmethod
    def gaussian(x, *param):
        p = [1,0,1] # amplitude, mean, standard deviation
        p[0:len(param)] = param
        return p[0]*np.exp(-(x-p[1])**2/(2.*p[2]**2))
    
    @staticmethod
    def linearFit(x,y,**kwargs):
        try:
            lin_bounds=kwargs.get('LIN_BOUNDS',Fit.LIN_BOUNDS)
            n=len(x)
            slope = (np.dot(x,y)-(1/n)*(np.sum(x)*np.sum(y)))/(np.dot(x,x) - (1/n)*math.pow(np.sum(x),2))
            intercept = (np.sum(y)*np.dot(x,x) - np.sum(x)*np.dot(x,y))/(np.dot(x,x)-math.pow(np.sum(x),2))       
            p0 = [slope,intercept]
            if not Fit.limitRange(lin_bounds,p0):
                p0 = Fit.LIN_P
            p, cov = curve_fit(Fit.line,x,y,p0=p0,bounds=LIN_BOUNDS)
        except:
            print("FIT Failed")
            return np.array(p0)
            print(p0)
        else:
            return np.array(p)
     
    @staticmethod
    def __peakFilter(x,y,peaks,**kwargs):
        # default
        WTH=kwargs.get('WTH',Fit.WTH_D) # width multiplier for peak finding (note that default is global call)
        WTH=kwargs.get('width',WTH) # now a local call
        
        _peaks = []
        for peak in peaks:
            d = peak[1]-peak[0]
            sig = (WTH-1)/2
            st = int(peak[0]-sig*d)
            if(st<0):
                st = 0
            ed = int(peak[1]+sig*d)
            if(ed>=len(x)):
                ed = len(x)-1
            _peaks.append([st,ed])
            
        _peaks = np.array(_peaks) 
        
        clean=np.zeros(len(x))
        signal=np.zeros(len(x))
        for peak in _peaks:
            signal[peak[0]:peak[1]] = y[peak[0]:peak[1]]
        clean = y-signal
            
        return clean,signal,_peaks
    
    @staticmethod
    def peaks(x,y,**kwargs):
        # default
        STDY=kwargs.get('STDY',Fit.STDY) # standard deviation cut for peak finding
        MINLENGTH=kwargs.get('MINLENGTH',Fit.MINLENGTH)
            
        σ = stat.stdev(y)
        TRHLD = STDY*σ
        _clean = np.zeros(len(x))
        for f in range(0,len(x)):
            if((y[f]<TRHLD) and (y[f]>-TRHLD)):
                _clean[f]=y[f]
        _peak_data = [] # [[init, final x],[//,//]..]
        
        st = 0;
        collect = False
        for f in range(0,len(x)-1):
            if(_clean[f]==0 and _clean[f+1]!=0):
                _peak_data.append([st,f])
                collect = True
            if(_clean[f]!=0 and _clean[f+1]==0):
                st = f
        j=0
        for i in range(0,len(_peak_data)):
            length = (_peak_data[j][1] - _peak_data[j][0])
            if(length<=MINLENGTH):
                _peak_data.pop(j)
                j-=1
            j+=1
        _clean = np.array(_clean)
        _signal = np.array(y - _clean)
        clean, signal, _peak_data = Fit.__peakFilter(x,y,_peak_data,**kwargs)
        return  clean, signal, _peak_data # background, signal, signal in array format
    @staticmethod
    def getPeak(x,y,peak):
        return x[peak[0]:peak[1]],y[peak[0]:peak[1]]
    
    @staticmethod
    def lastPeak(x,y,_peak_data):
        return Fit.getPeak(x,y,_peak_data[len(_peak_data)-1]) 
    
    @staticmethod
    def gaussianFit(x_peak,y_peak,x,y,**kwargs):
        # default
        STDFIT=kwargs.get('STDFIT',3) # standard deviation range for peak finding
                
        try:
            mean,std = Fit.weight_stat(x_peak,y_peak)
            if (mean is np.nan or std is np.nan):
                mean = np.median(x)
                std = 1
            p0 = np.array([np.max(y),mean,std])
            p, cov = curve_fit(Fit.gaussian,x_peak,y_peak,p0=p0,maxfev=100000)
        except:
            print(p0)
            print("FIT Failed")
        else:
            try:
                rf = Fit.limitSize([int(p[1]-STDFIT*p[2]),int(p[1]+STDFIT*p[2])],len(x))
                
                xs = x[rf[0]:rf[1]]
                ys = y[rf[0]:rf[1]]
                p2, cov = curve_fit(Fit.gaussian,x,y,p0=p)
            except:
                print("FIT2 Failed")
                return p
            else:
                return p2
    @staticmethod
    def removeBKG(x,y,bkg,**kwargs):
        try:
            indx = np.array([i for i in range(0,len(x)) if bkg[i]!=0]).astype(int)
            xs=x[indx]
            bs=bkg[indx]
            p = Fit.linearFit(xs,bs,**kwargs)
            rsq = Fit.RSQ(y,Fit.line(x,p[0],p[1]))
            ys = y - Fit.line(x,p[0],p[1])
        except E:
            print("Background Removal Failed")
            return y, None, None
        else:
            return ys, p, rsq
    # fit all gaussians in data with linear background (last fit is background)
    @staticmethod
    def lazyPeaks(x,y,**kwargs):
        # default
        MAXITER=kwargs.get('MAXITER',10) # number of width iterations for peak/background seperation
        RSQLIMIT=kwargs.get('RSQLIMIT',9000) # if RSQ is lower than this limit, then we have succeeded
        
        width=kwargs.get('WTH',Fit.WTH_D)

        for i in tqdm(range(MAXITER)):
            clean,signal,peak_data = Fit.peaks(x,y,width=width,**kwargs)
            ys, p, rsq = Fit.removeBKG(x,y,clean,**kwargs)
            if rsq < RSQLIMIT:
                break
            else:
                width=width*1.5
                    
        return clean,signal,peak_data,ys,p
    @staticmethod
    def lazyGaussianFit(x,y,**kwargs):
        USEBKG=kwargs.get('USEBKG',False)
        if USEBKG:
            clean,signal,peak_data,ys,p = Fit.lazyPeaks(x,y,**kwargs)
        else:
            clean,signal,peak_data = Fit.peaks(x,y,**kwargs)
            ys=y
            p=None
        fits=[]
        for peak in peak_data:
            xsz,ysz=Fit.getPeak(x,ys,peak)
            fits.append(Fit.gaussianFit(xsz,ysz,x,ys,**kwargs))
        return fits,p   
    @staticmethod
    def lastGaussianFit(x,y,**kwargs):
        USEBKG=kwargs.get('USEBKG',False)
        if USEBKG:
            clean,signal,peak_data,ys,p = Fit.lazyPeaks(x,y,**kwargs)
        else:
            clean,signal,peak_data = Fit.peaks(x,y,**kwargs)
            ys=y
            p=None
        xsz,ysz=Fit.lastPeak(x,ys,peak_data)
        return Fit.gaussianFit(xsz,ysz,x,ys,**kwargs),p   
    @staticmethod
    def __energyResolution(x,**kwargs):
        binwidth=kwargs.get("Display_Binwidth",Fit.binwidth_E)
        bins=Fit.bins(x,binwidth)
        values, binedges = np.histogram(x,bins=bins)
        bin_centers = np.mean(np.vstack([binedges[0:-1],binedges[1:]]), axis=0)
        xc = bin_centers*binwidth
        # actual fit binning:
        binwidth2=kwargs.get("Binwidth",Fit.binwidth_E)
        bins2=Fit.bins(x,binwidth2)
        values2, binedges2 = np.histogram(x,bins=bins2)
        bin_centers2 = np.mean(np.vstack([binedges2[0:-1],binedges2[1:]]), axis=0)
        x2 = bin_centers2*binwidth2
        fit,p = Fit.lastGaussianFit(x2,values2,**kwargs)
        return xc,values,fit,p
    @staticmethod
    def __timeResolution(x,**kwargs):
        binwidth=kwargs.get("Display_Binwidth",Fit.binwidth_T)
        bins=Fit.bins(x,binwidth)
        values, binedges = np.histogram(x,bins=bins)
        bin_centers = np.mean(np.vstack([binedges[0:-1],binedges[1:]]), axis=0)
        xc = bin_centers*binwidth
        # actual fit binning:
        binwidth2=kwargs.get("Binwidth",Fit.binwidth_T)
        bins2=Fit.bins(x,binwidth2)
        values2, binedges2 = np.histogram(x,bins=bins2)
        bin_centers2 = np.mean(np.vstack([binedges2[0:-1],binedges2[1:]]), axis=0)
        x2 = bin_centers2*binwidth2
        fit = Fit.gaussianFit(x2,values2,x2,values2,**kwargs)
        return xc,values,fit
    @staticmethod
    def energyResolution(x,**kwargs):
        return Fit.__energyResolution(x,MINLENGTH=0,WTH=20,**kwargs)
        
    @staticmethod
    def timeResolution(x,**kwargs):
        return Fit.__timeResolution(x,MINLENGTH=0,WTH=1,**kwargs)