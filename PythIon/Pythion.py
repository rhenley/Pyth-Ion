#!/usr/bin/python
# -*- coding: utf8 -*-
import sys
import numpy as np
from scipy import ndimage
import os
from scipy import signal
from scipy import io as spio
#from PlotGUI import *
#from plotgui4k import *
from plotguiretina import *
import pyqtgraph as pg
import pandas.io.parsers
import pandas as pd
from abfheader import *
from CUSUMV2 import detect_cusum
from PoreSizer import *
from batchinfo import *



class GUIForm(QtGui.QMainWindow):


    def __init__(self, master=None):
        ####Setup GUI and draw elements from UI file#########
        QtGui.QMainWindow.__init__(self,master)
        self.ui = Ui_PythIon()
        self.ui.setupUi(self)

        ##########Linking buttons to main functions############
        QtCore.QObject.connect(self.ui.loadbutton, QtCore.SIGNAL('clicked()'), self.getfile)
        QtCore.QObject.connect(self.ui.analyzebutton, QtCore.SIGNAL('clicked()'), self.analyze)
        QtCore.QObject.connect(self.ui.cutbutton, QtCore.SIGNAL('clicked()'), self.cut)
        QtCore.QObject.connect(self.ui.baselinebutton, QtCore.SIGNAL('clicked()'), self.baselinecalc)
        QtCore.QObject.connect(self.ui.clearscatterbutton, QtCore.SIGNAL('clicked()'), self.clearscatter)
        QtCore.QObject.connect(self.ui.deleteeventbutton, QtCore.SIGNAL('clicked()'), self.deleteevent)
        QtCore.QObject.connect(self.ui.invertbutton, QtCore.SIGNAL('clicked()'), self.invertdata)
        QtCore.QObject.connect(self.ui.concatenatebutton, QtCore.SIGNAL('clicked()'), self.concatenatetext)
        QtCore.QObject.connect(self.ui.nextfilebutton, QtCore.SIGNAL('clicked()'), self.nextfile)
        QtCore.QObject.connect(self.ui.previousfilebutton, QtCore.SIGNAL('clicked()'), self.previousfile)
        QtCore.QObject.connect(self.ui.savetargetbutton, QtCore.SIGNAL('clicked()'), self.savetarget)
        QtCore.QObject.connect(self.ui.showcatbutton, QtCore.SIGNAL('clicked()'), self.showcattrace)
        QtCore.QObject.connect(self.ui.savecatbutton, QtCore.SIGNAL('clicked()'), self.savecattrace)
        QtCore.QObject.connect(self.ui.gobutton, QtCore.SIGNAL('clicked()'), self.inspectevent)
        QtCore.QObject.connect(self.ui.previousbutton, QtCore.SIGNAL('clicked()'), self.previousevent)
        QtCore.QObject.connect(self.ui.nextbutton, QtCore.SIGNAL('clicked()'), self.nextevent)
        QtCore.QObject.connect(self.ui.savefitsbutton, QtCore.SIGNAL('clicked()'), self.saveeventfits)
        QtCore.QObject.connect(self.ui.fitbutton, QtCore.SIGNAL('clicked()'), self.CUSUM)
        QtCore.QObject.connect(self.ui.Poresizeraction, QtCore.SIGNAL('triggered()'), self.sizethepore)
        QtCore.QObject.connect(self.ui.actionBatch_Process, QtCore.SIGNAL('triggered()'), self.batchinfodialog)



        ###### Setting up plotting elements and their respective options######
        self.ui.signalplot.setBackground('w')
        self.ui.scatterplot.setBackground('w')
        self.ui.eventplot.setBackground('w')
        self.ui.frachistplot.setBackground('w')
        self.ui.delihistplot.setBackground('w')
        self.ui.dwellhistplot.setBackground('w')
        self.ui.dthistplot.setBackground('w')
#        self.ui.PSDplot.setBackground('w')

        self.p1 = self.ui.signalplot.addPlot()
        self.p1.setLabel('bottom', text='Time', units='s')
        self.p1.setLabel('left', text='Current', units='A')
        self.p1.enableAutoRange(axis = 'x')
        self.p1.setDownsampling(ds=True, auto=True, mode='peak')


        self.w1 = self.ui.scatterplot.addPlot()
        self.p2 = pg.ScatterPlotItem()
        self.p2.sigClicked.connect(self.clicked)
        self.w1.addItem(self.p2)
        self.w1.setLabel('bottom', text='Time', units=u'μs')
        self.w1.setLabel('left', text='Fractional Current Blockage')
        self.w1.setLogMode(x=True,y=False)
        self.w1.showGrid(x=True, y=True)
        self.cb = pg.ColorButton(self.ui.scatterplot, color=(0,0,255,50))
        self.cb.setFixedHeight(30)
        self.cb.setFixedWidth(30)
        self.cb.move(0,250)
        self.cb.show()

        self.w2 = self.ui.frachistplot.addPlot()
        self.w2.setLabel('bottom', text='Fractional Current Blockage')
        self.w2.setLabel('left', text='Counts')

        self.w3 = self.ui.delihistplot.addPlot()
        self.w3.setLabel('bottom', text='ΔI', units ='A')
        self.w3.setLabel('left', text='Counts')

        self.w4 = self.ui.dwellhistplot.addPlot()
        self.w4.setLabel('bottom', text='Log Dwell Time', units = 'μs')
        self.w4.setLabel('left', text='Counts')

        self.w5 = self.ui.dthistplot.addPlot()
        self.w5.setLabel('bottom', text='dt', units = 's')
        self.w5.setLabel('left', text='Counts')

#        self.w6 = self.ui.PSDplot.addPlot()
#        self.w6.setLogMode(x = True, y = True)
#        self.w6.setLabel('bottom', text='Frequency (Hz)')
#        self.w6.setLabel('left', text='PSD (pA^2/Hz)')

        self.p3 = self.ui.eventplot.addPlot()
        self.p3.hideAxis('bottom')
        self.p3.hideAxis('left')


        self.logo=ndimage.imread(os.getcwd()+os.sep+"pythionlogo.png")
        self.logo=np.rot90(self.logo,-1)
        self.logo = pg.ImageItem(self.logo)
        self.p3.addItem(self.logo)
        self.p3.setAspectLocked(True)


        ####### Initializing various variables used for analysis##############
        self.direc=[]
        self.lr=[]
        self.lastevent=[]
        self.lastClicked=[]
        self.hasbaselinebeenset=0
        self.lastevent=0
        self.deli=[]
        self.frac=[]
        self.dwell=[]
        self.dt=[]
        self.catdata=[]
        self.colors=[]
        self.sdf = pd.DataFrame(columns = ['fn','color','deli','frac',
            'dwell','dt','startpoints','endpoints'])

    def Load(self, loadandplot = True):
        self.catdata=[]
        self.batchinfo = pd.DataFrame(columns = list(['cutstart', 'cutend']))
        self.p3.clear()
        self.p3.setLabel('bottom', text='Current', units='A', unitprefix = 'n')
        self.p3.setLabel('left', text='', units = 'Counts')
        self.p3.setAspectLocked(False)

        colors = np.array(self.sdf.color)
        for i in range(len(colors)):
            colors[i] = pg.Color(colors[i])

        self.p2.setBrush(colors, mask=None)

        self.ui.eventinfolabel.clear()
        self.ui.eventcounterlabel.clear()
        self.ui.meandelilabel.clear()
        self.ui.meandwelllabel.clear()
        self.ui.meandtlabel.clear()
        self.totalplotpoints=len(self.p2.data)
        self.ui.eventnumberentry.setText(str(0))



        self.threshold=np.float64(self.ui.thresholdentry.text())*10**-9
        self.ui.filelabel.setText(self.datafilename)
        print(self.datafilename)
        self.LPfiltercutoff = np.float64(self.ui.LPentry.text())*1000
        self.outputsamplerate = np.float64(self.ui.outputsamplerateentry.text())*1000 #use integer multiples of 4166.67 ie 2083.33 or 1041.67


        if str(os.path.splitext(self.datafilename)[1])=='.log':
            self.CHIMERAfile = np.dtype('<u2')
            self.data=np.fromfile(self.datafilename,self.CHIMERAfile)

            self.matfilename=str(os.path.splitext(self.datafilename)[0])
            self.mat = spio.loadmat(self.matfilename)


            samplerate = np.float64(self.mat['ADCSAMPLERATE'])
            TIAgain = np.int32(self.mat['SETUP_TIAgain'])
            preADCgain = np.float64(self.mat['SETUP_preADCgain'])
            currentoffset = np.float64(self.mat['SETUP_pAoffset'])
            ADCvref = np.float64(self.mat['SETUP_ADCVREF'])
            ADCbits = np.int32(self.mat['SETUP_ADCBITS'])
            closedloop_gain = TIAgain*preADCgain;


            if samplerate < 4000e3:
                self.data=self.data[::round(samplerate/self.outputsamplerate)]


            bitmask = (2**16 - 1) - (2**(16-ADCbits) - 1)
            self.data = -ADCvref + (2*ADCvref) * (self.data & bitmask) / 2**16
            self.data = (self.data/closedloop_gain + currentoffset)

            if os.name =='posix':
                self.data=self.data[0]
        
            ###############################################data has now been loaded
            ###############################################now filtering data

            Wn = round(self.LPfiltercutoff/(samplerate/2),4)
            b,a = signal.bessel(4, Wn, btype='low');

            self.data = signal.filtfilt(b,a,self.data)
            

        if str(os.path.splitext(self.datafilename)[1])=='.opt':
            self.data = np.fromfile(self.datafilename, dtype = np.dtype('>d'))
            self.matfilename = str(os.path.splitext(self.datafilename)[0])  
            
            try:
                self.mat = spio.loadmat(self.matfilename + '_inf')  
                samplerate = np.float64(self.mat['samplerate'])
                lowpass = np.float64(self.mat['filterfreq'])
                print samplerate
                print lowpass
            except TypeError:
                pass
            
            if self.outputsamplerate > 250e3:
                    print('sample rate can not be >250kHz for axopatch files, displaying with a rate of 250kHz')
                    self.outputsamplerate  = 250e3
#            self.data=self.data*10**9

            if self.LPfiltercutoff < 100e3:
                Wn = round(self.LPfiltercutoff/(100*10**3/2),4)
                b,a = signal.bessel(4, Wn, btype='low');
                self.data = signal.filtfilt(b,a,self.data)
            else:
                print('Filter value too high, data not filtered')

        if str(os.path.splitext(self.datafilename)[1])=='.txt':
            self.data=pandas.io.parsers.read_csv(self.datafilename,skiprows=1)
#            self.data=np.reshape(np.array(self.data),np.size(self.data))*10**9
            self.data=np.reshape(np.array(self.data),np.size(self.data))
            self.matfilename=str(os.path.splitext(self.datafilename)[0])


        if str(os.path.splitext(self.datafilename)[1])=='.npy':
            self.data = np.load(self.datafilename)
            self.matfilename=str(os.path.splitext(self.datafilename)[0])

        if str(os.path.splitext(self.datafilename)[1])=='.abf':
            f = open(self.datafilename, "rb")  # reopen the file
            f.seek(6144, os.SEEK_SET)
            self.data = np.fromfile(f, dtype = np.dtype('<i2'))
            self.matfilename=str(os.path.splitext(self.datafilename)[0])
            self.header = read_header(self.datafilename)
            self.samplerate = 1e6/self.header['protocol']['fADCSequenceInterval']
            self.telegraphmode = int(self.header['listADCInfo'][0]['nTelegraphEnable'])
            if self.telegraphmode == 1:
                self.abflowpass = self.header['listADCInfo'][0]['fTelegraphFilter']
                self.gain = self.header['listADCInfo'][0]['fTelegraphAdditGain']
            else:
                self.gain = 1
                self.abflowpass = self.samplerate
                
            self.data=self.data.astype(float)*(20./(65536*self.gain))*10**-9                
 
            if len(self.header['listADCInfo']) == 2:
                self.v = self.data[1::2]*self.gain/10
                self.data = self. data[::2]
            else:
                self.v = [] 
               
                
            if self.outputsamplerate > self.samplerate:
                    print('output samplerate can not be higher than samplerate, resetting to original rate')
                    self.outputsamplerate  = self.samplerate
                    self.ui.outputsamplerateentry.setText(str((round(self.samplerate)/1000)))
            if self.LPfiltercutoff >= self.abflowpass:
                    print('Already LP filtered lower than or at entry, data will not be filtered')
                    self.LPfiltercutoff  = self.abflowpass
                    self.ui.LPentry.setText(str((round(self.LPfiltercutoff)/1000)))
            else:
                Wn = round(self.LPfiltercutoff/(100*10**3/2),4)
                b,a = signal.bessel(4, Wn, btype='low');
                self.data = signal.filtfilt(b,a,self.data)

                
            tags = self.header['listTag']
            for tag in tags:
                if tag['sComment'][0:21] == "Holding on 'Cmd 0' =>":
                    cmdv = tag['sComment'][22:]
#                    cmdv = [int(s) for s in cmdv.split() if s.isdigit()]
                    cmdt = tag ['lTagTime']/self.outputsamplerate
                    self.p1.addItem(pg.InfiniteLine(cmdt))
#                    cmdtext = pg.TextItem(text = str(cmdv)+' mV')
                    cmdtext = pg.TextItem(text = str(cmdv))
                    self.p1.addItem(cmdtext)
                    cmdtext.setPos(cmdt,np.max(self.data))


        self.t=np.arange(0,len(self.data))
        self.t=self.t/self.outputsamplerate

        if self.hasbaselinebeenset==0:
            self.baseline=np.median(self.data)
            self.var=np.std(self.data)
        self.ui.eventcounterlabel.setText('Baseline='+str(round(self.baseline*10**9,2))+' nA')


        if loadandplot == True:
            self.p1.clear()
            self.p1.setDownsampling(ds = True)
            #skips plotting first and last two points, there was a weird spike issue
            self.p1.plot(self.t[2:][:-2],self.data[2:][:-2],pen='b')
    
            if str(os.path.splitext(self.datafilename)[1]) != '.abf':
                self.p1.addLine(y=self.baseline,pen='g')
                self.p1.addLine(y=self.threshold,pen='r')
    
            self.p1.autoRange()
    
            self.p3.clear()
            aphy, aphx = np.histogram(self.data, bins = len(self.data)/1000, range = [np.min(self.data),np.max(self.data)])
            aphx = aphx
#            aphhist = pg.BarGraphItem(height = aphy, x0 = aphx[:-1], x1 = aphx[1:], brush = 'b', pen = None)
            aphhist = pg.PlotCurveItem(aphx, aphy, stepMode=True, fillLevel=0, brush='b')
            self.p3.addItem(aphhist)
#            self.p3.setXRange(np.min(self.data), np.max(self.data))
            
    
    #        if self.v != []:
    #            self.p1.plot(self.t[2:][:-2],self.v[2:][:-2],pen='r')
            
    #        self.w6.clear()
    #        f, Pxx_den = signal.welch(self.data*10**12, self.outputsamplerate, nperseg = self.outputsamplerate)
    #        self.w6.plot(x = f[1:], y = Pxx_den[1:], pen = 'b')
    #        self.w6.setXRange(0,np.log10(self.outputsamplerate))

    def getfile(self):

        try:
            ######## attempt to open dialog from most recent directory########
            self.datafilename = str(QtGui.QFileDialog.getOpenFileName(self,'Open file',self.direc,("*.log;*.opt;*.npy;*.txt;*.abf")))
            self.direc=os.path.dirname(self.datafilename)
            self.Load()
        except TypeError:
            ####### if no recent directory exists open from working directory##
            self.direc==[]
            self.datafilename = str(QtGui.QFileDialog.getOpenFileName(self, 'Open file',os.getcwd(),("*.log;*.opt;*.npy;*.txt;*.abf"))  )
            self.direc=os.path.dirname(self.datafilename)
            self.Load()
        except IOError:
            #### if user cancels during file selection, exit loop#############
            return
            

    def analyze(self):
        global startpoints,endpoints, mins
        self.w2.clear()
        self.w3.clear()
        self.w4.clear()
        self.w5.clear()

        self.threshold = np.float64(self.ui.thresholdentry.text())*10**-9

#### find all points below threshold ####

        below = np.where(self.data < self.threshold)[0]

#### locate the points where the current crosses the threshold ####

        startandend = np.diff(below)
        startpoints = np.insert(startandend, 0, 2)
        endpoints = np.insert(startandend, -1, 2)
        startpoints = np.where(startpoints>1)[0]
        endpoints = np.where(endpoints>1)[0]
        startpoints = below[startpoints]
        endpoints = below[endpoints]

#### Eliminate events that start before file or end after file ####

        if startpoints[0] == 0:
            startpoints = np.delete(startpoints,0)
            endpoints = np.delete(endpoints,0)
        if endpoints [-1] == len(self.data):
            startpoints = np.delete(startpoints,-1)
            endpoints = np.delete(endpoints,-1)

#### Track points back up to baseline to find true start and end ####

        numberofevents=len(startpoints)
        highthresh = self.baseline - self.var

        for j in range(numberofevents):
            sp = startpoints[j] #mark initial guess for starting point
            while self.data[sp] < highthresh and sp > 0:
                sp = sp-1 # track back until we return to baseline
            startpoints[j] = sp # mark true startpoint

            ep = endpoints[j] #repeat process for end points
            while self.data[ep] < highthresh:
                ep = ep+1
#                if j == numberofevents - 1: # if this is the last event, check to make
                if ep == len(self.data) -1:  # sure that the current returns to baseline
                    endpoints[j] = 0              # before file ends. If not, mark points for
                    startpoints[j] = 0              # deletion and break from loop
                    ep = 0
                    break
                else:
                    try:
                        if ep > startpoints[j+1]: # if we hit the next startpoint before we
                            startpoints[j+1] = 0    # return to baseline, mark for deletion
                            endpoints[j] = 0                  # and break out of loop
                            ep = 0
                            break
                    except:
                        IndexError
            endpoints[j] = ep

        startpoints = startpoints[startpoints!=0] # delete those events marked for
        endpoints = endpoints[endpoints!=0]       # deletion earlier
        self.numberofevents = len(startpoints)

#### Now we want to move the endpoints to be the last minimum for each ####
#### event so we find all minimas for each event, and set endpoint to last ####

        self.deli = np.zeros(self.numberofevents)
        self.dwell = np.zeros(self.numberofevents)

        for i in range(self.numberofevents):
            mins = np.array(signal.argrelmin(self.data[startpoints[i]:endpoints[i]])[0] + startpoints[i])
            mins = mins[self.data[mins] < self.baseline - 4*self.var]
            if len(mins) == 1:
                pass
                self.deli[i] = self.baseline - min(self.data[startpoints[i]:endpoints[i]])
                self.dwell[i] = (endpoints[i]-startpoints[i])*1e6/self.outputsamplerate
                endpoints[i] = mins[0]
            elif len(mins) > 1:
                self.deli[i] = self.baseline - np.mean(self.data[mins[0]:mins[-1]])
                endpoints[i] = mins[-1]
                self.dwell[i] = (endpoints[i]-startpoints[i])*1e6/self.outputsamplerate


        startpoints = startpoints[self.deli!=0]
        endpoints = endpoints[self.deli!=0]
        self.deli = self.deli[self.deli!=0]
        self.dwell = self.dwell[self.dwell!=0]
        self.frac = self.deli/self.baseline
        self.dt = np.array(0)
        self.dt=np.append(self.dt,np.diff(startpoints)/self.outputsamplerate)
        self.numberofevents = len(self.dt)

        self.p1.clear()

        #skips plotting first and last two points, there was a weird spike issue
#        self.p1.plot(self.t[::10][2:][:-2],self.data[::10][2:][:-2],pen='b')
        self.p1.plot(self.t[2:][:-2],self.data[2:][:-2],pen='b')
        self.p1.plot(self.t[startpoints], self.data[startpoints],pen=None, symbol='o',symbolBrush='g',symbolSize=10)
        self.p1.plot(self.t[endpoints], self.data[endpoints], pen=None, symbol='o',symbolBrush='r',symbolSize=10)

        self.ui.eventcounterlabel.setText('Events:'+str(self.numberofevents))
        self.ui.meandelilabel.setText('Deli:'+str(round(np.mean(self.deli*10**9),2))+' nA')
        self.ui.meandwelllabel.setText('Dwell:'+str(round(np.median(self.dwell),2))+ u' μs')
        self.ui.meandtlabel.setText('Rate:'+str(round(self.numberofevents/self.t[-1],1))+' events/s')

        try:
            self.p2.data = self.p2.data[np.where(np.array(self.sdf.fn) != self.matfilename)]
        except:
            IndexError
        self.sdf = self.sdf[self.sdf.fn != self.matfilename]

        fn = pd.Series([self.matfilename,] * self.numberofevents)
        color = pd.Series([self.cb.color(),] * self.numberofevents)

        self.sdf = self.sdf.append(pd.DataFrame({'fn':fn,'color':color,'deli':self.deli,
                                    'frac':self.frac,'dwell':self.dwell,
                                    'dt':self.dt,'startpoints':startpoints,
                                    'endpoints':endpoints}), ignore_index=True)

        self.p2.addPoints(x=np.log10(self.dwell),y=self.frac,
        symbol='o', brush=(self.cb.color()), pen = None, size = 10)


        self.w1.addItem(self.p2)
        self.w1.setLogMode(x=True,y=False)
        self.p1.autoRange()
        self.w1.autoRange()
        self.ui.scatterplot.update()
        self.w1.setRange(yRange=[0,1])

        colors = set(self.sdf.color)
        for i, x in enumerate(colors):
            fracy, fracx = np.histogram(self.sdf.frac[self.sdf.color == x], bins=np.linspace(0, 1, int(self.ui.fracbins.text())))
            deliy, delix = np.histogram(self.sdf.deli[self.sdf.color == x], bins=np.linspace(float(self.ui.delirange0.text())*10**-9, float(self.ui.delirange1.text())*10**-9, int(self.ui.delibins.text())))
            dwelly, dwellx = np.histogram(np.log10(self.sdf.dwell[self.sdf.color == x]), bins=np.linspace(float(self.ui.dwellrange0.text()), float(self.ui.dwellrange1.text()), int(self.ui.dwellbins.text())))
            dty, dtx = np.histogram(self.sdf.dt[self.sdf.color == x], bins=np.linspace(float(self.ui.dtrange0.text()), float(self.ui.dtrange1.text()), int(self.ui.dtbins.text())))

#            hist = pg.PlotCurveItem(fracy, fracx , stepMode = True, fillLevel=0, brush = x, pen = 'k')
#            self.w2.addItem(hist)

            hist = pg.BarGraphItem(height = fracy, x0 = fracx[:-1], x1 = fracx[1:], brush = x)
            self.w2.addItem(hist)

#            hist = pg.PlotCurveItem(delix, deliy , stepMode = True, fillLevel=0, brush = x, pen = 'k')
#            self.w3.addItem(hist)

            hist = pg.BarGraphItem(height = deliy, x0 = delix[:-1], x1 = delix[1:], brush = x)
            self.w3.addItem(hist)
#            self.w3.autoRange()
            self.w3.setRange(xRange = [float(self.ui.delirange0.text())*10**-9, float(self.ui.delirange1.text())*10**-9])

#            hist = pg.PlotCurveItem(dwellx, dwelly , stepMode = True, fillLevel=0, brush = x, pen = 'k')
#            self.w4.addItem(hist)

            hist = pg.BarGraphItem(height = dwelly, x0 = dwellx[:-1], x1 = dwellx[1:], brush = x)
            self.w4.addItem(hist)

#            hist = pg.PlotCurveItem(dtx, dty , stepMode = True, fillLevel=0, brush = x, pen = 'k')
#            self.w5.addItem(hist)

            hist = pg.BarGraphItem(height = dty, x0 = dtx[:-1], x1 = dtx[1:], brush = x)
            self.w5.addItem(hist)

        self.save()

    def save(self):
         np.savetxt(self.matfilename+'DB.txt',np.column_stack((self.deli,self.frac,self.dwell,self.dt)),delimiter='\t')

    def inspectevent(self, clicked = []):

        #Reset plot
        self.p3.setLabel('bottom', text='Time', units='s')
        self.p3.setLabel('left', text='Current', units='A')
        self.p3.clear()

        #Correct for user error if non-extistent number is entered
        eventbuffer=np.int(self.ui.eventbufferentry.text())
        firstindex = self.sdf.fn[self.sdf.fn == self.matfilename].index[0]
        if clicked == []:
            eventnumber = np.int(self.ui.eventnumberentry.text())
        else:
            eventnumber = clicked - firstindex
            self.ui.eventnumberentry.setText(str(eventnumber))
        if eventnumber>=self.numberofevents:
            eventnumber=self.numberofevents-1
            self.ui.eventnumberentry.setText(str(eventnumber))

        #plot event trace
        self.p3.plot(self.t[startpoints[eventnumber]-eventbuffer:endpoints[eventnumber]+eventbuffer],
                     self.data[startpoints[eventnumber]-eventbuffer:endpoints[eventnumber]+eventbuffer], pen='b')

        #plot event fit
        self.p3.plot(self.t[startpoints[eventnumber]-eventbuffer:endpoints[eventnumber]+eventbuffer],np.concatenate((
                     np.repeat(np.array([self.baseline]),eventbuffer),np.repeat(np.array([self.baseline-self.deli[eventnumber
                     ]]),endpoints[eventnumber]-startpoints[eventnumber]),np.repeat(np.array([self.baseline]),eventbuffer)),0),pen=pg.mkPen(color=(173,27,183),width=3))

        self.p3.autoRange()
        #Mark event that is being viewed on scatter plot

        colors = np.array(self.sdf.color)
        for i in range(len(colors)):
            colors[i] = pg.Color(colors[i])
        colors[firstindex + eventnumber] = pg.mkColor('r')

        self.p2.setBrush(colors, mask=None)


        #Mark event start and end points
        self.p3.plot([self.t[startpoints[eventnumber]], self.t[startpoints[eventnumber]]],[self.data[startpoints[eventnumber]], self.data[startpoints[eventnumber]]],pen=None, symbol='o',symbolBrush='g',symbolSize=12)
        self.p3.plot([self.t[endpoints[eventnumber]], self.t[endpoints[eventnumber]]],[self.data[endpoints[eventnumber]], self.data[endpoints[eventnumber]]],pen=None, symbol='o',symbolBrush='r',symbolSize=12)

        self.ui.eventinfolabel.setText('Dwell Time=' + str(round(self.dwell[eventnumber],2))+ u' μs,   Deli='+str(round(self.deli[eventnumber]*10**9,2)) +' nA')


#        if self.ui.cusumstepentry.text() != 'None':
#
# ########################################################################
#
#            x=self.data[startpoints[eventnumber]-eventbuffer:endpoints[eventnumber]+eventbuffer]
#            mins=signal.argrelmin(x)[0]
#            drift=.0
#            self.fitthreshold = np.float64(self.ui.cusumstepentry.text())
#            eventfit=np.array((0))
#
#            gp, gn = np.zeros(x.size), np.zeros(x.size)
#            ta, tai, taf = np.array([[], [], []], dtype=int)
#            tap, tan = 0, 0
#            # Find changes (online form)
#            for i in range(mins[0], mins[-1]):
#                s = x[i] - x[i-1]
#                gp[i] = gp[i-1] + s - drift  # cumulative sum for + change
#                gn[i] = gn[i-1] - s - drift  # cumulative sum for - change
#                if gp[i] < 0:
#                    gp[i], tap = 0, i
#                if gn[i] < 0:
#                    gn[i], tan = 0, i
#                if gp[i] > self.fitthreshold or gn[i] > self.fitthreshold:  # change detected!
#                    ta = np.append(ta, i)    # alarm index
#                    tai = np.append(tai, tap if gp[i] > self.fitthreshold else tan)  # start
#                    gp[i], gn[i] = 0, 0      # reset alarm
#
#            eventfit=np.repeat(np.array(self.baseline),ta[0])
#            for i in range(1,ta.size):
#                eventfit=np.concatenate((eventfit,np.repeat(np.array(np.mean(x[ta[i-1]:ta[i]])),ta[i]-ta[i-1])))
#            eventfit=np.concatenate((eventfit,np.repeat(np.array(self.baseline),x.size-ta[-1])))
#            self.p3.plot(self.t[startpoints[eventnumber]-eventbuffer:endpoints[eventnumber]+eventbuffer],eventfit
#                ,pen=pg.mkPen(color=(255,255,0),width=3))
#    #        pg.plot(eventfit)
#
#
#            self.p3.plot(self.t[ta+startpoints[eventnumber]-eventbuffer],x[ta],pen=None,symbol='o',symbolBrush='m',symbolSize=8)
#
#
# ########################################################################


    def nextevent(self):
        eventnumber=np.int(self.ui.eventnumberentry.text())

        if eventnumber>=self.numberofevents-1:
            eventnumber=0
        else:
            eventnumber=np.int(self.ui.eventnumberentry.text())+1
        self.ui.eventnumberentry.setText(str(eventnumber))
        self.inspectevent()

    def previousevent(self):

        eventnumber=np.int(self.ui.eventnumberentry.text())

        eventnumber=np.int(self.ui.eventnumberentry.text())-1
        if eventnumber<0:
            eventnumber=self.numberofevents-1
        self.ui.eventnumberentry.setText(str(eventnumber))
        self.inspectevent()

    def cut(self):
        
        ###### first check to see if cutting############

        if self.lr==[]:
            ######## if no cutting window exists, make one##########
            self.lr = pg.LinearRegionItem()
            self.lr.hide()

            ##### detect clears and auto-position window around the clear#####
            clears = np.where(np.abs(self.data) > self.baseline + 10*self.var)[0]
            if clears != []:
                clearstarts = clears[0]
                try:
                    clearends = clearstarts + np.where((self.data[clearstarts:-1] > self.baseline) &
                    (self.data[clearstarts:-1] < self.baseline+self.var))[0][10000]
                except:
                    clearends = -1
                clearstarts = np.where(self.data[0:clearstarts] > self.baseline)
                try:
                    clearstarts = clearstarts[0][-1]
                except:
                    clearstarts = 0

                self.lr.setRegion((self.t[clearstarts],self.t[clearends]))

            self.p1.addItem(self.lr)
            self.lr.show()


        #### if cut region has been set, cut region and replot remaining data####
        else:
            cutregion = self.lr.getRegion()
            self.p1.clear()
            self.data = np.delete(self.data,np.arange(np.int(cutregion[0]*self.outputsamplerate),np.int(cutregion[1]*self.outputsamplerate)))

            self.t=np.arange(0,len(self.data))
            self.t=self.t/self.outputsamplerate

            if self.hasbaselinebeenset==0:
                self.baseline = np.median(self.data)
                self.var=np.std(self.data)
                self.ui.eventcounterlabel.setText('Baseline='+str(round(self.baseline*10**9,2))+' nA')

            self.p1.plot(self.t,self.data,pen='b')
            if str(os.path.splitext(self.datafilename)[1]) != '.abf':
                self.p1.addLine(y=self.baseline,pen='g')
                self.p1.addLine(y=self.threshold,pen='r')
            self.lr=[]
#            self.p1.autoRange()
            self.p3.clear()
            aphy, aphx = np.histogram(self.data, bins = len(self.data)/1000,range = [np.min(self.data),np.max(self.data)])
            aphhist = pg.BarGraphItem(height = aphy, x0 = aphx[:-1], x1 = aphx[1:],brush = 'b', pen = None)
            self.p3.addItem(aphhist)
            self.p3.setXRange(np.min(self.data), np.max(self.data))
            
            cf = pd.DataFrame([cutregion], columns = list(['cutstart', 'cutend']))
            self.batchinfo = self.batchinfo.append(cf, ignore_index = True)


    def baselinecalc(self):
        if self.lr==[]:
            self.p1.clear()
            self.lr = pg.LinearRegionItem()
            self.lr.hide()
            self.p1.addItem(self.lr)

#            self.p1.plot(self.t[::100],self.data[::100],pen='b')
            self.p1.plot(self.t,self.data,pen='b')
            self.lr.show()

        else:
            calcregion=self.lr.getRegion()
            self.p1.clear()

            self.baseline=np.median(self.data[np.arange(np.int(calcregion[0]*self.outputsamplerate),np.int(calcregion[1]*self.outputsamplerate))])
            self.var=np.std(self.data[np.arange(np.int(calcregion[0]*self.outputsamplerate),np.int(calcregion[1]*self.outputsamplerate))])
#            self.p1.plot(self.t[::10][2:][:-2],self.data[::10][2:][:-2],pen='b')
            self.p1.plot(self.t,self.data,pen='b')
            self.p1.addLine(y=self.baseline,pen='g')
            self.p1.addLine(y=self.threshold,pen='r')
            self.lr=[]
            self.hasbaselinebeenset=1
            self.ui.eventcounterlabel.setText('Baseline='+str(round(self.baseline*10**9,2))+' nA')
            self.p1.autoRange()


    def clearscatter(self):
        self.p2.setData(x=[],y=[])
        self.lastevent=[]
        self.ui.scatterplot.update()
        self.w2.clear()
        self.w3.clear()
        self.w4.clear()
        self.w5.clear()
        self.sdf = pd.DataFrame(columns = ['fn','color','deli','frac',
            'dwell','dt','startpoints','endpoints'])

    def deleteevent(self):
        global startpoints,endpoints
        eventnumber = np.int(self.ui.eventnumberentry.text())
        firstindex = self.sdf.fn[self.sdf.fn == self.matfilename].index[0]
        if eventnumber > self.numberofevents:
            eventnumber = self.numberofevents-1
            self.ui.eventnumberentry.setText(str(eventnumber))
        self.deli=np.delete(self.deli,eventnumber)
        self.dwell=np.delete(self.dwell,eventnumber)
        self.dt=np.delete(self.dt,eventnumber)
        self.frac=np.delete(self.frac,eventnumber)
        startpoints=np.delete(startpoints,eventnumber)
        endpoints=np.delete(endpoints,eventnumber)
        self.p2.data=np.delete(self.p2.data,firstindex + eventnumber)

        self.numberofevents = len(self.dt)
        self.ui.eventcounterlabel.setText('Events:'+str(self.numberofevents))

        self.sdf = self.sdf.drop(firstindex + eventnumber).reset_index(drop = True)
        self.inspectevent()

        self.w2.clear()
        self.w3.clear()
        self.w4.clear()
        self.w5.clear()
        colors = set(self.sdf.color)
        for i, x in enumerate(colors):
            fracy, fracx = np.histogram(self.sdf.frac[self.sdf.color == x], bins=np.linspace(0, 1, int(self.ui.fracbins.text())))
            deliy, delix = np.histogram(self.sdf.deli[self.sdf.color == x], bins=np.linspace(float(self.ui.delirange0.text())*10**-9, float(self.ui.delirange1.text())*10**-9, int(self.ui.delibins.text())))
            dwelly, dwellx = np.histogram(np.log10(self.sdf.dwell[self.sdf.color == x]), bins=np.linspace(float(self.ui.dwellrange0.text()), float(self.ui.dwellrange1.text()), int(self.ui.dwellbins.text())))
            dty, dtx = np.histogram(self.sdf.dt[self.sdf.color == x], bins=np.linspace(float(self.ui.dtrange0.text()), float(self.ui.dtrange1.text()), int(self.ui.dtbins.text())))

#            hist = pg.PlotCurveItem(fracy, fracx , stepMode = True, fillLevel=0, brush = x, pen = 'k')
#            self.w2.addItem(hist)

            hist = pg.BarGraphItem(height = fracy, x0 = fracx[:-1], x1 = fracx[1:], brush = x)
            self.w2.addItem(hist)

#            hist = pg.PlotCurveItem(delix, deliy , stepMode = True, fillLevel=0, brush = x, pen = 'k')
#            self.w3.addItem(hist)

            hist = pg.BarGraphItem(height = deliy, x0 = delix[:-1], x1 = delix[1:], brush = x)
            self.w3.addItem(hist)
#            self.w3.autoRange()
            self.w3.setRange(xRange = [float(self.ui.delirange0.text())*10**-9, float(self.ui.delirange1.text())*10**-9])

#            hist = pg.PlotCurveItem(dwellx, dwelly , stepMode = True, fillLevel=0, brush = x, pen = 'k')
#            self.w4.addItem(hist)

            hist = pg.BarGraphItem(height = dwelly, x0 = dwellx[:-1], x1 = dwellx[1:], brush = x)
            self.w4.addItem(hist)

#            hist = pg.PlotCurveItem(dtx, dty , stepMode = True, fillLevel=0, brush = x, pen = 'k')
#            self.w5.addItem(hist)

            hist = pg.BarGraphItem(height = dty, x0 = dtx[:-1], x1 = dtx[1:], brush = x)
            self.w5.addItem(hist)

        self.save()

    def invertdata(self):
        self.p1.clear()
        self.data=-self.data

        if self.hasbaselinebeenset==0:
            self.baseline=np.median(self.data)
            self.var=np.std(self.data)

#        self.p1.plot(self.t[::10],self.data[::10],pen='b')
        self.p1.plot(self.t,self.data,pen='b')
        self.p1.addLine(y=self.baseline,pen='g')
        self.p1.addLine(y=self.threshold,pen='r')
        self.p1.autoRange()

    def clicked(self, plot, points):
        for i, p in enumerate(self.p2.points()):
            if p.pos() == points[0].pos():
                clickedindex = i

        if self.sdf.fn[clickedindex] != self.matfilename:
            print('Event is from an earlier file, not clickable')

        else:
            self.inspectevent(clickedindex)



    def concatenatetext(self):
        if self.direc==[]:
            textfilenames = QtGui.QFileDialog.getOpenFileNames(self, 'Open file','*.txt')
            self.direc=os.path.dirname(str(textfilenames[0]))
        else:
            textfilenames =QtGui.QFileDialog.getOpenFileNames(self, 'Open file',self.direc,'*.txt')
            self.direc=os.path.dirname(str(textfilenames[0]))
        i=0
        while i<len(textfilenames):
            temptextdata=np.fromfile(str(textfilenames[i]),sep='\t')
            temptextdata=np.reshape(temptextdata,(len(temptextdata)/4,4))
            if i==0:
                newtextdata=temptextdata
            else:
                newtextdata=np.concatenate((newtextdata,temptextdata))
            i=i+1

        newfilename = QtGui.QFileDialog.getSaveFileName(self, 'New File name',self.direc,'*.txt')
        np.savetxt(str(newfilename),newtextdata,delimiter='\t')

    def nextfile(self):
        if str(os.path.splitext(self.datafilename)[1])=='.log':
            startindex=self.matfilename[-6::]
            filebase=self.matfilename[0:len(self.matfilename)-6]
            nextindex=str(int(startindex)+1)
            while os.path.isfile(filebase+nextindex+'.log')==False:
                nextindex=str(int(nextindex)+1)
                if int(nextindex)>int(startindex)+1000:
                    print('no such file')
                    break
            if os.path.isfile(filebase+nextindex+'.log')==True:
                self.datafilename=(filebase+nextindex+'.log')
                self.Load()

        if str(os.path.splitext(self.datafilename)[1])=='.abf':
            startindex=self.matfilename[-4::]
            filebase=self.matfilename[0:len(self.matfilename)-4]
            nextindex=str(int(startindex)+1).zfill(4)
            while os.path.isfile(filebase+nextindex+'.abf')==False:
                nextindex=str(int(nextindex)+1).zfill(4)
                if int(nextindex)>int(startindex)+1000:
                    print('no such file')
                    break
            if os.path.isfile(filebase+nextindex+'.abf')==True:
                self.datafilename=(filebase+nextindex+'.abf')
                self.Load()



    def previousfile(self):
        if str(os.path.splitext(self.datafilename)[1])=='.log':
            startindex=self.matfilename[-6::]
            filebase=self.matfilename[0:len(self.matfilename)-6]
            nextindex=str(int(startindex)-1)
            while os.path.isfile(filebase+nextindex+'.log')==False:
                nextindex=str(int(nextindex)-1)
                if int(nextindex)<int(startindex)-1000:
                    print('no such file')
                    break
            if os.path.isfile(filebase+nextindex+'.log')==True:
                self.datafilename=(filebase+nextindex+'.log')
                self.Load()

        if str(os.path.splitext(self.datafilename)[1])=='.abf':
            startindex=self.matfilename[-4::]
            filebase=self.matfilename[0:len(self.matfilename)-4]
            nextindex=str(int(startindex)-1).zfill(4)
            while os.path.isfile(filebase+nextindex+'.abf')==False:
                nextindex=str(int(nextindex)-1).zfill(4)
                if int(nextindex)<int(startindex)-1000:
                    print('no such file')
                    break
            if os.path.isfile(filebase+nextindex+'.abf')==True:
                self.datafilename=(filebase+nextindex+'.abf')
                self.Load()

    def savetrace(self):
        self.data.astype('d').tofile(self.matfilename+'_trace.bin')

    def showcattrace(self):
        eventbuffer=np.int(self.ui.eventbufferentry.text())
        numberofevents=len(self.dt)

        self.p1.clear()
        eventtime = [0]
        for i in range(numberofevents):
            if i<numberofevents-1:
                if endpoints[i]+eventbuffer>startpoints[i+1]:
                    print('overlapping event')
                else:
                    eventdata = self.data[startpoints[i]-eventbuffer:endpoints[i]+eventbuffer]
                    fitdata = np.concatenate((np.repeat(np.array([self.baseline]),eventbuffer),np.repeat(np.array([
                        self.baseline-self.deli[i]]),endpoints[i]-startpoints[i]),np.repeat(np.array([self.baseline]),eventbuffer)),0)
                    eventtime = np.arange(0,len(eventdata)) + .75*eventbuffer + eventtime[-1]
                    self.p1.plot(eventtime/self.outputsamplerate, eventdata,pen='b')
                    self.p1.plot(eventtime/self.outputsamplerate, fitdata,pen=pg.mkPen(color=(173,27,183),width=2))

        self.p1.autoRange()

    def savecattrace(self):
        eventbuffer=np.int(self.ui.eventbufferentry.text())
        numberofevents=len(self.dt)
        self.catdata=self.data[startpoints[0]-eventbuffer:endpoints[0]+eventbuffer]
        self.catfits=np.concatenate((np.repeat(np.array([self.baseline]),eventbuffer),np.repeat(np.array([
            self.baseline-self.deli[0]]),endpoints[0]-startpoints[0]),
            np.repeat(np.array([self.baseline]),eventbuffer)),0)

        for i in range(numberofevents):
            if i<numberofevents-1:
                if endpoints[i]+eventbuffer>startpoints[i+1]:
                    print('overlapping event')
                else:
                    self.catdata=np.concatenate((self.catdata,self.data[startpoints[i]-eventbuffer:endpoints[i]+eventbuffer]),0)
                    self.catfits=np.concatenate((self.catfits,np.concatenate((np.repeat(np.array([self.baseline]),eventbuffer),np.repeat(np.array([
                        self.baseline-self.deli[i]]),endpoints[i]-startpoints[i]),np.repeat(np.array([self.baseline]),eventbuffer)),0)),0)

        self.tcat=np.arange(0,len(self.catdata))
        self.tcat=self.tcat/self.outputsamplerate
        self.catdata=self.catdata[::10]
        self.catdata.astype('d').tofile(self.matfilename+'_cattrace.bin')


    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key_Up:
            self.nextfile()
        if key == QtCore.Qt.Key_Down:
            self.previousfile()
        if key == QtCore.Qt.Key_Right:
            self.nextevent()
        if key == QtCore.Qt.Key_Left:
            self.previousevent()
        if key == QtCore.Qt.Key_Return:
            self.Load()
        if key == QtCore.Qt.Key_Space:
            self.analyze()
        if key == QtCore.Qt.Key_Delete:
            self.deleteevent()

    def saveeventfits(self):
        eventbuffer=np.int(self.ui.eventbufferentry.text())
        numberofevents=len(self.dt)
        self.catdata=self.data[startpoints[0]-eventbuffer:endpoints[0]+eventbuffer]
        self.catfits=np.concatenate((np.repeat(np.array([self.baseline]),eventbuffer),np.repeat(np.array([
            self.baseline-self.deli[0]]),endpoints[0]-startpoints[0]),
            np.repeat(np.array([self.baseline]),eventbuffer)),0)

        for i in range(numberofevents):
            if i<numberofevents-1:
                if endpoints[i]+eventbuffer>startpoints[i+1]:
                    print('overlapping event')
                else:
                    self.catdata=np.concatenate((self.catdata,self.data[startpoints[i]-eventbuffer:endpoints[i]+eventbuffer]),0)
                    self.catfits=np.concatenate((self.catfits,np.concatenate((np.repeat(np.array([self.baseline]),eventbuffer),np.repeat(np.array([
                        self.baseline-self.deli[i]]),endpoints[i]-startpoints[i]),np.repeat(np.array([self.baseline]),eventbuffer)),0)),0)

        self.tcat=np.arange(0,len(self.catdata))
        self.tcat=self.tcat/self.outputsamplerate
        self.catfits.astype('d').tofile(self.matfilename+'_cattrace.bin')

    def CUSUM(self):
        self.p1.clear()
        self.p1.setDownsampling(ds = False)
        cusum = detect_cusum(self.data, basesd = self.var, dt = 1/self.outputsamplerate, 
                             threshhold  = np.float64(self.ui.thresholdentry.text()),
                             stepsize = np.float64(self.ui.levelthresholdentry.text()), 
                             minlength = 10)
        np.savetxt(self.matfilename+'_Levels.txt', np.abs(cusum['jumps']*10**12),delimiter='\t')

        self.p1.plot(self.t[2:][:-2],self.data[2:][:-2],pen='b')

        self.w3.clear()
        amp = np.abs(cusum['jumps']*10**12)
        ampy, ampx = np.histogram(amp, bins=np.linspace(float(self.ui.delirange0.text()), float(self.ui.delirange1.text()), int(self.ui.delibins.text())))
        hist = pg.BarGraphItem(height = ampy, x0 = ampx[:-1], x1 = ampx[1:], brush = 'b')
        self.w3.addItem(hist)
#        self.w3.autoRange()
        self.w3.setRange(xRange = [np.min(ampx),np.max(ampx)])

        cusumlines = np.array([]).reshape(0,2)
        for i,level in enumerate(cusum['CurrentLevels']):
            y = 2*[level]
            x = cusum['EventDelay'][i:i+2]
            self.p1.plot(y = y, x = x, pen = 'r')
            cusumlines = np.concatenate((cusumlines,np.array(zip(x,y))))
            try:
                y = cusum['CurrentLevels'][i:i+2]
                x = 2*[cusum['EventDelay'][i+1]]
                self.p1.plot(y = y, x = x, pen = 'r')
                cusumlines = np.concatenate((cusumlines,np.array(zip(x,y))))
            except Exception:
                pass
            
        cusumlines.astype('d').tofile(self.matfilename+'_cusum.bin')
        self.savetrace()

#        amp = np.abs(cusum['jumps']*10**12)*10**9
#        ampy, ampx = np.histogram(amp,bins=np.linspace(0, np.max(amp), 100))
#        hist = pg.BarGraphItem(height = ampy, x0 = ampx[:-1], x1 = ampx[1:])
#        levelplot = pg.plot()
#        levelplot.addItem(hist)

    def savetarget(self):
        self.batchinfo = self.batchinfo.append(pd.DataFrame({'deli':self.deli,
                    'frac':self.frac,'dwell':self.dwell,'dt':self.dt, 
                    'startpoints':startpoints,'endpoints':endpoints}), ignore_index=True)
        self.batchinfo.to_pickle(self.matfilename+'batchinfo.pkl')

    def batchinfodialog(self):
        self.bp = batchprocesser()
        self.bp.show()
        
        QtCore.QObject.connect(self.bp.uibp.okbutton, QtCore.SIGNAL('clicked()'), self.batchprocess)
        
    def batchprocess(self):
        global endpoints, startpoints
        
        self.p1.setDownsampling(ds = False)
        self.mindwell = np.float64(self.bp.uibp.mindwellbox.text())
        self.minfrac = np.float64(self.bp.uibp.minfracbox.text())
        self.minlevelt = np.float64(self.bp.uibp.minleveltbox.text())*10**-6
        self.samplerate = self.bp.uibp.sampratebox.text()
        self.LPfiltercutoff = self.bp.uibp.LPfilterbox.text()
        self.ui.outputsamplerateentry.setText(self.samplerate)
        self.ui.LPentry.setText(self.LPfiltercutoff)
        cusumstep = np.float64(self.bp.uibp.cusumstepentry.text())
        cusumthresh = np.float64(self.bp.uibp.cusumthreshentry.text())
        self.bp.destroy()   
        self.p1.clear()
        
        try:
            ######## attempt to open dialog from most recent directory########
            self.filelist = QtGui.QFileDialog.getOpenFileNames(self,'Select Files',self.direc,("*.pkl"))
            self.direc=os.path.dirname(self.filelist[0])
        except TypeError:
            ####### if no recent directory exists open from working directory##
            self.direc==[]
            self.filelist = QtGui.QFileDialog.getOpenFileNames(self, 'Select Files',os.getcwd(),("*.pkl"))
            self.direc=os.path.dirname(self.filelist[0])
        except IOError:
            #### if user cancels during file selection, exit loop#############
            return

        eventbuffer=np.int(self.ui.eventbufferentry.text())
        eventtime = [0]
        ll = np.array([])


        for f in self.filelist: 
            batchinfo = pd.read_pickle(f)
            try:
                self.datafilename = f[:-13] + '.opt'
                self.Load(loadandplot = False)
            except IOError:
                self.datafilename = f[:-13] + '.log'
                self.Load(loadandplot = False)
                
            
            try:
                cs = batchinfo.cutstart[np.isfinite(batchinfo.cutstart)]
                ce = batchinfo.cutend[np.isfinite(batchinfo.cutend)]
                for i, cut in enumerate(cs):
                    self.data = np.delete(self.data,np.arange(np.int(cut*self.outputsamplerate),np.int(ce[i]*self.outputsamplerate)))
            except TypeError:
                pass
             
             
            self.deli = np.array(batchinfo.deli[np.isfinite(batchinfo.deli)])
            self.frac = np.array(batchinfo.frac[np.isfinite(batchinfo.frac)])
            self.dwell = np.array(batchinfo.dwell[np.isfinite(batchinfo.dwell)])
            self.dt = np.array(batchinfo.dt[np.isfinite(batchinfo.dt)])
            startpoints = np.array(batchinfo.startpoints[np.isfinite(batchinfo.startpoints)])
            endpoints = np.array(batchinfo.endpoints[np.isfinite(batchinfo.endpoints)])
            
            for i,dwell in enumerate(self.dwell):
                print(str(i) + '/' + str(len(self.dwell)))
                toffset = (eventtime[-1] + .75*eventbuffer)/self.outputsamplerate
                if i < len(self.dt)-1 and dwell > self.mindwell and self.frac[i] >self.minfrac:
                    if endpoints[i]+eventbuffer>startpoints[i+1]:
                        print('overlapping event')
                    else:
                        eventdata = self.data[startpoints[i]-eventbuffer:endpoints[i]+eventbuffer]
                        eventtime = np.arange(0,len(eventdata)) + .75*eventbuffer + eventtime[-1]
                        self.p1.plot(eventtime/self.outputsamplerate, eventdata,pen='b')
                        cusum = detect_cusum(eventdata, basesd = np.std(eventdata[0:eventbuffer])
                            , dt = 1/self.outputsamplerate, threshhold  = cusumthresh
                            , stepsize = cusumstep, minlength = self.minlevelt*self.outputsamplerate, maxstates = 10)
                        
                        while len(cusum['CurrentLevels']) < 3:
                            cusumthresh = cusumthresh *.9
                            cusumstep = cusumstep * .9
                            cusum = detect_cusum(eventdata, basesd = np.std(eventdata[0:eventbuffer])
                                , dt = 1/self.outputsamplerate, threshhold  = cusumthresh
                                , stepsize = cusumstep, minlength = self.minlevelt*self.outputsamplerate, maxstates = 10)
                      
#                        print len(cusum['CurrentLevels'])

                        
#                        if np.max(cusum['CurrentLevels'])-np.min(cusum['CurrentLevels']) == 0:
#                            cusum = detect_cusum(eventdata, basesd = np.std(eventdata)
#                                , dt = 1/self.outputsamplerate, threshhold  = cusumthresh/10
#                                , stepsize = cusumstep/10, minlength = self.minlevelt*self.outputsamplerate)
                            
                        ll = np.concatenate((ll,[(np.max(cusum['CurrentLevels'])-np.min(cusum['CurrentLevels']))/np.max(cusum['CurrentLevels'])]))
                        cusumthresh = cusum['Threshold']
                        cusumstep = cusum['stepsize']
                                                    
                        
                        for j,level in enumerate(cusum['CurrentLevels']):
                            self.p1.plot(y = 2*[level], x = toffset + cusum['EventDelay'][j:j+2], pen = pg.mkPen( 'r', width = 5))
                            try:
                                self.p1.plot(y = cusum['CurrentLevels'][j:j+2], x = toffset + 2*[cusum['EventDelay'][j+1]], pen = pg.mkPen( 'r', width = 5))
                            except Exception:
                                pass

        np.savetxt(self.matfilename+'llDB.txt',ll,delimiter='\t')
        self.p1.autoRange()
        
        print('\007')
        

    def sizethepore(self):
        self.ps = PoreSizer()
        self.ps.show()

def start():
    app = QtGui.QApplication(sys.argv)
    myapp = GUIForm()
    myapp.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    global myapp
    app = QtGui.QApplication(sys.argv)
    myapp = GUIForm()
    myapp.show()
    sys.exit(app.exec_())

