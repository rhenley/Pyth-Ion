# -*- coding: utf8 -*-
import sys
import numpy as np
import scipy as sp
import os
from scipy import signal
from scipy import io
from PlotGUI import *
import pyqtgraph as pg
import pandas.io.parsers
#from lmfit import Minimizer, Parameters
 
class GUIForm(QtGui.QMainWindow):
    
 
    def __init__(self, master=None):
        QtGui.QMainWindow.__init__(self,master)
        self.ui = Ui_PythIon()
        self.ui.setupUi(self)
        
        QtCore.QObject.connect(self.ui.loadbutton, QtCore.SIGNAL('clicked()'), self.getfile)
        QtCore.QObject.connect(self.ui.analyzebutton, QtCore.SIGNAL('clicked()'), self.analyze)
#        QtCore.QObject.connect(self.ui.savebutton, QtCore.SIGNAL('clicked()'), self.save)
        QtCore.QObject.connect(self.ui.cutbutton, QtCore.SIGNAL('clicked()'), self.cut)
        QtCore.QObject.connect(self.ui.baselinebutton, QtCore.SIGNAL('clicked()'), self.baselinecalc)
        QtCore.QObject.connect(self.ui.clearscatterbutton, QtCore.SIGNAL('clicked()'), self.clearscatter)
        QtCore.QObject.connect(self.ui.deleteeventbutton, QtCore.SIGNAL('clicked()'), self.deleteevent)
        QtCore.QObject.connect(self.ui.invertbutton, QtCore.SIGNAL('clicked()'), self.invertdata)
        QtCore.QObject.connect(self.ui.concatenatebutton, QtCore.SIGNAL('clicked()'), self.concatenatetext)
        QtCore.QObject.connect(self.ui.nextfilebutton, QtCore.SIGNAL('clicked()'), self.nextfile)
        QtCore.QObject.connect(self.ui.previousfilebutton, QtCore.SIGNAL('clicked()'), self.previousfile)
        QtCore.QObject.connect(self.ui.savetracebutton, QtCore.SIGNAL('clicked()'), self.savetrace)
        QtCore.QObject.connect(self.ui.showcatbutton, QtCore.SIGNAL('clicked()'), self.showcattrace)
        QtCore.QObject.connect(self.ui.savecatbutton, QtCore.SIGNAL('clicked()'), self.savecattrace)
        
        QtCore.QObject.connect(self.ui.gobutton, QtCore.SIGNAL('clicked()'), self.inspectevent)
        QtCore.QObject.connect(self.ui.previousbutton, QtCore.SIGNAL('clicked()'), self.previousevent)
        QtCore.QObject.connect(self.ui.nextbutton, QtCore.SIGNAL('clicked()'), self.nextevent)
                

        self.ui.signalplot.setBackground('w')
        self.ui.scatterplot.setBackground('w')
        self.ui.eventplot.setBackground('w')
        
        self.p1 = self.ui.signalplot.addPlot()
        self.p1.setLabel('bottom', text='Time', units='s')
        self.p1.setLabel('left', text='Current', units='nA')
        self.p1.enableAutoRange(axis = 'x')
        
#        self.p2 = self.ui.scatterplot.addPlot()

        self.w1 = self.ui.scatterplot.addPlot()
        self.p2=pg.ScatterPlotItem()
        self.p2.sigClicked.connect(self.clicked)
        self.w1.addItem(self.p2)
        self.w1.setLabel('bottom', text='Time', units=u'μs')
        self.w1.setLabel('left', text='Fractional Current Blockage')
        self.w1.setLogMode(x=True,y=False)
        self.cb = pg.ColorButton(self.ui.scatterplot, color=(0,0,255,50))
        self.cb.move(450,250)
        self.cb.show()
        
        self.p3=self.ui.eventplot.addPlot()
#        self.p3.setLabel('bottom', text='Time', units=u'μs')
        self.p3.setLabel('bottom', text='Time', units='s')
        self.p3.setLabel('left', text='Current', units='nA')
        self.p3.hideAxis('bottom')
        self.p3.hideAxis('left')
        
        
        self.logo=sp.ndimage.imread(os.getcwd()+os.sep+"pythionlogo.png")
        self.logo=np.rot90(self.logo,-1)
        self.logo=pg.ImageItem(self.logo)
        self.p3.addItem(self.logo)
        
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

 
    def Load(self): 
        self.catdata=[]
        self.p1.clear()
        self.p3.clear()
        self.p3.hideAxis('bottom')
        self.p3.hideAxis('left')
        self.p3.addItem(self.logo)
        self.ui.eventinfolabel.clear()
        self.ui.eventcounterlabel.clear()
        self.ui.meandelilabel.clear()
        self.ui.meandwelllabel.clear()
        self.ui.meandtlabel.clear()
        self.totalplotpoints=len(self.p2.data)
        self.ui.eventnumberentry.setText(str(0))

#        colors=[]
#        colors[0:self.totalplotpoints]=[.5]*self.totalplotpoints
#        self.p2.setBrush(colors, mask=None) 

        self.threshold=np.float64(self.ui.thresholdentry.text()) 
        self.ui.filelabel.setText(self.datafilename)
        print self.datafilename  
        self.LPfiltercutoff = np.float64(self.ui.LPentry.text())*1000
        self.outputsamplerate = np.float64(self.ui.outputsamplerateentry.text())*1000 #use integer multiples of 4166.67 ie 2083.33 or 1041.67
                

        if str(os.path.splitext(self.datafilename)[1])=='.log':         
            self.data=np.fromfile(self.datafilename,self.CHIMERAfile) 
                              
            self.matfilename=str(os.path.splitext(self.datafilename)[0])       
            self.mat = io.loadmat(self.matfilename)
      
            
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
            self.data = 10**9*(self.data/closedloop_gain + currentoffset)
            
            if os.name=='posix':
                self.data=self.data[0]
           
            ###############################################data has now been loaded
            ###############################################now filtering data
            
            Wn = round(self.LPfiltercutoff/(samplerate/2),4)
            b,a = signal.bessel(4, Wn, btype='low');
            
            self.data = signal.filtfilt(b,a,self.data)

        if str(os.path.splitext(self.datafilename)[1])=='.opt': 
            self.data=np.fromfile(self.datafilename, dtype = np.dtype('>d'))            
#            self.outputsamplerate  =250e3                
            self.data=self.data*10**9
            
                        
            Wn = round(self.LPfiltercutoff/(100*10**3/2),4)
            b,a = signal.bessel(4, Wn, btype='low');
            
            self.data = signal.filtfilt(b,a,self.data)

        if str(os.path.splitext(self.datafilename)[1])=='.txt':
            self.data=pandas.io.parsers.read_csv(self.datafilename,skiprows=1)
            self.data=np.reshape(np.array(self.data),np.size(self.data))*10**9
        
        self.t=np.arange(0,len(self.data))
        self.t=self.t/self.outputsamplerate
        
        if self.hasbaselinebeenset==0:
            self.baseline=np.median(self.data)  
            self.var=np.std(self.data)
        self.ui.eventcounterlabel.setText('Baseline='+str(round(self.baseline,2))+' nA')

        #skips plotting first and last two points, there was a weird spike issue
        self.p1.plot(self.t[::10][2:][:-2],self.data[::10][2:][:-2],pen='b')
        self.p1.addLine(y=self.baseline,pen='g')
        self.p1.addLine(y=self.threshold,pen='r')
        self.p1.autoRange()
        
#        del b,a,Wn,closedloop_gain,ADCvref,currentoffset,preADCgain,TIAgain,
#        samplerate,self.LPfiltercutoff

    def getfile(self):        
#        self.p1 = self.ui.signalplot.addPlot()
        self.CHIMERAfile = np.dtype('<u2')
               
        
        if self.direc==[]:
            self.datafilename = str(QtGui.QFileDialog.getOpenFileName(self, 'Open file',os.getcwd(),("*.log;*.opt;*.txt"))  )
            self.direc=os.path.dirname(self.datafilename)            
        else:
            self.datafilename = str(QtGui.QFileDialog.getOpenFileName(self,'Open file',self.direc,("*.log;*.opt;*.txt"))) 
            self.direc=os.path.dirname(self.datafilename)   
#        filelistsize=np.size()
        
        self.Load() 
        
    def analyze(self):
        global startpoints,endpoints, mins

        self.threshold = np.float64(self.ui.thresholdentry.text())  
        
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
        if endpoints [-1] == len(self.data)-1:
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
                if j == numberofevents - 1: # if this is the last event, check to make
                    if ep == len(self.data) -1:  # sure that the current returns to baseline
                        endpoints[j] = 0              # before file ends. If not, mark points for
                        startpoints[j] = 0              # deletion and break from loop
                        ep = 0                        
                        break
                elif ep > startpoints[j+1]: # if we hit the next startpoint before we
                    startpoints[j+1] = 0    # return to baseline, mark for deletion
                    endpoints[j] = 0                  # and break out of loop  
                    ep = 0                    
                    break
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
#                self.deli[i] = self.bwlanalysis(i, 1000)
                endpoints[i] = mins[0]
            elif len(mins) > 1:
                self.deli[i] = self.baseline - np.mean(self.data[mins[0]:mins[-1]])
#                self.dwell[i] = (endpoints[i]-startpoints[i])*1e6/self.outputsamplerate
#                self.deli[i] = self.bwlanalysis(i, 100)
                endpoints[i] = mins[-1]
                self.dwell[i] = (endpoints[i]-startpoints[i])*1e6/self.outputsamplerate
                

        startpoints = startpoints[self.deli!=0]
        endpoints = endpoints[self.deli!=0]  
        self.deli = self.deli[self.deli!=0]
        self.dwell = self.dwell[self.dwell!=0]
        self.frac = self.deli/self.baseline 
        self.dt = np.array(0)
        self.dt=np.append(self.dt,np.diff(startpoints)/self.outputsamplerate)
        
        
        self.p1.clear()       
        
        #skips plotting first and last two points, there was a weird spike issue
        self.p1.plot(self.t[::10][2:][:-2],self.data[::10][2:][:-2],pen='b')
        self.p1.plot(self.t[startpoints], self.data[startpoints],pen=None, symbol='o',symbolBrush='g',symbolSize=5)
        self.p1.plot(self.t[endpoints], self.data[endpoints], pen=None, symbol='o',symbolBrush='r',symbolSize=5)
        
        self.ui.eventcounterlabel.setText('Events:'+str(self.numberofevents))
        self.ui.meandelilabel.setText('Deli:'+str(round(np.mean(self.deli),2))+' nA')
#        self.ui.meandwelllabel.setText('Dwell:'+str(round(np.e**mean(log(self.dwell)),2))+ u' μs')
        self.ui.meandwelllabel.setText('Dwell:'+str(round(np.median(self.dwell),2))+ u' μs')
        self.ui.meandtlabel.setText('Rate:'+str(round(self.numberofevents/self.t[-1],1))+' events/s')

#        self.p2.addPoints(x=np.log10(self.dwell),y=self.deli, symbol='o',brush='b')
        self.p2.addPoints(x=np.log10(self.dwell),y=self.frac, symbol='o',brush=(self.cb.color()), pen = None)

        self.w1.addItem(self.p2)
        self.w1.setLogMode(x=True,y=False)
        self.p1.autoRange()
        self.w1.autoRange()
        self.ui.scatterplot.update()
        self.w1.setRange(yRange=[0,1])
        
        self.save()

    def save(self):  
         np.savetxt(self.matfilename+'DB.txt',np.column_stack((self.deli,self.frac,self.dwell,self.dt)),delimiter='\t')

    def inspectevent(self):
                
        #Reset plot
        self.p3.showAxis('bottom')
        self.p3.showAxis('left')
        self.numberofevents=len(self.dt)
        self.totalplotpoints = len(self.p2.data)
        self.p3.clear()
        
        #Correct for user error if non-extistent number is entered        
        eventbuffer=np.int(self.ui.eventbufferentry.text())
        eventnumber=np.int(self.ui.eventnumberentry.text())
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

        #Mark event that is being viewed on scatter plot
        colors=[]
        colors[0:self.totalplotpoints-self.numberofevents]=[pg.mkColor(128,128,128,50)]*self.totalplotpoints
        colors[self.totalplotpoints-self.numberofevents:self.totalplotpoints]=[self.cb.color()]*self.numberofevents
        colors[self.totalplotpoints-self.numberofevents+eventnumber]= pg.mkColor('r')
        self.p2.setBrush(colors, mask=None)


        #Mark event start and end points
        self.p3.plot([self.t[startpoints[eventnumber]], self.t[startpoints[eventnumber]]],[self.data[startpoints[eventnumber]], self.data[startpoints[eventnumber]]],pen=None, symbol='o',symbolBrush='g',symbolSize=7)
        self.p3.plot([self.t[endpoints[eventnumber]], self.t[endpoints[eventnumber]]],[self.data[endpoints[eventnumber]], self.data[endpoints[eventnumber]]],pen=None, symbol='o',symbolBrush='r',symbolSize=7)

        self.ui.eventinfolabel.setText('Dwell Time=' + str(round(self.dwell[eventnumber],2))+ u' μs,   Deli='+str(round(self.deli[eventnumber],2)) +' nA')
        self.lastevent=eventnumber
        self.p3.autoRange()

 ########################################################################   
        
#        x=self.data[startpoints[eventnumber]-eventbuffer:endpoints[eventnumber]+eventbuffer]
#        mins=signal.argrelmin(x)[0]
#        drift=.0
#        self.fitthreshold=np.float64(self.ui.levelthresholdentry.text()) 
#        eventfit=np.array((0))
#        
#        gp, gn = np.zeros(x.size), np.zeros(x.size)
#        ta, tai, taf = np.array([[], [], []], dtype=int)
#        tap, tan = 0, 0
#        # Find changes (online form)
#        for i in range(mins[0], mins[-1]):
#            s = x[i] - x[i-1]
#            gp[i] = gp[i-1] + s - drift  # cumulative sum for + change
#            gn[i] = gn[i-1] - s - drift  # cumulative sum for - change
#            if gp[i] < 0:
#                gp[i], tap = 0, i
#            if gn[i] < 0:
#                gn[i], tan = 0, i
#            if gp[i] > self.fitthreshold or gn[i] > self.fitthreshold:  # change detected!
#                ta = np.append(ta, i)    # alarm index
#                tai = np.append(tai, tap if gp[i] > self.fitthreshold else tan)  # start
#                gp[i], gn[i] = 0, 0      # reset alarm                
#        
#        eventfit=np.repeat(np.array(self.baseline),ta[0])
#        for i in range(1,ta.size):
#            eventfit=np.concatenate((eventfit,np.repeat(np.array(np.mean(x[ta[i-1]:ta[i]])),ta[i]-ta[i-1])))
#        eventfit=np.concatenate((eventfit,np.repeat(np.array(self.baseline),x.size-ta[-1])))
#        self.p3.plot(self.t[startpoints[eventnumber]-eventbuffer:endpoints[eventnumber]+eventbuffer],eventfit
#            ,pen=pg.mkPen(color=(255,255,0),width=3))
##        pg.plot(eventfit)
#        
#        
#        self.p3.plot(self.t[ta+startpoints[eventnumber]-eventbuffer],x[ta],pen=None,symbol='o',symbolBrush='m',symbolSize=8)
        
        
 ########################################################################   

       
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
        if self.lr==[]:
            self.p1.clear()
            self.lr = pg.LinearRegionItem()
            self.lr.hide()
            self.p1.addItem(self.lr)     
            
            self.p1.plot(self.t[::100],self.data[::100],pen='b')
            self.lr.show()

        else:    
            cutregion=self.lr.getRegion()
            self.p1.clear()
            self.data=np.delete(self.data,np.arange(np.int(cutregion[0]*self.outputsamplerate),np.int(cutregion[1]*self.outputsamplerate)))
            
            self.t=np.arange(0,len(self.data))
            self.t=self.t/self.outputsamplerate
            
            if self.hasbaselinebeenset==0:
                self.baseline=np.median(self.data)  
                self.var=np.std(self.data)
                self.ui.eventcounterlabel.setText('Baseline='+str(round(self.baseline,2))+' nA')

            
            self.p1.plot(self.t[::10][2:][:-2],self.data[::10][2:][:-2],pen='b')
            self.p1.addLine(y=self.baseline,pen='g')
            self.p1.addLine(y=self.threshold,pen='r')
            self.lr=[]
#            self.p1.autoRange()
        
            
    def baselinecalc(self):
        if self.lr==[]:
            self.p1.clear()
            self.lr = pg.LinearRegionItem()
            self.lr.hide()
            self.p1.addItem(self.lr)     
            
            self.p1.plot(self.t[::100],self.data[::100],pen='b')
            self.lr.show()

        else:      
            calcregion=self.lr.getRegion()
            self.p1.clear()
            
            self.baseline=np.median(self.data[np.arange(np.int(calcregion[0]*self.outputsamplerate),np.int(calcregion[1]*self.outputsamplerate))])  
            self.var=np.std(self.data[np.arange(np.int(calcregion[0]*self.outputsamplerate),np.int(calcregion[1]*self.outputsamplerate))])
            self.p1.plot(self.t[::10][2:][:-2],self.data[::10][2:][:-2],pen='b')
            self.p1.addLine(y=self.baseline,pen='g')
            self.p1.addLine(y=self.threshold,pen='r')
            self.lr=[]
            self.hasbaselinebeenset=1
            self.ui.eventcounterlabel.setText('Baseline='+str(round(self.baseline,2))+' nA')
            self.p1.autoRange()
            
            
    def clearscatter(self):
        self.p2.setData(x=[],y=[])
        self.lastevent=[]
        self.ui.scatterplot.update()
        
    def deleteevent(self):
        global startpoints,endpoints
        eventnumber=np.int(self.ui.eventnumberentry.text())
        if eventnumber>self.numberofevents:
            eventnumber=self.numberofevents-1
            self.ui.eventnumberentry.setText(str(eventnumber))
        self.deli=np.delete(self.deli,eventnumber)
        self.dwell=np.delete(self.dwell,eventnumber)
        self.dt=np.delete(self.dt,eventnumber)
        self.frac=np.delete(self.frac,eventnumber)
        startpoints=np.delete(startpoints,eventnumber)
        endpoints=np.delete(endpoints,eventnumber)
        self.p2.data=np.delete(self.p2.data,self.totalplotpoints-self.numberofevents+eventnumber)
        
      
        numberofevents=len(self.dt)
        self.ui.eventcounterlabel.setText('Events:'+str(numberofevents))
 
        self.inspectevent()
        
    def invertdata(self):
        self.p1.clear()
        self.data=-self.data
        
        if self.hasbaselinebeenset==0:
            self.baseline=np.median(self.data)  
            self.var=np.std(self.data)
            
        self.p1.plot(self.t[::10],self.data[::10],pen='b')
        self.p1.addLine(y=self.baseline,pen='g')
        self.p1.addLine(y=self.threshold,pen='r')
        self.p1.autoRange()        
        
    def clicked(self, plot, points):
        self.totalplotpoints=len(self.p2.data)
        self.numberofevents=len(self.dt)
        colors=[]
        colors[0:self.totalplotpoints-self.numberofevents]=[pg.mkColor(.5)]*self.totalplotpoints
        colors[self.totalplotpoints-self.numberofevents:self.totalplotpoints]=[pg.mkColor('b')]*self.numberofevents
        self.p2.setBrush(colors, mask=None)
        
        
        for p in points:
            p.setBrush(pg.mkColor('r'))
            

        i=0
        while i < self.totalplotpoints :
            if self.p2.data[i][5]==pg.mkColor('b') or self.p2.data[i][5]==None or self.p2.data[i][5]==pg.mkColor(.5):
                i=i+1
            else:
                if i<self.totalplotpoints-self.numberofevents:
                    print 'Event is from an earlier file, not clickable'
                    break
                
                i=i+self.numberofevents-self.totalplotpoints
                self.ui.eventnumberentry.setText(str(i))
                self.inspectevent()                
                i=self.totalplotpoints
                
        
            
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
        startindex=self.matfilename[-6::]
        filebase=self.matfilename[0:len(self.matfilename)-6]
        nextindex=str(int(startindex)+1)
        while os.path.isfile(filebase+nextindex+'.log')==False:
            nextindex=str(int(nextindex)+1)
            if int(nextindex)>int(startindex)+1000:
                print 'no such file'                
                break
        self.datafilename=(filebase+nextindex+'.log')
        self.Load()
        
    def previousfile(self):
        startindex=self.matfilename[-6::]
        filebase=self.matfilename[0:len(self.matfilename)-6]
        nextindex=str(int(startindex)-1)
        while os.path.isfile(filebase+nextindex+'.log')==False:
            nextindex=str(int(nextindex)-1)
            if int(nextindex)<int(startindex)-1000:
                print 'no such file'                
                break
        self.datafilename=(filebase+nextindex+'.log')
        self.Load()
        
    def savetrace(self):
        self.data.astype('d').tofile(self.matfilename+'_trace.bin')
        
    def showcattrace(self):
        global eventstruct
        eventbuffer=np.int(self.ui.eventbufferentry.text())
        numberofevents=len(self.dt)
        self.catdata=self.data[startpoints[0]-eventbuffer:endpoints[0]+eventbuffer]
        self.catfits=np.concatenate((np.repeat(np.array([self.baseline]),eventbuffer),np.repeat(np.array([
            self.baseline-self.deli[0]]),endpoints[0]-startpoints[0]),
            np.repeat(np.array([self.baseline]),eventbuffer)),0)
        eventstruct = self.catdata                  
        
        for i in range(numberofevents):
            if i<numberofevents-1:
                if endpoints[i]+eventbuffer>startpoints[i+1]:
                    print 'overlapping event'
                else:
                    self.catdata=np.concatenate((self.catdata,self.data[startpoints[i]-eventbuffer:endpoints[i]+eventbuffer]),0)
                    self.catfits=np.concatenate((self.catfits,np.concatenate((np.repeat(np.array([self.baseline]),eventbuffer),np.repeat(np.array([
                        self.baseline-self.deli[i]]),endpoints[i]-startpoints[i]),np.repeat(np.array([self.baseline]),eventbuffer)),0)),0)
                    eventstruct  = [np.array(self.data[startpoints[i]-eventbuffer:endpoints[i]+eventbuffer]) for i in range(numberofevents)]
                    
        self.tcat=np.arange(0,len(self.catdata))
        self.tcat=self.tcat/self.outputsamplerate        
        
        self.p1.clear()
        self.p1.plot(self.tcat,self.catdata,pen='b')
        self.p1.plot(self.tcat,self.catfits,pen=pg.mkPen(color=(173,27,183),width=1))
        self.p1.autoRange()                
        
 
    def savecattrace(self):
        if self.catdata==[]:
            self.showcattrace
#        self.catdata=self.catdata[::10]
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
        
    def saveeventfits(self):
        if self.catdata==[]:
            self.showcattrace
        self.catdata.astype('d').tofile(self.matfilename+'_cattrace.bin')  
        
        
#_________________________________________________________________________
    def objfunc(self, params, t, data):
    
        tau = params['tau'].value
        mu1 = params['mu1'].value
        mu2 = params['mu2'].value
        a = params['a'].value
        b = params['b'].value
    		    
        model = self.stepResponseFunc(t, tau, mu1, mu2, a, b)
    
        return model - data
        
    def heaviside(self, x):
    	out=np.array(x)
    
    	out[out==0]=0.5
    	out[out<0]=0
    	out[out>0]=1
    
    	return out
    	
    def stepResponseFunc(self, t, tau, mu1, mu2, a, b):
    	try:
    		t1=(np.exp((mu1-t)/tau)-1)*self.heaviside(t-mu1)
    		t2=(1-np.exp((mu2-t)/tau))*self.heaviside(t-mu2)
    
    		# Either t1, t2 or both could contain NaN due to fixed precision arithmetic errors.
    		# In this case, we can set those values to zero.
    		t1[np.isnan(t1)]=0
    		t2[np.isnan(t2)]=0
    
    		return a*( t1+t2 ) + b
    	except:
    		raise       
      
    def bwlanalysis(self, eventnumber, eventbuffer):
        params=Parameters()        
#        params.add('mu1', value= self.t[startpoints[eventnumber]], vary=True)
#        params.add('mu2', value=self.t[endpoints[eventnumber]], vary=True)
#        p1 = params.add('a', value=(self.baseline - self.deli[eventnumber]), vary=True)
#        p1.set(max =self.baseline - self.deli[eventnumber])
#        params.add('b', value = self.baseline,vary = True)
#        params.add('tau', value = 1/4166.67e3, vary=True)
        
        upperbound = self.baseline - self.deli[eventnumber] + 2*self.var
        if self.baseline - self.deli[eventnumber]-2*self.var > 0:
            lowerbound = self.baseline - self.deli[eventnumber] - 2*self.var
        else:
            lowerbound = 0
        
        params.add_many(('mu1', self.t[startpoints[eventnumber]],  True, None, None,  None),
           ('mu2',   self.t[endpoints[eventnumber]],  True,  None,  None,  None),
           ('a',   self.baseline - self.deli[eventnumber],  True, lowerbound, upperbound,  None),
           ('b',   self.baseline,  True, self.baseline-2*self.var, self.baseline+2*self.var,  None),
           ('tau',  1/self.LPfiltercutoff,  False,  None,  None,  None))
        
        
        optfit = Minimizer(self.objfunc, params, fcn_args=(self.t[startpoints[eventnumber]-eventbuffer:endpoints[eventnumber]+eventbuffer],
                     self.data[startpoints[eventnumber]-eventbuffer:endpoints[eventnumber]+eventbuffer],))
        optfit.prepare_fit()
        optfit.leastsq()
        
        return optfit.params['b'].value  - optfit.params['a'].value        
 

def start():
    app = QtGui.QApplication(sys.argv)
    myapp = GUIForm()
    myapp.show()
    sys.exit(app.exec_())

 
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    myapp = GUIForm()
    myapp.show()
    sys.exit(app.exec_())
    
    
