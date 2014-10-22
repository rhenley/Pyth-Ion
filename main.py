# -*- coding: utf8 -*-
import sys
import numpy as np
import scipy as sp
from scipy import signal
import scipy.io
from PlotGUI import *
import pyqtgraph as pg
 
class GUIForm(QtGui.QMainWindow):
    
 
    def __init__(self, master=None):
        QtGui.QMainWindow.__init__(self,master)
        self.ui = Ui_PythIon()
        self.ui.setupUi(self)
        
        QtCore.QObject.connect(self.ui.loadbutton, QtCore.SIGNAL('clicked()'), self.getfile)
        QtCore.QObject.connect(self.ui.analyzebutton, QtCore.SIGNAL('clicked()'), self.analyze)
        QtCore.QObject.connect(self.ui.savebutton, QtCore.SIGNAL('clicked()'), self.save)
        QtCore.QObject.connect(self.ui.cutbutton, QtCore.SIGNAL('clicked()'), self.cut)
        QtCore.QObject.connect(self.ui.baselinebutton, QtCore.SIGNAL('clicked()'), self.baselinecalc)
        QtCore.QObject.connect(self.ui.clearscatterbutton, QtCore.SIGNAL('clicked()'), self.clearscatter)
        QtCore.QObject.connect(self.ui.deleteeventbutton, QtCore.SIGNAL('clicked()'), self.deleteevent)
        QtCore.QObject.connect(self.ui.invertbutton, QtCore.SIGNAL('clicked()'), self.invertdata)
        QtCore.QObject.connect(self.ui.concatenatebutton, QtCore.SIGNAL('clicked()'), self.concatenatetext)
        QtCore.QObject.connect(self.ui.nextfilebutton, QtCore.SIGNAL('clicked()'), self.nextfile)
        QtCore.QObject.connect(self.ui.previousfilebutton, QtCore.SIGNAL('clicked()'), self.previousfile)
 
        
        QtCore.QObject.connect(self.ui.gobutton, QtCore.SIGNAL('clicked()'), self.inspectevent)
        QtCore.QObject.connect(self.ui.previousbutton, QtCore.SIGNAL('clicked()'), self.previousevent)
        QtCore.QObject.connect(self.ui.nextbutton, QtCore.SIGNAL('clicked()'), self.nextevent)
                

        self.ui.signalplot.setBackground('w')
        self.ui.scatterplot.setBackground('w')
        self.ui.eventplot.setBackground('w')
        
        self.p1 = self.ui.signalplot.addPlot()
        self.p1.setLabel('bottom', text='Time', units='s')
        self.p1.setLabel('left', text='Current', units='nA')
        
#        self.p2 = self.ui.scatterplot.addPlot()

        self.w1 = self.ui.scatterplot.addPlot()
        self.p2=pg.ScatterPlotItem()
        self.p2.sigClicked.connect(self.clicked)
        self.w1.addItem(self.p2)
        self.w1.setLabel('bottom', text='Time', units=u'μs')
        self.w1.setLabel('left', text='Current', units='nA')
        self.w1.setLogMode(x=True,y=False)
        
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

 
    def Load(self): 
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

        colors=[]
        colors[0:self.totalplotpoints]=[.5]*self.totalplotpoints
        self.p2.setBrush(colors, mask=None)                  
        
        self.data=np.fromfile(self.datafilename,self.CHIMERAfile) 
        
        self.LPfiltercutoff = np.float64(self.ui.LPentry.text())*1000
        self.outputsamplerate = np.float64(self.ui.outputsamplerateentry.text())*1000 #use integer multiples of 4166.67 ie 2083.33 or 1041.67
        self.threshold=np.float64(self.ui.thresholdentry.text())
        
        self.ui.filelabel.setText(self.datafilename)        
        self.matfilename=str(os.path.splitext(self.datafilename)[0])       
        self.mat = scipy.io.loadmat(self.matfilename)

        print self.datafilename   
        
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
        
        del b,a,Wn,closedloop_gain,ADCvref,currentoffset,preADCgain,TIAgain,
        samplerate,self.LPfiltercutoff

    def getfile(self):        
#        self.p1 = self.ui.signalplot.addPlot()
        self.CHIMERAfile = np.dtype('<u2')
               
        
        if self.direc==[]:
            self.datafilename = str(QtGui.QFileDialog.getOpenFileName(self, 'Open file','*.log')  )
            self.direc=os.path.dirname(self.datafilename)            
        else:
            self.datafilename = str(QtGui.QFileDialog.getOpenFileName(self,'Open file',self.direc,'*.log')) 
            self.direc=os.path.dirname(self.datafilename)   
#        filelistsize=np.size()
        
        
        self.Load()
        
    def analyze(self):
        global startpoints,endpoints

        self.threshold=np.float64(self.ui.thresholdentry.text())   
        
        below=np.where(self.data<self.threshold)[0]
        startandend=np.diff(below)
        startandend[len(startandend)-1]=1
        startandend=np.where(startandend>1)[0]
        endpoints=below[startandend]
        startpoints=below[startandend+1]
        startpoints=np.append(below[0],startpoints)
        endpoints=np.append(endpoints,below[len(below)-1])
        numberofevents=len(startpoints)
#        print numberofevents 
        
        
        thresholdcrossingdown=startpoints
        thresholdcrossingup=endpoints
        highthresh=self.baseline-self.var
        j=0
        while j<len(startpoints):
            i=startpoints[j]
            while self.data[i]< highthresh:
                i=i-1
                if j!=0:
                    if i<endpoints[j-1]:
                        i=0
                        break
            startpoints[j]=i
            j=j+1
        j=0
        endpoints=endpoints[startpoints!=0]
        thresholdcrossingdown=thresholdcrossingdown[startpoints!=0]
        thresholdcrossingup=thresholdcrossingup[startpoints!=0]
        startpoints=startpoints[startpoints!=0]
        while j<len(endpoints):
            i=endpoints[j]
            while self.data[i]< highthresh:
                if i==len(self.data)-1:
                    i=1
                    break                                
                i=i+1
            i=i-1
            endpoints[j]=i
            j=j+1
        startpoints=startpoints[endpoints!=0]
        thresholdcrossingdown=thresholdcrossingdown[endpoints!=0]
        thresholdcrossingup=thresholdcrossingup[endpoints!=0]
        endpoints=endpoints[endpoints!=0]
        numberofevents=len(startpoints) 
#        print(numberofevents)
        
        
        
        
        
        
            
        self.dwell=np.zeros(numberofevents)
        self.deli=np.zeros(numberofevents)  
        self.dt=np.zeros(numberofevents)
        
        i=0
        while i<numberofevents:
#            mins=signal.argrelmin(self.data[thresholdcrossingdown[i]:thresholdcrossingup[i]],mode='wrap')[0]+thresholdcrossingdown[i]           
            mins=signal.argrelmin(self.data[startpoints[i]:endpoints[i]],mode='wrap')[0]+startpoints[i]           
            while len(mins)==0:
                startpoints=np.delete(startpoints,i)
                endpoints=np.delete(endpoints,i)
                numberofevents=numberofevents-1
                mins=signal.argrelmin(self.data[startpoints[i]:endpoints[i]],mode='wrap')[0]+startpoints[i]           
            if mins[0]==0:
                startpoints=np.delete(startpoints,i)
                endpoints=np.delete(endpoints,i)
                numberofevents=numberofevents-1
                mins=signal.argrelmin(self.data[startpoints[i]:endpoints[i]],mode='wrap')[0]+startpoints[i]       
            else:
                mins=mins[self.data[mins]<self.threshold]            
            if len(mins)==1: 
                if mins[-1]==startpoints[i]:
                    self.deli[i]=-1
                    self.dwell[i]=-1
                else:
                    self.deli[i]=self.baseline-np.mean(self.data[mins[0]-1:mins[0]+1])
                    self.dwell[i]=(mins[-1]-startpoints[i])/self.outputsamplerate*1e6
                    endpoints[i]=mins[-1]
            if len(mins)>1:
                if mins[-1]==startpoints[i]:
                    self.deli[i]=-1
                    self.dwell[i]=-1
                else:
                    self.deli[i]=self.baseline-np.mean(self.data[mins[0]:mins[-1]])
                    self.dwell[i]=(mins[-1]-startpoints[i])/self.outputsamplerate*1e6
                    endpoints[i]=mins[-1]
            i=i+1
        
        self.deli=self.deli[self.deli!=-1]
        self.dwell=self.dwell[self.dwell!=-1]
        self.dt=np.diff(startpoints)/self.outputsamplerate
        self.deli=self.deli[self.deli!=0]
        self.dwell=self.dwell[self.dwell!=0]
        self.dt=self.dt[self.dt!=0]
        self.dt=np.append(0,self.dt)
        self.frac=self.deli/self.baseline
        
#        print self.dwell, self.deli, self.dt
        
        self.p1.clear()       
        
        #skips plotting first and last two points, there was a weird spike issue
        self.p1.plot(self.t[::10][2:][:-2],self.data[::10][2:][:-2],pen='b')
        self.p1.plot(self.t[startpoints], self.data[startpoints],pen=None, symbol='o',symbolBrush='g',symbolSize=5)
        self.p1.plot(self.t[endpoints], self.data[endpoints], pen=None, symbol='o',symbolBrush='r',symbolSize=5)
        
        self.ui.eventcounterlabel.setText('Events:'+str(numberofevents))
        self.ui.meandelilabel.setText('Deli:'+str(round(np.mean(self.deli),2))+' nA')
#        self.ui.meandwelllabel.setText('Dwell:'+str(round(np.e**mean(log(self.dwell)),2))+ u' μs')
        self.ui.meandwelllabel.setText('Dwell:'+str(round(np.median(self.dwell),2))+ u' μs')
        self.ui.meandtlabel.setText('Rate:'+str(round(numberofevents/self.t[-1],1))+' events/s')

        self.p2.addPoints(x=np.log10(self.dwell),y=self.deli, symbol='o',brush='b')
        self.w1.addItem(self.p2)
        self.w1.setLogMode(x=True,y=False)
        self.p1.autoRange()
        self.p2.autoRange()
        self.ui.scatterplot.update()
        

    def save(self):  
         np.savetxt(self.matfilename+'DB.txt',np.column_stack((self.deli,self.frac,self.dwell,self.dt)),delimiter='\t')

    def inspectevent(self):
        self.p3.showAxis('bottom')
        self.p3.showAxis('left')
        self.numberofevents=len(self.dt)
        self.totalplotpoints=len(self.p2.data)
        self.p3.clear()
        
        eventbuffer=np.int(self.ui.eventbufferentry.text())
        eventnumber=np.int(self.ui.eventnumberentry.text())
        if eventnumber>=self.numberofevents:
            eventnumber=self.numberofevents-1
            self.ui.eventnumberentry.setText(str(eventnumber))
        self.p3.plot(self.t[startpoints[eventnumber]-eventbuffer:endpoints[eventnumber]+eventbuffer],self.data[startpoints[eventnumber]-eventbuffer:endpoints[eventnumber]+eventbuffer],pen='b')
        self.p3.addLine(y=self.baseline-self.deli[eventnumber],pen=(173,27,183))

        colors=[]
        colors[0:self.totalplotpoints-self.numberofevents]=[.5]*self.totalplotpoints
        colors[self.totalplotpoints-self.numberofevents:self.totalplotpoints]=['b']*self.numberofevents
        colors[self.totalplotpoints-self.numberofevents+eventnumber]='r'
        self.p2.setBrush(colors, mask=None)

        self.p3.plot([self.t[startpoints[eventnumber]], self.t[startpoints[eventnumber]]],[self.data[startpoints[eventnumber]], self.data[startpoints[eventnumber]]],pen=None, symbol='o',symbolBrush='g',symbolSize=5)
        self.p3.plot([self.t[endpoints[eventnumber]], self.t[endpoints[eventnumber]]],[self.data[endpoints[eventnumber]], self.data[endpoints[eventnumber]]],pen=None, symbol='o',symbolBrush='r',symbolSize=5)

        self.ui.eventinfolabel.setText('Dwell Time=' + str(round(self.dwell[eventnumber],2))+ u' μs,   Deli='+str(round(self.deli[eventnumber],2)) +' nA')
        self.lastevent=eventnumber
        self.p3.autoRange()
        
    def nextevent(self):
        self.p3.showAxis('bottom')
        self.p3.showAxis('left')
        self.numberofevents=len(self.dt)
        self.totalplotpoints=len(self.p2.data)
        self.p3.clear()
        eventnumber=np.int(self.ui.eventnumberentry.text())
        eventbuffer=np.int(self.ui.eventbufferentry.text())


        if eventnumber>=self.numberofevents-1:  
            eventnumber=0
        else:
            eventnumber=np.int(self.ui.eventnumberentry.text())+1  

        self.p3.plot(self.t[startpoints[eventnumber]-eventbuffer:endpoints[eventnumber]+eventbuffer],self.data[startpoints[eventnumber]-eventbuffer:endpoints[eventnumber]+eventbuffer],pen='b')
        self.p3.addLine(y=self.baseline-self.deli[eventnumber],pen=(173,27,183))
        self.ui.eventnumberentry.setText(str(eventnumber))
        
        #cant plot only one item? so I doubled it
        
        colors=[]
        colors[0:self.totalplotpoints-self.numberofevents]=[.5]*self.totalplotpoints
        colors[self.totalplotpoints-self.numberofevents:self.totalplotpoints]=['b']*self.numberofevents
        colors[self.totalplotpoints-self.numberofevents+eventnumber]='r'
        self.p2.setBrush(colors, mask=None)
        
        
        self.p3.plot([self.t[startpoints[eventnumber]], self.t[startpoints[eventnumber]]],[self.data[startpoints[eventnumber]], self.data[startpoints[eventnumber]]],pen=None, symbol='o',symbolBrush='g',symbolSize=5)
        self.p3.plot([self.t[endpoints[eventnumber]], self.t[endpoints[eventnumber]]],[self.data[endpoints[eventnumber]], self.data[endpoints[eventnumber]]],pen=None, symbol='o',symbolBrush='r',symbolSize=5)

        self.ui.eventinfolabel.setText('Dwell Time=' + str(round(self.dwell[eventnumber],2))+ u' μs,   Deli='+str(round(self.deli[eventnumber],2)) +' nA')
        self.lastevent=eventnumber
        self.p3.autoRange()
        
    def previousevent(self):      
        self.p3.showAxis('bottom')
        self.p3.showAxis('left')
        self.numberofevents=len(self.dt)
        self.totalplotpoints=len(self.p2.data)
        self.p3.clear()
        eventbuffer=np.int(self.ui.eventbufferentry.text())
        
        
        eventnumber=np.int(self.ui.eventnumberentry.text())-1
        if eventnumber<0:
            eventnumber=self.numberofevents-1
        self.p3.plot(self.t[startpoints[eventnumber]-eventbuffer:endpoints[eventnumber]+eventbuffer],self.data[startpoints[eventnumber]-eventbuffer:endpoints[eventnumber]+eventbuffer],pen='b')
        self.p3.addLine(y=self.baseline-self.deli[eventnumber],pen=(173,27,183))
        self.ui.eventnumberentry.setText(str(eventnumber)  )


        colors=[]
        colors[0:self.totalplotpoints-self.numberofevents]=[.5]*self.totalplotpoints
        colors[self.totalplotpoints-self.numberofevents:self.totalplotpoints]=['b']*self.numberofevents
        colors[self.totalplotpoints-self.numberofevents+eventnumber]='r'
        self.p2.setBrush(colors, mask=None)



        self.p3.plot([self.t[startpoints[eventnumber]], self.t[startpoints[eventnumber]]],[self.data[startpoints[eventnumber]], self.data[startpoints[eventnumber]]],pen=None, symbol='o',symbolBrush='g',symbolSize=5)
        self.p3.plot([self.t[endpoints[eventnumber]], self.t[endpoints[eventnumber]]],[self.data[endpoints[eventnumber]], self.data[endpoints[eventnumber]]],pen=None, symbol='o',symbolBrush='r',symbolSize=5)

        self.ui.eventinfolabel.setText('Dwell Time=' + str(round(self.dwell[eventnumber],2))+ u' μs,   Deli='+str(round(self.deli[eventnumber],2)) +' nA')
        self.lastevent=eventnumber
        self.p3.autoRange()
        
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
            self.p1.autoRange()
        
            
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

#        self.p3.clear()
#        self.p3.showAxis('bottom')
#        self.p3.showAxis('left')        
#        
#        self.p2.setData(x=[],y=[])
#        
#        self.p3.plot(self.t[startpoints[eventnumber]-1000:endpoints[eventnumber]+1000]*1e6,self.data[startpoints[eventnumber]-1000:endpoints[eventnumber]+1000],pen='b')
#        self.p3.addLine(y=self.baseline-self.deli[eventnumber],pen=(173,27,183))
#
#        self.p2.addPoints(np.log10(self.dwell),self.deli, symbol='o',brush='b')
#        self.p2.addPoints(x=[np.log10(self.dwell[eventnumber]),np.log10(self.dwell[eventnumber])],y=[self.deli[eventnumber],self.deli[eventnumber]], symbol='o',brush='r')
#        self.totalplotpoints=len(self.p2.data)        
#        
#        colors=[]
#        colors[0:self.totalplotpoints-self.numberofevents]=[.5]*self.totalplotpoints
#        colors[self.totalplotpoints-self.numberofevents:self.totalplotpoints]=['b']*self.numberofevents
#        colors[self.totalplotpoints-self.numberofevents+eventnumber]='r'
#        self.p2.setBrush(colors, mask=None)
#        
#        self.p3.plot([self.t[startpoints[eventnumber]]*1e6, self.t[startpoints[eventnumber]]*1e6],[self.data[startpoints[eventnumber]], self.data[startpoints[eventnumber]]],pen=None, symbol='o',symbolBrush='g',symbolSize=5)
#        self.p3.plot([self.t[endpoints[eventnumber]]*1e6, self.t[endpoints[eventnumber]]*1e6],[self.data[endpoints[eventnumber]], self.data[endpoints[eventnumber]]],pen=None, symbol='o',symbolBrush='r',symbolSize=5)

        
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
        colors[0:self.totalplotpoints-self.numberofevents]=[.5]*self.totalplotpoints
        colors[self.totalplotpoints-self.numberofevents:self.totalplotpoints]=['b']*self.numberofevents
        self.p2.setBrush(colors, mask=None)
        
        
        for p in points:
            p.setBrush('r')
            

        i=0
        while i < self.totalplotpoints :
            if self.p2.data[i][5]=='b' or self.p2.data[i][5]==None or self.p2.data[i][5]==.5:
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

 
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    myapp = GUIForm()
    myapp.show()
    sys.exit(app.exec_())
    
    