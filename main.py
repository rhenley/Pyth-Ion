import sys
import Tkinter, tkFileDialog
import numpy as np
import os
import scipy as sp
from scipy import signal
import scipy.io
from Tkinter import *
from PlotGUI import *
import pyqtgraph as pg
 
class GUIForm(QtGui.QMainWindow):
 
    def __init__(self, master=None):
        QtGui.QMainWindow.__init__(self,master)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        QtCore.QObject.connect(self.ui.loadbutton, QtCore.SIGNAL('clicked()'), self.Load)
        QtCore.QObject.connect(self.ui.analyzebutton, QtCore.SIGNAL('clicked()'), self.analyze)
        QtCore.QObject.connect(self.ui.savebutton, QtCore.SIGNAL('clicked()'), self.save)
        QtCore.QObject.connect(self.ui.cutbutton, QtCore.SIGNAL('clicked()'), self.cut)
        QtCore.QObject.connect(self.ui.baselinebutton, QtCore.SIGNAL('clicked()'), self.baselinecalc)
        QtCore.QObject.connect(self.ui.clearscatterbutton, QtCore.SIGNAL('clicked()'), self.clearscatter)
        QtCore.QObject.connect(self.ui.deleteeventbutton, QtCore.SIGNAL('clicked()'), self.deleteevent)
        
        QtCore.QObject.connect(self.ui.gobutton, QtCore.SIGNAL('clicked()'), self.inspectevent)
        QtCore.QObject.connect(self.ui.previousbutton, QtCore.SIGNAL('clicked()'), self.previousevent)
        QtCore.QObject.connect(self.ui.nextbutton, QtCore.SIGNAL('clicked()'), self.nextevent)
                
        
        self.ui.scatterplot.setLabel('bottom', text='Dwell Time', units='us')
        self.ui.scatterplot.setLabel('left', text='Current Blocked', units='nA')
        
        self.ui.eventplot.setLabel('bottom', text='Time', units='us')
        self.ui.eventplot.setLabel('left', text='Current', units='nA')

        self.ui.signalplot.setBackground('w')
        self.ui.scatterplot.setBackground('w')
        self.ui.eventplot.setBackground('w')
        self.ui.scatterplot.plotItem.setLogMode(x=True,y=False)
        
        self.p1 = self.ui.signalplot.addPlot()
        self.p1.setLabel('bottom', text='Time', units='us')
        self.p1.setLabel('left', text='Current', units='nA')

 
    def Load(self):
        self.p1.clear()
#        self.p1 = self.ui.signalplot.addPlot()
        CHIMERAfile = np.dtype('<u2')
        
        root = Tkinter.Tk()
        root.withdraw()        
        
        LPfiltercutoff = np.float64(self.ui.LPentry.text())*1000
        self.outputsamplerate = np.float64(self.ui.outputsamplerateentry.text())*1000 #use integer multiples of 4166.67 ie 2083.33 or 1041.67
        self.threshold=np.float64(self.ui.thresholdentry.text())   
        
        datafilename = tkFileDialog.askopenfilename(filetypes=[("LOG files", "*.log")])
        self.ui.filelabel.setText(datafilename)

        self.matfilename=str(os.path.splitext(datafilename)[0])
        mat = scipy.io.loadmat(self.matfilename)
        self.data=np.fromfile(datafilename,CHIMERAfile) 
        
        samplerate = np.float64(mat['ADCSAMPLERATE'])
        TIAgain = np.int32(mat['SETUP_TIAgain'])
        preADCgain = np.float64(mat['SETUP_preADCgain'])
        currentoffset = np.float64(mat['SETUP_pAoffset'])
        ADCvref = np.float64(mat['SETUP_ADCVREF'])
        ADCbits = np.int32(mat['SETUP_ADCBITS'])
        closedloop_gain = TIAgain*preADCgain;
        
        if samplerate < 4000e3:
            self.data=self.data[::round(samplerate/self.outputsamplerate)]
        
        
        bitmask = (2**16 - 1) - (2**(16-ADCbits) - 1)
        self.data = -ADCvref + (2*ADCvref) * (self.data & bitmask) / 2**16
        self.data = 10**9*(self.data/closedloop_gain + currentoffset)
       
        ###############################################data has now been loaded
        ###############################################now filtering data
        
        Wn = round(LPfiltercutoff/(samplerate/2),4)
        b,a = signal.bessel(4, Wn, btype='low');
        
        self.data = signal.filtfilt(b,a,self.data)
        
        self.t=np.arange(0,len(self.data))
        self.t=self.t/self.outputsamplerate
        
        self.baseline=np.median(self.data)  
        self.var=std(self.data)
        self.p1.plot(self.t[::10],self.data[::10],pen='b')
        self.p1.addLine(y=self.baseline,pen='g')
        self.p1.addLine(y=self.threshold,pen='r')
        
        del b,a,Wn,closedloop_gain,ADCvref,currentoffset,preADCgain,TIAgain,
        samplerate,LPfiltercutoff
        
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
        print numberofevents 
        
        
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
        print(numberofevents)
        
        
        
        
        
        
            
        self.dwell=np.zeros(numberofevents)
        self.deli=np.zeros(numberofevents)  
        self.dt=np.zeros(numberofevents)
        
        i=0
        while i<numberofevents:
#            mins=signal.argrelmin(self.data[thresholdcrossingdown[i]:thresholdcrossingup[i]],mode='wrap')[0]+thresholdcrossingdown[i]           
            mins=signal.argrelmin(self.data[startpoints[i]:endpoints[i]],mode='wrap')[0]+startpoints[i]           
            if mins[0]==0:
                mins=mins    
            else:
                mins=mins[self.data[mins]<self.threshold]            
            if len(mins)==0:
                self.deli[i]=self.baseline-np.mean(self.data[thresholdcrossingdown[i]-thresholdcrossingup[i]])
                self.dwell[i]=(thresholdcrossingup[i]-startpoints[i])/self.outputsamplerate*1e6 
                print 'hmmm'
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
        
        self.deli=self.deli[self.deli!=1]
        self.dwell=self.dwell[self.dwell!=-1]
        self.dt=np.diff(startpoints) 
        self.dt=np.append(0,self.dt)
        
        self.p1.clear()       
        
        self.p1.plot(self.t[::10],self.data[::10],pen='b')
        self.p1.plot(self.t[startpoints], self.data[startpoints],pen=None, symbol='o',symbolBrush='g',symbolSize=5)
        self.p1.plot(self.t[endpoints], self.data[endpoints], pen=None, symbol='o',symbolBrush='r',symbolSize=5)
        
        self.ui.eventcounterlabel.setText('Events:'+str(numberofevents))

        self.ui.scatterplot.plot(self.dwell,self.deli,pen=None, symbol='o',symbolBrush='b',symbolSize=10)
#        self.p1.plot(self.t[::10],np.concatenate((np.diff(self.data[::10]),[0]),axis=0)*10)

    def save(self):  
         np.savetxt(self.matfilename+'DB.txt',np.column_stack((self.deli,self.dwell,self.dt)),delimiter='\t')

    def inspectevent(self):
        self.ui.eventplot.plotItem.clear()
        
        eventnumber=np.int(self.ui.eventnumberentry.text())   
        self.ui.eventplot.plot(self.t[startpoints[eventnumber]-1000:endpoints[eventnumber]+1000]*1e6,self.data[startpoints[eventnumber]-1000:endpoints[eventnumber]+1000],pen='b')
        self.ui.eventplot.plotItem.addLine(y=self.baseline-self.deli[eventnumber],pen=(173,27,183))
        self.ui.scatterplot.plot(self.dwell,self.deli,pen=None, symbol='o',symbolBrush='b',symbolSize=10)
        self.ui.scatterplot.plot([self.dwell[eventnumber],self.dwell[eventnumber]],[self.deli[eventnumber],self.deli[eventnumber]],pen=None, symbol='o',symbolBrush='r',symbolSize=10)
        self.ui.eventplot.plot([self.t[startpoints[eventnumber]]*1e6, self.t[startpoints[eventnumber]]*1e6],[self.data[startpoints[eventnumber]], self.data[startpoints[eventnumber]]],pen=None, symbol='o',symbolBrush='g',symbolSize=5)
        self.ui.eventplot.plot([self.t[endpoints[eventnumber]]*1e6, self.t[endpoints[eventnumber]]*1e6],[self.data[endpoints[eventnumber]], self.data[endpoints[eventnumber]]],pen=None, symbol='o',symbolBrush='r',symbolSize=5)


        
    def nextevent(self):       
        self.ui.eventplot.plotItem.clear()
        
        eventnumber=np.int(self.ui.eventnumberentry.text())+1   
        self.ui.eventplot.plot(self.t[startpoints[eventnumber]-1000:endpoints[eventnumber]+1000]*1e6,self.data[startpoints[eventnumber]-1000:endpoints[eventnumber]+1000],pen='b')
        self.ui.eventplot.plotItem.addLine(y=self.baseline-self.deli[eventnumber],pen=(173,27,183))
        self.ui.eventnumberentry.setText(str(eventnumber))
        
        #cant plot only one item? so I doubled it
        self.ui.scatterplot.plot(self.dwell,self.deli,pen=None, symbol='o',symbolBrush='b',symbolSize=10)
        self.ui.scatterplot.plot([self.dwell[eventnumber],self.dwell[eventnumber]],[self.deli[eventnumber],self.deli[eventnumber]],pen=None, symbol='o',symbolBrush='r',symbolSize=10)
        self.ui.eventplot.plot([self.t[startpoints[eventnumber]]*1e6, self.t[startpoints[eventnumber]]*1e6],[self.data[startpoints[eventnumber]], self.data[startpoints[eventnumber]]],pen=None, symbol='o',symbolBrush='g',symbolSize=5)
        self.ui.eventplot.plot([self.t[endpoints[eventnumber]]*1e6, self.t[endpoints[eventnumber]]*1e6],[self.data[endpoints[eventnumber]], self.data[endpoints[eventnumber]]],pen=None, symbol='o',symbolBrush='r',symbolSize=5)


        
    def previousevent(self):      
        self.ui.eventplot.plotItem.clear()
        
        eventnumber=np.int(self.ui.eventnumberentry.text())-1   
        self.ui.eventplot.plot(self.t[startpoints[eventnumber]-1000:endpoints[eventnumber]+1000]*1e6,self.data[startpoints[eventnumber]-1000:endpoints[eventnumber]+1000],pen='b')
        self.ui.eventplot.plotItem.addLine(y=self.baseline-self.deli[eventnumber],pen=(173,27,183))
        self.ui.eventnumberentry.setText(str(eventnumber)  )
        self.ui.scatterplot.plot(self.dwell,self.deli,pen=None, symbol='o',symbolBrush='b',symbolSize=10)
        self.ui.scatterplot.plot([self.dwell[eventnumber],self.dwell[eventnumber]],[self.deli[eventnumber],self.deli[eventnumber]],pen=None, symbol='o',symbolBrush='r',symbolSize=10)
        self.ui.eventplot.plot([self.t[startpoints[eventnumber]]*1e6, self.t[startpoints[eventnumber]]*1e6],[self.data[startpoints[eventnumber]], self.data[startpoints[eventnumber]]],pen=None, symbol='o',symbolBrush='g',symbolSize=5)
        self.ui.eventplot.plot([self.t[endpoints[eventnumber]]*1e6, self.t[endpoints[eventnumber]]*1e6],[self.data[endpoints[eventnumber]], self.data[endpoints[eventnumber]]],pen=None, symbol='o',symbolBrush='r',symbolSize=5)

        
    def cut(self):              
        if 'lr' in globals():      
                cutregion=lr.getRegion()
                print cutregion[0], cutregion[1]
                self.p1.clear()
                self.data=np.delete(self.data,np.arange(np.int(cutregion[0]*self.outputsamplerate),np.int(cutregion[1]*self.outputsamplerate)))
                
                self.t=np.arange(0,len(self.data))
                self.t=self.t/self.outputsamplerate
                
                self.baseline=np.median(self.data)  
                self.var=std(self.data)
                self.p1.plot(self.t[::10],self.data[::10],pen='b')
                self.p1.addLine(y=self.baseline,pen='g')
                self.p1.addLine(y=self.threshold,pen='r')
                del lr
        
        else:
            global lr
            self.p1.clear()
            lr = pg.LinearRegionItem()
            lr.hide()
            self.p1.addItem(lr)     
            
            self.p1.plot(self.t[::100],self.data[::100],pen='b')
            lr.show()
            
    def baselinecalc(self):
        if 'lr' in globals():      
                calcregion=lr.getRegion()
                print calcregion[0], calcregion[1]
                self.p1.clear()
                
                self.baseline=np.median(self.data[np.arange(np.int(calcregion[0]*self.outputsamplerate),np.int(calcregion[1]*self.outputsamplerate))])  
                self.var=std(self.data[np.arange(np.int(calcregion[0]*self.outputsamplerate),np.int(calcregion[1]*self.outputsamplerate))])
                self.p1.plot(self.t[::10],self.data[::10],pen='b')
                self.p1.addLine(y=self.baseline,pen='g')
                self.p1.addLine(y=self.threshold,pen='r')
                del lr
        
        else:
            global lr
            self.p1.clear()
            lr = pg.LinearRegionItem()
            lr.hide()
            self.p1.addItem(lr)     
            
            self.p1.plot(self.t[::100],self.data[::100],pen='b')
            lr.show()
            
    def clearscatter(self):
        self.ui.scatterplot.clear()
        
    def deleteevent(self):
        global startpoints,endpoints
        eventnumber=np.int(self.ui.eventnumberentry.text())
        self.deli=np.delete(self.deli,eventnumber)
        self.dwell=np.delete(self.dwell,eventnumber)
        self.dt=np.delete(self.dt,eventnumber)
        startpoints=np.delete(startpoints,eventnumber)
        endpoints=np.delete(endpoints,eventnumber)
        numberofevents=len(self.dt)
        self.ui.eventcounterlabel.setText('Events:'+str(numberofevents))
 

        self.ui.eventplot.plotItem.clear()
        self.ui.scatterplot.plotItem.clear()
        
        self.ui.eventplot.plot(self.t[startpoints[eventnumber]-1000:endpoints[eventnumber]+1000]*1e6,self.data[startpoints[eventnumber]-1000:endpoints[eventnumber]+1000],pen='b')
        self.ui.eventplot.plotItem.addLine(y=self.baseline-self.deli[eventnumber],pen=(173,27,183))
        self.ui.scatterplot.plot(self.dwell,self.deli,pen=None, symbol='o',symbolBrush='b',symbolSize=10)
        self.ui.scatterplot.plot([self.dwell[eventnumber],self.dwell[eventnumber]],[self.deli[eventnumber],self.deli[eventnumber]],pen=None, symbol='o',symbolBrush='r',symbolSize=10)
        self.ui.eventplot.plot([self.t[startpoints[eventnumber]]*1e6, self.t[startpoints[eventnumber]]*1e6],[self.data[startpoints[eventnumber]], self.data[startpoints[eventnumber]]],pen=None, symbol='o',symbolBrush='g',symbolSize=5)
        self.ui.eventplot.plot([self.t[endpoints[eventnumber]]*1e6, self.t[endpoints[eventnumber]]*1e6],[self.data[endpoints[eventnumber]], self.data[endpoints[eventnumber]]],pen=None, symbol='o',symbolBrush='r',symbolSize=5)
        
        
            
        

 
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    myapp = GUIForm()
    myapp.show()
    sys.exit(app.exec_())
    
    