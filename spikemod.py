

import wx
import random
import numpy as np

from HypoModPy.hypomods import *
from HypoModPy.hypoparams import *
from HypoModPy.hypodat import *
from HypoModPy.hypogrid import *
from HypoModPy.hypospikes import *

from spikepanels import SpikeBox, SecBox



class SecData():
    def __init__(self, size):
        self.size = size
        self.secP = pdata(size)   # releasable pool
        self.secR = pdata(size)   # reserve pool
        self.secX = pdata(size)   # secretion rate
        self.secPlasma = pdata(size)   # plasma concentration
        self.secE = pdata(size)   # fast Ca2+
        self.secC = pdata(size)   # slow Ca2+
        self.secB = pdata(size)   # spike broadening


class StateData():
    def __init__(self, size):
        self.size = size
        self.memV = pdata(size)   # membrane potential
        self.nmdaX = pdata(size)  # fast NMDA trigger state
        self.nmdaS = pdata(size)  # slow NMDA synapse state
        self.nmdaI = pdata(size)  # voltage-dependent NMDA depolarisation

    

class SpikeMod(Mod):
    def __init__(self, mainwin, tag):
        Mod.__init__(self, mainwin, tag)

        if mainwin.modpath != "": self.path = mainwin.modpath + "/Spike"
        else: self.path = "Spike"

        if os.path.exists(self.path) == False: 
            os.mkdir(self.path)

        self.mainwin = mainwin
        self.datsample = 1       # ms sample interval for secretion model data, 1000 = 1 sample per second

        # tool boxes
        self.gridbox = GridBox(self, "Data Grid", wx.Point(0, 0), wx.Size(320, 500), 100, 20)
        self.gridbox.NeuroButton()
        self.secbox = SecBox(self, "secmod", "Secretion Model", wx.Point(0, 0), wx.Size(320, 500))
        self.spikebox = SpikeBox(self, "spikemod", "Spike Model", wx.Point(0, 0), wx.Size(320, 500))
        self.spikedatabox = SpikeDataBox(self, "spikedatabox", "Spike Data", wx.Point(0, 0), wx.Size(320, 500))

        # link mod owned boxes
        mainwin.gridbox = self.gridbox
        mainwin.spikedatabox = self.spikedatabox

        # spike data analysis stores
        self.cellspike = SpikeDat()
        self.modspike = SpikeDat()

        # spike data stores
        self.spikedata = []

        # secretion data stores
        self.secdata = SecData(1000000)
        self.statedata = StateData(1000000)

        self.AddTool(self.spikebox)
        self.AddTool(self.secbox)
        self.AddTool(self.gridbox)
        self.AddTool(self.spikedatabox)
    
        self.spikebox.Show(True)
        self.modbox = self.spikebox

        self.ModLoad()
        print("Spike Model OK")

        self.PlotData()
        self.graphload = True


    ## PlotData() defines all the available plots, each linked to a data array in osmodata
    ##
    def PlotData(self):
        # Data plots
        #
        # AddPlot(PlotDat(data array, xfrom, xto, yfrom, yto, label string, plot type, bin size, colour), tag string)
        # ----------------------------------------------------------------------------------
        self.plotbase.AddPlot(PlotDat(self.cellspike.hist5, 0, 2000, 0, 500, "Cell Hist 5ms", "line", 1, "blue"), "datahist5")
        self.plotbase.AddPlot(PlotDat(self.cellspike.hist5norm, 0, 2000, 0, 500, "Cell Hist 5ms Norm", "line", 1, "blue"), "datahist5norm")
        self.plotbase.AddPlot(PlotDat(self.cellspike.haz5, 0, 2000, 0, 100, "Cell Haz 5ms", "line", 1, "blue"), "datahaz5")
        self.plotbase.AddPlot(PlotDat(self.modspike.hist5, 0, 2000, 0, 100, "Mod Hist 5ms", "line", 1, "green"), "modhist5")
        self.plotbase.AddPlot(PlotDat(self.modspike.hist5norm, 0, 2000, 0, 100, "Mod Hist 5ms Norm", "line", 1, "green"), "modhist5norm")
        self.plotbase.AddPlot(PlotDat(self.modspike.haz5, 0, 2000, 0, 100, "Mod Haz 5ms", "line", 1, "green"), "modhaz5")

        self.plotbase.AddPlot(PlotDat(self.cellspike.srate1s, 0, 500, 0, 20, "Cell Spike Rate 1s", "spikes", 1, "red"), "cellrate1s")
        self.plotbase.AddPlot(PlotDat(self.modspike.srate1s, 0, 500, 0, 20, "Mod Spike Rate 1s", "spikes", 1, "purple"), "modrate1s")

        self.IoDGraph(self.cellspike.IoDdata, self.cellspike.IoDdataX, "IoD Cell", "iodcell", "lightblue", 10)
        self.IoDGraph(self.modspike.IoDdata, self.modspike.IoDdataX, "IoD Mod", "iodmod", "lightgreen", 0)

        self.plotbase.AddPlot(PlotDat(self.secdata.secP, 0, 500, 0, 5000, "Secretion P", "line", 1, "blue", 1000 / self.datsample), "secP")
        self.plotbase.AddPlot(PlotDat(self.secdata.secR, 0, 500, 0, 20000, "Secretion R", "line", 1, "green", 1000 / self.datsample), "secR")
        self.plotbase.AddPlot(PlotDat(self.secdata.secX, 0, 500, 0, 30, "Secretion X", "line", 1, "lightred", 1000 / self.datsample), "secX")
        self.plotbase.AddPlot(PlotDat(self.secdata.secPlasma, 0, 500, 0, 30, "Plasma Conc", "line", 1, "purple", 1000 / self.datsample), "secPlasma")
        self.plotbase.AddPlot(PlotDat(self.secdata.secE, 0, 500, 0, 5, "Secretion E", "line", 1, "lightred", 1000 / self.datsample), "secE")
        self.plotbase.AddPlot(PlotDat(self.secdata.secC, 0, 500, 0, 5, "Secretion C", "line", 1, "lightred", 1000 / self.datsample), "secC")
        self.plotbase.AddPlot(PlotDat(self.secdata.secB, 0, 500, 0, 5, "Secretion B", "line", 1, "green", 1000 / self.datsample), "secB")
        self.plotbase.AddPlot(PlotDat(self.statedata.memV, 0, 500, -80, -30, "Membrane V", "line", 1, "blue", 1000 / self.datsample), "memV")
        self.plotbase.AddPlot(PlotDat(self.statedata.nmdaX, 0, 500, 0, 2, "NMDA X", "line", 1, "yellow", 1000 / self.datsample), "nmdaX")
        self.plotbase.AddPlot(PlotDat(self.statedata.nmdaS, 0, 500, 0, 30, "NMDA S", "line", 1, "green", 1000 / self.datsample), "nmdaS")
        self.plotbase.AddPlot(PlotDat(self.statedata.nmdaI, 0, 500, 0, 6, "NMDA I", "line", 1, "purple", 1000 / self.datsample), "nmdaI")


    def DefaultPlots(self):
        if len(self.mainwin.panelset) > 0: self.mainwin.panelset[0].settag = "datahist5"
        if len(self.mainwin.panelset) > 1: self.mainwin.panelset[1].settag = "datahaz5"
        if len(self.mainwin.panelset) > 2: self.mainwin.panelset[2].settag = "modhist5"


    def NeuroData(self):
        DiagWrite("NeuroData() call\n")

        self.cellindex = self.spikedatabox.cellpanel.cellindex
        self.cellspike.Analysis(self.spikedata[self.cellindex])
        self.cellspike.id = self.cellindex
        self.cellspike.name = self.spikedata[self.cellindex].name
        self.spikedatabox.cellpanel.PanelData(self.cellspike)

        self.mainwin.scalebox.GraphUpdateAll()


    def ModelData(self):
        DiagWrite("ModelData() call\n")

        self.modspike.Analysis()
        self.spikebox.SpikeData(self.modspike)
        self.mainwin.scalebox.GraphUpdateAll()


    def OnModThreadComplete(self, event):
        self.mainwin.scalebox.GraphUpdateAll()
        DiagWrite(f"Model thread OK, test value {event.GetInt()}\n\n")
        self.runflag = False
        self.ModelData()


    def OnModThreadProgress(self, event):
        self.spikebox.SetCount(event.GetInt())
        #DiagWrite(f"Model thread progress, value {event.GetInt()}\n\n")


    def RunModel(self):
        if not self.runflag:
            self.mainwin.SetStatusText("Spike Model Run")
            self.runflag = True
            params = {
                "spike": self.modbox.GetParams(),
                "sec": self.secbox.GetParams()
            }

            # multibox example
            #
            # params = {
            #     "spike": self.spikebox.GetParams(),
            #     "neuro": self.neurobox.GetParams(),
            #     "syn": self.synbox.GetParams(),
            # }
            #

            modthread = SpikeModel(self, params)
            modthread.start()
           


class SpikeModel(ModThread):
    def __init__(self, mod, params):
        ModThread.__init__(self, params, mod.mainwin)

        self.params = params
        self.mod = mod
        self.spikebox = mod.spikebox
        self.mainwin = mod.mainwin
        self.scalebox = mod.mainwin.scalebox


    # run() is the thread entry function, used to initialise and call the main Model() function   
    def run(self):
        # Read model flags
        self.randomflag = self.spikebox.modflags["randomflag"]      # model flags are useful for switching elements of the model code while running

        if self.randomflag: random.seed(0)
        else: random.seed(datetime.now().microsecond)

        DiagWrite("Running Spike Model\n")

        self.Model()
        completeevent = ModThreadEvent(ModThreadCompleteEvent)
        wx.QueueEvent(self.mod, completeevent)


    # Model() reads in the model parameters, initialises variables, and runs the main model loop
    def Model(self):

        # Data stores
        datsample = self.mod.datsample
        secsize = self.mod.secdata.size
        spikedata = self.mod.modspike
        secdata = self.mod.secdata
        statedata = self.mod.statedata

        # Parameter stores
        spikeparams = self.params["spike"]
        secparams = self.params["sec"]

        # Read parameters
        runtime = int(spikeparams["runtime"])
        hstep = spikeparams["hstep"]
        Vthresh = spikeparams["Vthresh"]
        Vrest = spikeparams["Vrest"]
        pspmag = spikeparams["pspmag"]
        psprate = spikeparams["psprate"]
        pspratio = spikeparams["pspratio"]
        halflifeMem = spikeparams["halflifeMem"]
        kHAP = spikeparams["kHAP"]
        halflifeHAP = spikeparams["halflifeHAP"]
        kAHP = spikeparams["kAHP"]
        halflifeAHP = spikeparams["halflifeAHP"]
        kDAP = spikeparams['kDAP']
        halflifeDAP = spikeparams['halflifeDAP']

        # NMDA-based DAP parameters (temporary hard-coded defaults with script overrides)
        useNMDA = spikeparams.get("useNMDA", 1)
        kNMDA = spikeparams.get("kNMDA", 0.8)
        halflifeNMDARise = spikeparams.get("halflifeNMDARise", 8.0)
        halflifeNMDADecay = spikeparams.get("halflifeNMDADecay", 120.0)

        halflifeB = secparams["halflifeB"]
        halflifeE = secparams["halflifeE"]
        halflifeC = secparams["halflifeC"]
        Ethresh = secparams["Eth"]
        Cthresh = secparams["Cth"]
        Bbase = secparams["Bbase"]
        alpha = secparams["alpha"]
        Pmax = secparams["Pmax"]
        Rinit = secparams["Rinit"]
        secExp = secparams["secExp"]
        beta = secparams["beta"]
        Rmax = secparams["Rmax"]
        kB = secparams["kB"]
        kE = secparams["kE"]
        kC = secparams["kC"]
        VolPlasma = secparams["VolPlasma"]

        epspmag = pspmag
        ipspmag = pspmag
        absref = 2
        

        # Time Constants - conversion from half-life

        # Spiking 
        tauMem = math.log(2) / halflifeMem
        tauHAP = math.log(2) / halflifeHAP
        tauAHP = math.log(2) / halflifeAHP

        

        if kDAP > 0 and halflifeDAP>0:
            tauDAP = math.log(2) / halflifeDAP
        else:
            tauDAP = 0.0
            kDAP = 0.0

        if useNMDA and kNMDA > 0 and halflifeNMDARise > 0 and halflifeNMDADecay > 0:
            tauNMDARise = math.log(2) / halflifeNMDARise
            tauNMDADecay = math.log(2) / halflifeNMDADecay
        else:
            tauNMDARise = 0.0
            tauNMDADecay = 0.0
            kNMDA = 0.0

        # Secretion
        tauB = math.log(2) / halflifeB
        tauE = math.log(2) / halflifeE
        tauC = math.log(2) / halflifeC
        tauDiff = math.log(2) / secparams["halflifeDiff"]
        tauClear = math.log(2) / secparams["halflifeClear"]

        alpha = alpha / 1000   # convert from per second to per ms, as model runs in ms time steps
        tauClear = tauClear / 1000   # convert from per second to per ms, as model runs in ms time steps
        tauDiff = tauDiff / 1000   # convert from per second to per ms, as model runs in ms time steps

        # Initialise variables
        epsprate = 0
        ipsprate = 0
        epspt = 0
        ipspt = 0
        ttime = 0
        V = Vrest
        inputPSP = 0
        tPSP = 0
        tHAP = 0
        tAHP = 0

        tDAP = 0
        xNMDA = 0.0
        sNMDA = 0.0
        iNMDA = 0.0
        Bnmda = 0.0

        tB = 0  # Broadening
        tE = 0  # Fast Ca2+
        tC = 0.03   # Slow Ca2+
        tR = Rinit  # Reserve Pool
        tP = Pmax   # Releasable Pool 
        tPlasma = 0 # Hormone plasma concentration
        tEVF = 0

        Ethpow = Ethresh * Ethresh * Ethresh * Ethresh * Ethresh  # precalculate instead of each loop
        Cthpow = Cthresh * Cthresh * Cthresh

        spikedata.spikecount = 0
        maxspikes = spikedata.maxspikes
        neurotime = 0
        runtime = runtime * 1000

        # Run model loop
        for i in range(1, runtime + 1):
            ttime += 1
            neurotime += 1
            if i%(runtime/100) == 0: 
                progevent = ModThreadEvent(ModThreadProgressEvent)
                progevent.SetInt(math.floor(neurotime / runtime * 100)) 
                wx.QueueEvent(self.mod, progevent)                        # Update run progress % in model panel

            # PSP input signal
            nepsp = 0
            nipsp = 0
            epsprate = psprate / 1000
            ipsprate = epsprate * pspratio

            if epsprate > 0: 
                while epspt < hstep:
                    erand = random.random()
                    nepsp += 1
                    epspt = -math.log(1 - erand) / epsprate + epspt
                epspt = epspt - hstep

            if ipsprate > 0: 
                while ipspt < hstep:
                    irand = random.random()
                    nipsp += 1
                    ipspt = -math.log(1 - irand) / ipsprate + ipspt
                ipspt = ipspt - hstep

            inputPSP = nepsp * epspmag - nipsp * ipspmag


            # Spiking Model
            tPSP = tPSP + inputPSP - tPSP * tauMem
            tHAP = tHAP - tHAP * tauHAP
            tAHP = tAHP - tAHP * tauAHP
            tDAP = tDAP - tDAP * tauDAP

            # NMDA synapse-based DAP
            xNMDA = xNMDA - xNMDA * tauNMDARise
            sNMDA = sNMDA + xNMDA - sNMDA * tauNMDADecay

            # Use the non-NMDA membrane potential as the voltage-gating input.
            Vbase = Vrest + tPSP + tDAP - tHAP - tAHP
            Bnmda = 1.0 / (1.0 + math.exp(-(Vbase + 40.0) / 6.0))
            iNMDA = sNMDA * Bnmda

            V = Vbase + iNMDA

            #print(f"SpikeModel step {i}  V {V:.2f}  tPSP {tPSP:.2f}  inputPSP {inputPSP:.2f}  nepsp {nepsp}")


            # Secretion Model
            tB = tB - (tB * tauB)    # broadening
            tE = tE - (tE * tauE)    # fast Ca2+
            tC = tC - (tC * tauC)    # slow Ca2+

            EKpow = tE * tE * tE * tE * tE
            Einh = 1 - EKpow / (EKpow + Ethpow)
            CKpow = tC * tC * tC
            Cinh = 1 - CKpow / (CKpow + Cthpow) 
            CaEnt = Einh * Cinh * (tB + Bbase)   # Ca Entry
            if secExp == 3: tEpow = tE * tE * tE            # vaso
            if secExp == 2: tEpow = tE * tE                 # oxy
            secX = tEpow * alpha * tP       # secretion rate, proportional to releasable pool and fast Ca2+

            # Reserve Store (tR) and Releasable Pool (tP)
            if tP < Pmax: 
                fillP = beta * tR / Rmax
            else:
                fillP = 0

            tP = tP - secX + fillP     # Releasable pool is decremented by secretion, incremented by filling from reserve store
            fillR = 0   # no synthesis in this version
            tR = tR + fillR - fillP    # Reserve store is decremented by filling releasable pool, incremented by synthesis

            # Plasma Model
            tPlasma = tPlasma + secX - tPlasma * tauClear
            #tEVF = tEVF + tPlasma * tauDiff - tEVF


            if i % datsample == 0 and i < secsize * datsample:
                secdata.secP[int(i/datsample)] = tP
                secdata.secR[int(i/datsample)] = tR
                secdata.secX[int(i/datsample)] = secX
                secdata.secPlasma[int(i/datsample)] = tPlasma / VolPlasma   # convert to concentration 
                secdata.secE[int(i/datsample)] = tE
                secdata.secC[int(i/datsample)] = tC
                secdata.secB[int(i/datsample)] = tB
                statedata.memV[int(i/datsample)] = V
                statedata.nmdaX[int(i/datsample)] = xNMDA
                statedata.nmdaS[int(i/datsample)] = sNMDA
                statedata.nmdaI[int(i/datsample)] = iNMDA


            # Spiking
            if V > Vthresh and ttime >= absref:

                # record spike time
                if spikedata.spikecount < maxspikes: 
                    spikedata.times[spikedata.spikecount] = neurotime
                    spikedata.spikecount += 1
    
                # Spike incremented variable
                # afterpotentials
                tHAP = tHAP + kHAP
                tAHP = tAHP + kAHP

                tDAP = tDAP + kDAP
                xNMDA = xNMDA + kNMDA

                tB = tB + kB
                tE = tE + kE * CaEnt
                tC = tC + kC * CaEnt	
                
                ttime = 0   # reset time since last spike

            

        # Finalise data
        freq = spikedata.spikecount / (runtime / 1000)
        DiagWrite(f"Spike Model OK, generated {spikedata.spikecount} spikes, freq {freq:.2f}\n")



    
