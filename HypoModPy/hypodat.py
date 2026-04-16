
from HypoModPy.hypobase import *
from HypoModPy.hypotools import DiagWrite
import numpy as np


class datarray():
    def __init__(self, size, dtype=float):
        self.data = np.zeros(size, dtype=dtype)
        self.xmax = size
        self.empty = True


    def __getitem__(self, key):      
        return self.data.__getitem__(key)


    def __setitem__(self, key, value):
        self.data.__setitem__(key, value)


    def __getattr__(self, name):
        """Delegate to NumPy array."""
        try:
            return getattr(self.data, name)
        except AttributeError:
            raise AttributeError(
                 "'Array' object has no attribute {}".format(name))
        
    
    def clear(self):
        self.data.fill(0)
        self.empty = False

# subclass code from 
# https://numpy.org/doc/stable/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array

class pdata(np.ndarray):
    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None, info=None):
        obj = super().__new__(subtype, shape, dtype,
                              buffer, offset, strides, order)
        # set the new 'info' attribute to the value passed
        obj.info = info
        obj.xmax = len(obj)
        obj.empty = True
        # Finally, we must return the newly created object:
        return obj
    
    
    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'info', None)
        self.xmax = getattr(obj, 'xmax', len(self))
        self.empty = getattr(obj, 'empty', True)


    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.empty = False


    def reset(self):
        self.fill(0)
        self.empty = True

    

class PlotSet():
    def __init__(self):
        self.plottags = []
        self.plotcodes = {}
        self.label = ""
        self.tag = ""
        self.modeflags = []           # Set of flags is used to control the selected, displayed graph 
        self.modeweights = []
        self.single = True
        self.submenu = 0
        self.modesum = 0
        #self.subplot = []


    def AddPlot(self, plottag, plotcode = -1): 
        self.plottags.append(plottag)
        self.plotcodes[plottag] = plotcode
        if len(self.plottags) > 1: self.single = False


    def AddFlag(self, flag, weight):
        self.modeflags.append(flag)
        self.modeweights.append(weight)


    def GetPlot(self, gflags, subplot = None):
        if self.single: return self.plottags[0]

        if self.submenu: 
            if subplot: return self.plottags[subplot]    
            else: return self.plottags[0]

        self.modesum = 0
        plottag = self.plottags[0]
        for modeflag in self.modeflags: self.modesum = self.modesum + gflags[modeflag] * self.modeweights[modeflag]
        for tag in self.plottags:
            #if self.plotcodes[tag] == self.modesum: plottag = self.plottags[tag]
            if self.plotcodes[tag] == self.modesum: plottag = tag    # ChatGPT bug pick up and suggested fix 4/3/26

        return plottag
        

class PlotDat():
    def __init__(self, data = pdata(0), xf = 0, xt = 500, yf = 0, yt = 1000, label = "", type = None, binsize = 1, colour = "red", xs = 1, xd = 0):

        self.xscale = xs
        self.xdis = xd
        self.spikedata = None

        self.xdata = None
        self.xcount = 0
        self.ycount = 0
        self.data = data

        self.type = type
        self.samprate = 1
        self.scattermode = 0
        self.linemode = 1
        self.scattersize = 2

        self.xfrom = xf
        self.xto = xt
        self.xmin = xf
        self.xmax = xt
        self.yfrom = yf
        self.yto = yt
        self.label = label
        self.colour = colour
        self.binsize = binsize

        self.oversync = True
        self.barwidth = 10
        self.bargap = 10

        self.Default()


    def SyncAxes(self, source):
        self.xfrom = source.xfrom
        self.xto = source.xto
        self.yfrom = source.yfrom
        self.yto = source.yto


    def Default(self):

        self.xtitle = "X"
        self.ytitle = "Y"

        self.xaxis = 1
        self.yaxis = 1

        self.yscale = 1

        self.xlabels = 0
        self.ylabels = 0
        self.xstep = 0
        self.ystep = 0

        self.xtickmode = 1
        self.ytickmode = 1

        self.xshift = 0
        self.yshift  = 0

        self.xunitscale = 1
        self.yunitscale = 1
        self.xunitdscale = 1
        self.yunitdscale = 1
        self.xsample = 1

        self.xlabelmode = 1
        self.ylabelmode = 1
        self.xticklength = 5
        self.yticklength = 5
        self.xlabelplaces = -1
        self.ylabelplaces = -1
        self.xscalemode = 0 
        self.yscalemode = 0

        self.xticklength = 5
        self.yticklength = 5

        self.scrollpos = 0
        self.xrel = 0

        self.negscale = 0   # check purpose
        self.synchx = 1     # toggle to allow x-axis synchronisation, typically used for common time axis

        self.plotstroke = 0.5
        self.strokecolour = wx.Colour(0, 0, 0)
        self.fillcolour = wx.Colour(255, 255, 255)

        self.xplot = 500
        self.yplot = 200
        self.xsample = 1

        self.xlabelgap = 30  #40
        self.ylabelgap = 30  #20 #30
        self.labelfontsize = 10
        self.tickfontsize = 10
        self.clipmode = 0

        self.barshift = 0
        self.barwidth = 10
        self.bargap = 10

        self.labelfont = 0   #default Helvetica
        self.fillmode = 1
        self.fillstroke = 0


    def GetData(self, xval): 
        data = -1
        xindex = xval / self.binsize
        data = self.data[xindex]
        return data


    def StoreDat(self, tag):
        if self.label != "": storetitle = self.label         # replace spaces with underscores for textfile storing
        else: storetitle = " "
        storetitle.replace(" ", "_")

        if self.xtitle != "": storextitle = self.xtitle
        else: storextitle = " "
        storextitle.replace(" ", "_")

        if self.ytitle != "": storeytitle = self.ytitle
        else: storeytitle = " "
        storeytitle.replace(" ", "_")
    
        strokecolourtext = self.strokecolour.GetAsString(wx.C2S_CSS_SYNTAX)
        fillcolourtext = self.fillcolour.GetAsString(wx.C2S_CSS_SYNTAX)
        #DiagWrite("strokecolourtext: " + strokecolourtext + "\n")

        gtext = "v1"
        gtext += f" tag {tag} xf {self.xfrom} xt {self.xto} yf {self.yfrom} yt {self.yto} xl {self.xlabels:d} xs {self.xstep} xm {self.xtickmode:d}"
        gtext += f" yl {self.ylabels:d} ys {self.ystep} ym {self.ytickmode:d} c {self.colour} srgb {strokecolourtext} xs {self.xshift} xu {self.xunitscale}"
        gtext += f" ps {self.plotstroke} name {storetitle} xtag {storextitle} ytag {storeytitle} xp {self.xplot} yp {self.yplot} pf {self.labelfontsize}"
        gtext += f" cm {self.clipmode} type {self.type} xd {self.xunitdscale} xsam {self.xsample} bw {self.barwidth} bg {self.bargap} yu {self.yunitscale}" 
        gtext += f" xl {self.xlabelplaces} yl {self.ylabelplaces} xm {self.xlabelmode} ym {self.ylabelmode} xs {self.xscalemode} ys {self.yscalemode}"
        gtext += f" xa {self.xaxis} ya {self.yaxis} yd {self.yunitdscale} xg {self.xlabelgap} yg {self.ylabelgap} lf {self.labelfont} sc {self.scattersize}"
        gtext += f" frgb {fillcolourtext} xfm {self.fillmode} fs {self.fillstroke} lm {self.linemode} sm {self.scattermode}"
        return gtext


    def LoadDat(self, readline, version):   
        self.xfrom, readline = ParseFloat(readline, 'f')
        self.xto, readline = ParseFloat(readline, 't')
        self.yfrom, readline = ParseFloat(readline, 'f')
        self.yto, readline = ParseFloat(readline, 't')
        self.xlabels, readline = ParseInt(readline, 'l')
        self.xstep, readline = ParseFloat(readline, 's')
        self.xtickmode, readline = ParseInt(readline, 'm')
        self.ylabels, readline = ParseInt(readline, 'l')
        self.ystep, readline = ParseFloat(readline, 's')
        self.ytickmode, readline = ParseInt(readline, 'm')

        self.colour, readline = ParseString(readline, 'c')
        colourstring, readline = ParseString(readline, 'b', 'x')
        self.strokecolour.Set(colourstring.strip())
        self.xshift, readline = ParseFloat(readline, 's')
        self.xunitscale, readline = ParseFloat(readline, 'u')
        self.plotstroke, readline = ParseFloat(readline, 's')
        self.gtitle, readline = ParseString(readline, 'e')
        self.gtitle.replace("_", " ")

        self.xtitle, readline = ParseString(readline, 'g')
        self.xtitle.replace("_", " ")
        if self.xtitle == " ": self.xtitle = ""

        self.ytitle, readline = ParseString(readline, 'g')
        self.ytitle.replace("_", " ")
        if self.ytitle == " ": self.ytitle = ""

        self.xplot, readline = ParseFloat(readline, 'p')
        self.yplot, readline = ParseFloat(readline, 'p')
        self.labelfontsize, readline = ParseFloat(readline, 'f')
        self.clipmode, readline = ParseInt(readline, 'm')
        self.type, readline = ParseString(readline, 'e')
        self.xunitdscale, readline = ParseFloat(readline, 'd')
        self.xsample, readline = ParseFloat(readline, 'm')
        self.barwidth, readline = ParseFloat(readline, 'w')
        self.bargap, readline = ParseFloat(readline, 'g')

        self.yunitscale, readline = ParseFloat(readline, 'u')
        self.xlabelplaces, readline = ParseInt(readline, 'l')
        self.ylabelplaces, readline = ParseInt(readline, 'l')
        self.xlabelmode, readline = ParseInt(readline, 'm')
        self.ylabelmode, readline = ParseInt(readline, 'm')
        self.xscalemode, readline = ParseInt(readline, 's')
        self.yscalemode, readline = ParseInt(readline, 's')
        self.xaxis, readline = ParseInt(readline, 'a')
        self.yaxis, readline = ParseInt(readline, 'a')
        self.yunitdscale, readline = ParseFloat(readline, 'd')
        self.xlabelgap, readline = ParseFloat(readline, 'g')
        self.ylabelgap, readline = ParseFloat(readline, 'g')

        self.labelfont, readline = ParseInt(readline, 'f')
        self.scattersize, readline = ParseFloat(readline, 'c')
        colourstring, readline = ParseString(readline, 'b', 'x')
        self.fillcolour.Set(colourstring.strip())
        self.fillmode, readline = ParseInt(readline, 'm')
        self.fillstroke, readline = ParseInt(readline, 's')
        self.linemode, readline = ParseInt(readline, 'm')
        self.scattermode, readline = ParseInt(readline, 'm')


class PlotBase():
    def __init__(self, mainwin):
        self.plotstore = {}
        self.setstore = {}
        self.mainwin = mainwin


    def BaseStore(self, filepath):
        outfile = TextFile(filepath)
        outfile.Open('w')

        for plot in self.plotstore:
            outfile.WriteLine(self.plotstore[plot].StoreDat(plot))

        outfile.Close()
        DiagWrite("BaseStore {} graphs\n".format(len(self.plotstore)))


    def BaseLoad(self, filepath):
        infile = TextFile(filepath)
        if not infile.Open('r'): 
            print("BaseLoad bad file")
            return
        pcount = 0

        # read file
        filetext = infile.ReadLines()
        for readline in filetext:
            # parse line
            readline = readline.strip()
            if readline == "": continue
            # version check
            if readline[0] == 'v': 
                version, readline = ParseInt(readline, 'v')    #  check gbase file version for backwards compatability

            else: version = 0
            #DiagWrite(f"Base file version {version}\n")
            #DiagWrite(f"Readline {readline}\n")

            plottag, readline = ParseString(readline, 'g')     # parse plot tag
            #DiagWrite(f"ptag {ptag}\n")
            if plottag in self.plotstore: 
                plot = self.plotstore[plottag]                     # access plot from store
                plot.LoadDat(readline, version)        # parse plot parameters
                pcount += 1                                     # count only for diagnostics

        infile.Close()
        DiagWrite(f"BaseLoad {pcount} graphs\n")


    def AddPlot(self, newplot, plottag, settag = ""):       # default settag = "", for no set use settag = "null"
        plotset = None
        diag = False

        if diag: DiagWrite("Plotbase Add {} to set {}, numgraphs {}\n".format(plottag, settag, len(self.plotstore)))
    
        # colour setting is done here since GraphDat doesn't have access to mainwin colour chart
        newplot.strokecolour = self.mainwin.colourpen[newplot.colour]
        newplot.fillcolour = newplot.strokecolour
        newplot.plottag = plottag
    
        #mainwin->diagbox->Write(text.Format("GraphBase Add colour index %d string %s\n", newgraph.colour, ColourString(newgraph.strokecolour, 1)));

        # If single graph, create new single graph set, otherwise add to set 'settag'
        # single plot sets use the same tag as the plot
        if settag is not None:
            if settag == "": plotset = self.NewSet(newplot.label, plottag)
            else: plotset = self.setstore[settag]

            if plotset:   # extra check, should only fail if 'settag' is invalid
                plotset.AddPlot(plottag)
                newplot.settag = settag

        if diag: DiagWrite("GraphSet Add OK\n")

        # Add the new graph to graphbase
        self.plotstore[plottag] = newplot
        
        if diag: DiagWrite("GraphBase Add OK\n")


    def NewSet(self, label, tag): 
        self.setstore[tag] = PlotSet()
        self.setstore[tag].label = label
        self.setstore[tag].tag = tag
        return self.setstore[tag]


    def GetSet(self, tag):
        if tag == "": return None
        return self.setstore[tag]


    def GetPlot(self, tag):
        return self.plotstore[tag]





