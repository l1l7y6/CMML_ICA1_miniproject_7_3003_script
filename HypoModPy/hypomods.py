
import wx
from HypoModPy.hypoparams import *
from HypoModPy.hypodat import *
from threading import Thread
from datetime import datetime
import wx.lib.newevent


# Custom Thread Event
ModThreadCompleteEvent = wx.NewEventType()
EVT_MODTHREAD_COMPLETE = wx.PyEventBinder(ModThreadCompleteEvent, 0)

ModThreadProgressEvent = wx.NewEventType()
EVT_MODTHREAD_PROGRESS = wx.PyEventBinder(ModThreadProgressEvent, 0)

class ModThreadEvent(wx.PyCommandEvent):
    def __init__(self, evtType):
        wx.PyCommandEvent.__init__(self, evtType)


# Mod Class
class Mod(wx.EvtHandler):
    def __init__(self, mainwin, tag):
        wx.EvtHandler.__init__(self)

        self.mainwin = mainwin
        self.tag = tag
        self.type = type
        self.graphload = False
        self.runflag = False

        self.modtools = {}
        self.modbox = None

        self.plotbase = PlotBase(mainwin)
        self.settags = []

        self.Bind(EVT_MODTHREAD_COMPLETE, self.OnModThreadComplete)
        self.Bind(EVT_MODTHREAD_PROGRESS, self.OnModThreadProgress)


    def IoDGraph(self, IoDdata, IoDdataX, label, tag, colour, barshift):
        self.plotbase.AddPlot(PlotDat(IoDdata, 0, 70, 0, 2, label, "barX", 1, colour), tag)
        self.plotbase.GetPlot(tag).xdata = IoDdataX
        self.plotbase.GetPlot(tag).xcount = 7
        self.plotbase.GetPlot(tag).synchx = False
        self.plotbase.GetPlot(tag).barshift  = barshift


    def OnModThreadComplete(self, event):
        #runmute->Lock();
        #runflag = 0;
        #runmute->Unlock();
        self.mainwin.scalebox.GraphUpdateAll()
        #DiagWrite("Model thread OK\n\n")


    def GetPath(self):
        if self.mainwin.modpath == "": 
            if self.path != "": fullpath = self.path
            else: fullpath = self.mainwin.initpath
        else:
            if self.path != "": fullpath = self.mainwin.modpath + "/" + self.path
            else: fullpath = self.mainwin.modpath

        print("path " + self.path)
        print("fullpath " + fullpath)

        if os.path.exists(fullpath) == False: 
            os.mkdir(fullpath)

        return fullpath
    

    def AddTool(self, toolbox):
        self.modtools[toolbox.tag] = toolbox
        self.mainwin.toolset.AddBox(toolbox)  
        print("AddTool", toolbox.tag, toolbox.boxtag)
      

    def ModStore(self):
        filepath = self.path
        
        # box store
        filename = self.tag + "-box.ini"
        outfile = TextFile(filepath + "/" + filename)
        outfile.Open('w')

        for box in self.modtools.values():
            outfile.WriteLine("{} {} {} {} {} {}".format(box.tag, box.mpos.x, box.mpos.y, box.size.x, box.size.y, box.IsShown()))
            if box.storetag is not None: box.storetag.HistStore()

        outfile.Close()
        print("ModStore OK")

 
    def ModLoad(self):
        filepath = self.path

        # box load
        filename = self.tag + "-box.ini"
        infile = TextFile(filepath + "/" + filename)
        check = infile.Open('r')
        if check == False: 
            print("ModLoad box file not found")
            return
        filetext = infile.ReadLines()

        for line in filetext:
            linedata = line.split(' ')
            boxtag = linedata[0]
            if boxtag in self.modtools.keys():      
                pos = wx.Point(int(linedata[1]), int(linedata[2]))
                size = wx.Size(int(linedata[3]), int(linedata[4]))
                if linedata[5] == 'True\n': visible = True
                else: visible = False
                self.modtools[boxtag].visible = visible
                self.modtools[boxtag].mpos = pos
                self.modtools[boxtag].size = size

        infile.Close()

        for box in self.modtools.values():
            box.SetSize(box.size)
            box.SetPosition(self.mainwin.GetPosition(), self.mainwin.GetSize())
            box.Show(box.visible)

        print("ModLoad OK")


    def GridColumn(self, col):
        return 0


    def GridRow(self, row):
        return 0



class ModThread(Thread):
    def __init__(self, params, mainwin):
        Thread.__init__(self)

        #self.modbox = box
        self.mainwin = mainwin
        self.diag = False

    

