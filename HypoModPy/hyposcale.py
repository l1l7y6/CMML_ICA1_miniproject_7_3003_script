
import wx
from HypoModPy.hypotools import *



class OverDat():
    def __init__(self, overlayID, panel1, panel2):
        self.overlayID = overlayID
        self.panel1 = panel1
        self.panel2 = panel2
        self.toggle = 0
        self.numdisps = 0


class ScaleBox(ToolPanel):
    def __init__(self, parent, size, numdraw):
        ToolPanel.__init__(self, parent, wx.DefaultPosition, size)

        iconpath = parent.respath
        self.ostype = GetSystem()
        self.numdraw = numdraw   # number of graph panels
        #self.mainwin.panelset = parent.panelset
        self.gsynch = 0    # x-axis synchronisation toggle
        self.synchcon = 0  # index of graph panel with synch control
        self.gflags = {}
        self.redtag = ""    # store box overwrite warning tag
        self.mod = parent.mod
        self.mainwin = parent
        self.overset = {}

        # Default scale parameter limits
        self.xmin = -1000000
        self.xmax = 10000000  # 1000000, extended for VasoMod long runs
        self.ymin = -1000000
        self.ymax = 1000000

        self.SetFont(self.boxfont)
        if self.ostype == 'Mac': self.buttonheight = 20
        else: self.buttonheight = 23

        # Load Icons
        if self.ostype == 'Mac' or self.ostype == 'Windows':
            self.rightarrow = wx.Bitmap(iconpath + "/rightarrow12.png", wx.BITMAP_TYPE_PNG)
            self.leftarrow = wx.Bitmap(iconpath + "/leftarrow12.png", wx.BITMAP_TYPE_PNG)
            self.uparrow = wx.Bitmap(iconpath + "/uparrow12.png", wx.BITMAP_TYPE_PNG)
            self.downarrow = wx.Bitmap(iconpath + "/downarrow12.png", wx.BITMAP_TYPE_PNG)
        else:
            self.rightarrow = wx.Bitmap(iconpath + "/rightarrow.png", wx.BITMAP_TYPE_PNG)
            self.leftarrow = wx.Bitmap(iconpath + "/leftarrow.png", wx.BITMAP_TYPE_PNG)
            self.uparrow = wx.Bitmap(iconpath + "/uparrow.png", wx.BITMAP_TYPE_PNG)
            self.downarrow = wx.Bitmap(iconpath + "/downarrow.png", wx.BITMAP_TYPE_PNG)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.AddSpacer(5)
        self.vconbox = wx.BoxSizer(wx.VERTICAL)

        panelindex = 0

        for graphpanel in self.mainwin.panelset:
            self.AddGraphConsole(graphpanel)

            if panelindex == 2:
                DiagWrite("ScaleBox panel 2\n")
                hbox = wx.BoxSizer(wx.HORIZONTAL)
                self.ScaleButton(ID_Overlay, "Overlay", 48, hbox).Bind(wx.EVT_BUTTON, self.OnOverlay)
                graphpanel.consolebox.Add(hbox, 0, wx.ALIGN_CENTRE_HORIZONTAL|wx.ALIGN_CENTRE_VERTICAL|wx.ALL, 2)
                self.overset[ID_Overlay] = OverDat(ID_Overlay, 2, 3)

            if panelindex == 4:
                DiagWrite("ScaleBox panel 4\n")
                hbox = wx.BoxSizer(wx.HORIZONTAL)
                self.ScaleButton(ID_Overlay2, "Overlay", 48, hbox).Bind(wx.EVT_BUTTON, self.OnOverlay)
                graphpanel.consolebox.Add(hbox, 0, wx.ALIGN_CENTRE_HORIZONTAL|wx.ALIGN_CENTRE_VERTICAL|wx.ALL, 2)
                self.overset[ID_Overlay2] = OverDat(ID_Overlay, 4, 5)
        
            panelindex += 1
        
            #gsync[i] = NULL;
            
        vbox.Add(self.vconbox, 1)

        buttonbox = wx.BoxSizer(wx.HORIZONTAL)
        if self.ostype == 'Mac':
            self.ScaleButton(wx.ID_OK, "OK", 35, buttonbox).Bind(wx.EVT_BUTTON, self.OnOK)
            buttonbox.AddSpacer(2)
            self.ScaleButton(ID_Sync, "Sync", 35, buttonbox).Bind(wx.EVT_BUTTON, self.OnSync)
        else:
            self.ScaleButton(wx.ID_OK, "OK", 35, buttonbox).Bind(wx.EVT_BUTTON, self.OnOK)
            buttonbox.AddSpacer(2)
            syncbutton = self.ToggleButton(ID_Sync, "Sync", 35, buttonbox).Bind(wx.EVT_BUTTON, self.OnSync)

        vbox.Add(buttonbox, 0, wx.ALIGN_CENTRE_HORIZONTAL|wx.ALIGN_CENTRE_VERTICAL)
        vbox.AddSpacer(3)
        storebox = self.StoreBox()
        vbox.Add(storebox, 0, wx.ALIGN_CENTRE_HORIZONTAL|wx.ALIGN_CENTRE_VERTICAL|wx.ALL, 0)	

        self.SetSizer(vbox)

        pub.subscribe(self.Scroll_Listener, "scroll_listener")
        pub.subscribe(self.Scale_Listener, "scale_listener")

        self.Bind(wx.EVT_TEXT_ENTER, self.OnEnter)

        #Connect(ID_overlay, wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler(ScaleBox::OnOverlay));



    def OnOverlay(self, event):
        #DiagWrite("OnOverlay()\n")
        overlay = self.overset[event.GetId()]
        if overlay is None:
            DiagWrite("ScaleBox overlay ID not found\n")
            return
 
        # Graph panel indexes
        pan1 = self.mainwin.panelset[overlay.panel1]
        pan2 = self.mainwin.panelset[overlay.panel2]
        
        # Overlay, add panel1 plot(s) to panel2
        if not overlay.toggle:  
            overlay.numdisps = len(pan1.dispset)  # records how many disps were moved
            if overlay.numdisps == 0: return
            
            # Synchronise axis scales in panel2
            refplot = pan1.GetFrontPlot()
            if refplot.oversync:
                #DiagWrite("ScaleBox overlay synch\n")
                for disp in pan2.dispset:
                    plot = disp.GetFront()
                    plot.SyncAxes(refplot)

            # Move GraphDisps down
            for disp in pan1.dispset:
                pan2.dispset.append(disp)
                pan1.dispset.remove(disp)
        
        # Reverse overlay, return added plot(s) to panel1
        else:
            numdisps = len(pan2.dispset)
            if numdisps == 0: return
            # Move GraphDisps up
            for i in range(numdisps - overlay.numdisps, numdisps):
                disp = pan2.dispset[i]
                pan1.dispset.append(disp)
                pan2.dispset.remove(disp)
             
        # DiagWrite(f"Overlay pan1 {overlay.panel1} numdisps {overlay.numdisps}\n")
        # DiagWrite("pan1 ")
        # for disp in pan1.dispset: DiagWrite(disp.GetFront().label + " ")
        # DiagWrite("end\n")
        # DiagWrite("pan2 ")
        # for disp in pan2.dispset: DiagWrite(disp.GetFront().label + " ")
        # DiagWrite("end\n")

        overlay.toggle = 1 - overlay.toggle
        self.ScaleUpdate()


    def OnGStore(self, event):

        graphpath = self.mod.path + "/Graphs"
        if os.path.exists(graphpath) == False: 
            os.mkdir(graphpath)

        filetag = self.storetag.GetValue()

        # Graph data file
        filepath = graphpath + "/" + "graph-" + filetag + ".dat"

        # Graph file history
        tagpos = self.storetag.FindString(filetag)
        if tagpos != wx.NOT_FOUND: self.storetag.Delete(tagpos)
        self.storetag.Insert(filetag, 0)
        #print("tag inserted " + filetag)

         # Overwrite Warning
        graphfile = TextFile(filepath)
        if graphfile.Exists() and self.redtag != filetag: 
            self.storetag.SetForegroundColour(self.redpen)
            self.storetag.SetValue("")
            self.storetag.SetValue(filetag)
            self.redtag = filetag
            return

        # Clear Overwrite Warning
        self.redtag = ""
        self.storetag.SetForegroundColour(self.blackpen)
        self.storetag.SetValue("")
        self.storetag.SetValue(filetag)

        # Open File
        graphfile.Open('w')

        # Write Panel PlotSet Tags
        for graphpanel in self.mainwin.panelset:
            graphfile.WriteLine("{} {}".format(graphpanel.index, graphpanel.settag))

         # Write Graph Flag Values
        graphfile.WriteLine("")
        for tag in self.gflags:
            outline = "%.0f".format(self.gflags[tag])
            graphfile.WriteLine(tag + " " + outline)

        graphfile.Close()

        #if(mainwin->graphbox) mainwin->graphbox->SetParams();
        self.mod.plotbase.BaseStore(graphpath + "/" + "gbase-" + filetag + ".dat")


    def OnGLoad(self, event):
        self.GLoad()


    def GLoad(self, tag = ""):
        diag = False
        pbase = self.mod.plotbase

        graphpath = self.mod.path + "/Graphs"
        if tag == "": filetag = self.storetag.GetValue()
        else: filetag = tag
        filename = "graph-" + filetag + ".dat"

        # Graph data file
        filepath = graphpath + "/" + "graph-" + filetag + ".dat"
        graphfile = TextFile(filepath)
        if not graphfile.Exists():
            if self.storetag: self.storetag.SetValue("Not found")
            return

        # File history
        if filetag != "default": 
            tagpos = self.storetag.FindString(filetag)
            if tagpos != wx.NOT_FOUND: self.storetag.Delete(tagpos)
            self.storetag.Insert(filetag, 0)

        # Clear Overwrite Warning
        self.redtag = ""
        self.storetag.SetForegroundColour(self.blackpen)
        self.storetag.SetValue("")
        self.storetag.SetValue(filetag)

        # Load file
        if not graphfile.Open('r'): 
            if diag: DiagWrite("GLoad graphfile error\n")
            return
        filetext = graphfile.ReadLines()
        mode = "panel"

        for readline in filetext:

            # Parse line type
            readline = readline.strip()
            if readline == "":  # marks end of file or section
                if mode == "panel": mode = "flag"
                continue  
            readdata = readline.split(' ')

            # Read graph panel
            if mode == "panel":
                index = int(readdata[0])    # panel index
                if index >= len(self.mainwin.panelset): break
                if len(readdata) < 2: break
                tag = readdata[1]  # plot set tag
                if tag in pbase.plotstore or tag in pbase.setstore: 
                    self.mainwin.panelset[index].settag = tag
                    if diag: DiagWrite(f"GLoad panel {index} graph/set {tag}\n")
                elif diag: DiagWrite(f"GLoad graph/set {tag} not found\n")
                if len(readdata) > 2: 
                    subtag = readdata[2]
                    if subtag in pbase.plotstore:
                        if tag in pbase.setstore: pbase.GetSet(tag).subplot[index] = subtag      # ported code, likely needs work, odd use of index

            # Read graph flags
            if mode == "flag":
                tag = readdata[0]  # graph flag tag
                data = readdata[1]  # flag data
                if tag in self.gflags:
                    flagval = int(data)
                    self.gflags[tag] = flagval
                    if diag: DiagWrite(f"Graph flag tag {tag}, value {flagval}\n") 

        graphfile.Close()

        pbase.BaseLoad(graphpath + "/" + "gbase-" + filetag + ".dat")
        self.GraphSwitch(self.mod.plotbase)


    def OnEnter(self, event):
        self.synchcon = event.GetId()
        self.OnOK(event)


    def XSynch(self, pos = -1):
        if self.gsynch:
            plotzero = self.mainwin.panelset[self.synchcon].GetFrontPlot()
            if not plotzero: return
            if not plotzero.synchx: return
            for graphpanel in self.mainwin.panelset:
                # sync check box code goes here
                plot = graphpanel.GetFrontPlot()
                if not plot.synchx: continue
                if pos >= 0: plot.scrollpos = pos
                plot.xfrom = plotzero.xfrom
                plot.xto = plotzero.xto


    def OnSync(self, event):
        pub.sendMessage("diagbox", message="OnSync\n")
        self.gsynch = 1 - self.gsynch
        self.ScaleUpdate()


    def Scale_Listener(self):
        self.ScaleUpdate()


    def Scroll_Listener(self, index, pos):
        self.ScrollUpdate(index, pos)

        
    def ScaleUpdate(self):
        self.XSynch()
        for graphpanel in self.mainwin.panelset:
            self.PanelUpdate(graphpanel)
            self.GraphUpdate(graphpanel)
      

    def ScrollUpdate(self, index, pos = -1):
        self.synchcon = index
        self.XSynch(pos)
        if(pos == -1 or self.gsynch):
            for graphpanel in self.mainwin.panelset:
                self.PanelUpdate(graphpanel)
                self.GraphUpdate(graphpanel)
        else:
            self.PanelUpdate(self.mainwin.panelset[index])
            self.GraphUpdate(self.mainwin.panelset[index])


    def GraphUpdate(self, graphpanel):
        graphpanel.ScrollUpdate()
        graphpanel.Refresh()


    def GraphUpdateAll(self):
        for graphpanel in self.mainwin.panelset:
            graphpanel.ScrollUpdate()
            graphpanel.Refresh()

            
    # PanelUpdate() - update scale panel after changing plot scale parameters
    def PanelUpdate(self, graphpanel):
            if len(graphpanel.dispset) == 0: return
            plot = graphpanel.dispset[0].plots[0]
            if not plot: return

            graphpanel.yf.SetNumValue(plot.yfrom, abs(plot.yto - plot.yfrom))
            graphpanel.yt.SetNumValue(plot.yto, abs(plot.yto - plot.yfrom))
            graphpanel.xf.SetNumValue(plot.xfrom, abs(plot.xto - plot.xfrom))
            graphpanel.xt.SetNumValue(plot.xto, abs(plot.xto - plot.xfrom))

            # overlay sync
            for i in range(1, len(graphpanel.dispset)):
                overplot = graphpanel.dispset[i].plots[0]
                if overplot.oversync:
                    overplot.yfrom = plot.yfrom
                    overplot.yto = plot.yto
                    plot.xfrom = plot.xfrom
                    plot.xto = plot.xto


    def PanelUpdateAll(self):
        for graphpanel in self.mainwin.panelset:
            self.PanelUpdate(graphpanel)
        

    def OnOK(self, event):
        for graphpanel in self.mainwin.panelset:
            plot = graphpanel.GetFrontPlot()
            oldxfrom = plot.xfrom
            oldxto = plot.xto

            plot.yfrom = graphpanel.yf.GetNumValue()
            plot.yto = graphpanel.yt.GetNumValue()
            plot.xfrom = graphpanel.xf.GetNumValue()
            plot.xto = graphpanel.xt.GetNumValue()

            if plot.xfrom < plot.xmin or plot.xfrom > self.xmax:
                pub.sendMessage("status_listener", message="X From, value out of range, max 100000")
                snum = "ScaleBox X out of range, value {} xmin {}\n".format(plot.xfrom, plot.xmin)
                pub.sendMessage("diagbox", message=snum)
                plot.xfrom = oldxfrom
                graphpanel.xf.SetNumValue(oldxfrom, plot.xfrom)

            if plot.xto < self.xmin or plot.xto > self.xmax: 
                pub.sendMessage("status_listener", message="X To, value out of range, max 100000")
                plot.xto = oldxto
                # need panel set here?

        graphpanel.XYSynch()
        self.ScaleUpdate()


    def StoreBox(self):
        label = 'gtest1'

        filebox = wx.BoxSizer(wx.VERTICAL)
        buttons = wx.BoxSizer(wx.HORIZONTAL)
        
        self.storetag = TagBox(self, label, wx.Size(80, -1), "scalebox", self.mainwin.modpath)
        self.storetag.SetFont(self.confont)

        if self.ostype == 'Mac': buttonwidth = 36
        else: buttonwidth = 38
        self.AddButton(ID_Store, "Store", buttonwidth, buttons).Bind(wx.EVT_BUTTON, self.OnGStore)
        buttons.AddSpacer(2)
        self.AddButton(ID_Load, "Load", buttonwidth, buttons).Bind(wx.EVT_BUTTON, self.OnGLoad)
        
        
        filebox.Add(self.storetag, 0, wx.ALIGN_CENTRE_HORIZONTAL|wx.ALIGN_CENTRE_VERTICAL|wx.ALL, 2)
        filebox.Add(buttons, 0, wx.ALIGN_CENTRE_HORIZONTAL|wx.ALIGN_CENTRE_VERTICAL|wx.ALL, 2)
        return filebox
        

    def AddButton(self, id, label, width, box, pad = 1, height = 0):
        if height == 0: height = self.buttonheight
        button = ToolButton(self, id, label, wx.DefaultPosition, wx.Size(width, height))
        button.SetFont(self.confont)
        box.Add(button, 0, wx.ALIGN_CENTRE_HORIZONTAL|wx.ALIGN_CENTRE_VERTICAL|wx.TOP|wx.BOTTOM, pad)
        return button


    def ScaleButton(self, id, label, width, box):
        button = ToolButton(self, id, label, wx.DefaultPosition, wx.Size(width, self.buttonheight))
        button.SetFont(self.confont)
        box.Add(button, 0, wx.ALIGN_CENTRE_HORIZONTAL|wx.ALIGN_CENTRE_VERTICAL|wx.TOP|wx.BOTTOM, 1)
        return button

    
    def ToggleButton(self, id, label, width, box):
        button = wx.ToggleButton(self, id, label, wx.DefaultPosition, wx.Size(width, self.buttonheight))
        button.SetFont(self.confont)
        box.Add(button, 0, wx.ALIGN_CENTRE_HORIZONTAL|wx.ALIGN_CENTRE_VERTICAL|wx.TOP|wx.BOTTOM, 1)
        return button


    def AddGraphConsole(self, graphpanel):
        graphpanel.scalebox = self
        psetbox = wx.BoxSizer(wx.VERTICAL)
        graphpanel.consolebox = wx.BoxSizer(wx.VERTICAL)
        zoombox = wx.BoxSizer(wx.HORIZONTAL)

        graphpanel.yf = self.AddScaleParam("YF", 0, psetbox, graphpanel.index)
        graphpanel.yt = self.AddScaleParam("YT", 10, psetbox, graphpanel.index)
        graphpanel.xf = self.AddScaleParam("XF", 0, psetbox, graphpanel.index)
        graphpanel.xt = self.AddScaleParam("XT", 500, psetbox, graphpanel.index)

        graphpanel.yzoomin = wx.BitmapButton(self, 1000 + graphpanel.index, self.downarrow, wx.DefaultPosition, wx.Size(20, 20))
        graphpanel.yzoomout = wx.BitmapButton(self, 1010 + graphpanel.index, self.uparrow, wx.DefaultPosition, wx.Size(20, 20))
        graphpanel.xzoomin = wx.BitmapButton(self, 1100 + graphpanel.index, self.leftarrow, wx.DefaultPosition, wx.Size(20, 20))
        graphpanel.xzoomout = wx.BitmapButton(self, 1110 + graphpanel.index, self.rightarrow, wx.DefaultPosition, wx.Size(20, 20))

        graphpanel.yzoomin.Bind(wx.EVT_BUTTON, self.OnYZoomIn)
        graphpanel.yzoomout.Bind(wx.EVT_BUTTON, self.OnYZoomOut)
        graphpanel.xzoomin.Bind(wx.EVT_BUTTON, self.OnXZoomIn)
        graphpanel.xzoomout.Bind(wx.EVT_BUTTON, self.OnXZoomOut)

        zoombox.Add(graphpanel.yzoomin, 0, wx.ALIGN_CENTRE_HORIZONTAL|wx.ALIGN_CENTRE_VERTICAL|wx.ALL, 0)
        zoombox.Add(graphpanel.yzoomout, 0, wx.ALIGN_CENTRE_HORIZONTAL|wx.ALIGN_CENTRE_VERTICAL|wx.ALL, 0)
        zoombox.AddSpacer(2)
        zoombox.Add(graphpanel.xzoomin, 0, wx.ALIGN_CENTRE_HORIZONTAL|wx.ALIGN_CENTRE_VERTICAL|wx.ALL, 0)
        zoombox.Add(graphpanel.xzoomout, 0, wx.ALIGN_CENTRE_HORIZONTAL|wx.ALIGN_CENTRE_VERTICAL|wx.ALL, 0)

        psetbox.Add(zoombox, 0, wx.ALIGN_CENTRE_HORIZONTAL)
        graphpanel.consolebox.Add(psetbox, 0, wx.ALIGN_CENTRE_HORIZONTAL)
        self.vconbox.Add(graphpanel.consolebox, 1, wx.ALIGN_CENTRE_HORIZONTAL, 0)
        self.vconbox.Layout()


    def RemoveGraphConsole(self, graphpanel):
        graphpanel.consolebox.Clear(True)
        self.vconbox.Detach(graphpanel.consolebox)
        self.vconbox.Layout()


    def OnYZoomIn(self, event):
        graphpanel = self.mainwin.panelset[event.GetId() - 1000]
        if len(graphpanel.dispset) == 0: return
        plot = graphpanel.GetFrontPlot()
        diff = plot.yto - plot.yfrom

        if plot.negscale or plot.yfrom < 0:
            plot.yto = plot.yto - diff / 4
            plot.yfrom = plot.yfrom + diff / 4
        else:
            plot.yto = plot.yto - diff / 2

        graphpanel.XYSynch
        self.synchcon = graphpanel.index
        self.ScaleUpdate()
        
    
    def OnYZoomOut(self, event):
        graphpanel = self.mainwin.panelset[event.GetId() - 1010]
        if len(graphpanel.dispset) == 0: return
        plot = graphpanel.GetFrontPlot()
        diff = plot.yto - plot.yfrom
        if plot.negscale or plot.yfrom < 0:
            plot.yto = plot.yto  + diff / 2
            plot.yfrom = plot.yfrom - diff / 2
        else:
            plot.yto = plot.yto + diff
        graphpanel.XYSynch
        self.synchcon = graphpanel.index
        self.ScaleUpdate()


    def OnXZoomIn(self, event):
        graphpanel = self.mainwin.panelset[event.GetId() - 1100]
        if len(graphpanel.dispset) == 0: return
        plot = graphpanel.GetFrontPlot()
        diff = plot.xto - plot.xfrom
        plot.xto = plot.xto - diff / 2
        graphpanel.XYSynch
        self.synchcon = graphpanel.index
        self.ScaleUpdate()
        

    def OnXZoomOut(self, event):
        graphpanel = self.mainwin.panelset[event.GetId() - 1110]
        if len(graphpanel.dispset) == 0: return
        plot = graphpanel.GetFrontPlot()
        oldxto = plot.xto
        diff = plot.xto - plot.xfrom
        plot.xto = plot.xto + diff
        # if plot.xto < plot.xmin or plot.xto > plot.xmax:
        #     #mainwin->SetStatusText("X To, out of range, max 100000");
        #     plot.xto = oldxto
        #     return
        graphpanel.XYSynch
        self.synchcon = graphpanel.index
        self.ScaleUpdate()
    
   
    def AddScaleParam(self, label, initval, psetbox, index):
        boxwidth = 45
        boxheight = -1
        boxgap = 2

        pbox = wx.BoxSizer(wx.HORIZONTAL)
        labeltext = wx.StaticText(self, wx.ID_STATIC, label, wx.DefaultPosition, wx.Size(-1, -1), 0)

        snum = "{}".format(initval)
        numbox = TextBox(self, index, snum, wx.DefaultPosition, wx.Size(boxwidth, boxheight), wx.TE_PROCESS_ENTER)
        labeltext.SetFont(self.confont)
        numbox.SetFont(self.confont)
        pbox.Add(labeltext, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 2)
        pbox.Add(numbox, 0, wx.ALIGN_CENTER_VERTICAL, 0)

        psetbox.Add(pbox, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALL, boxgap)
        #numbox.Bind(wx.EVT_SET_FOCUS, self.OnConFocus, self)
        numbox.val = index
            
        return numbox


    def GraphSwitch(self, plotbase, command = ""):
        diag = False
        if diag: DiagWrite("GSwitch call\n")

        for graphpanel in self.mainwin.panelset:
            plotset = plotbase.GetSet(graphpanel.settag)
            #plotset = plotbase.GetSet(self.pstags[i])
            if not plotset: continue
            plottag = plotset.GetPlot(self.gflags, graphpanel.subplot)
            if not plottag: continue
            if diag: DiagWrite("graphpanel {}  set {}  plot {}  modesum {}  sync {}\n".format( 
                graphpanel.index, graphpanel.settag, plotset.tag, plotset.modesum, plotbase.GetPlot(plottag).synchx))

            # Graph Switch commands
            if command == "XSYNCH":
                refplot = graphpanel.GetFrontPlot()
                newplot = plotbase.GetPlot(plottag)
                newplot.xto = refplot.xto
                newplot.xfrom = refplot.xfrom

            # Set Panel Plots
            graphpanel.SetFrontPlot(plotbase.GetPlot(plottag))
            graphpanel.settag = plotset.tag
        
        # Update scales and plots
        self.ScaleUpdate()