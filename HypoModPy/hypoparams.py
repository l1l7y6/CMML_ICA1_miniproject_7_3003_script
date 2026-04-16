
import wx
from HypoModPy.hypotools import *



class ParamCon(wx.Control):
    def __init__(self, panel, type, tag, labeltext, initval, step=0, places=0, labelwidth=60, datawidth=45):
        ostype = GetSystem()
        wx.Control.__init__(self, panel, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.BORDER_NONE)
        self.numstep = step
        self.tag = tag
        self.labeltext = labeltext
        self.decimals = places
        self.type = type
        self.labelwidth = labelwidth
        self.datawidth = datawidth
        self.buttonwidth = 0
        self.panel = panel
        self.pad = panel.controlborder
        self.cycle = False
        self.oldvalue = initval

        self.diagmode = False

        if ostype == "Mac": pad = 0
        else: pad = 0

        textfont = wx.Font(wx.FontInfo(8).FaceName("Tahoma"))

        if ostype == "Mac":
            textfont = wx.Font(wx.FontInfo(11).FaceName("Tahoma"))
            smalltextfont = wx.Font(wx.FontInfo(9).FaceName("Tahoma"))

        self.min = 0
        self.max = 1000000

        if type == 'numcon' or type == 'spincon':
            if initval < 0: self.min = -1000000
            if initval < self.min: self.min = initval * 10
            if initval > self.max: self.max = initval * 100
            oldvalue = initval
            inittext = numstring(initval, places)
        else:
            inittext = initval

        self.sizer = wx.BoxSizer(wx.HORIZONTAL)

        if labeltext == "":
            self.label = None
            self.labelwidth = 0
        else:
            self.label = ToolText(self, panel.parent, tag, labeltext, wx.DefaultPosition, wx.Size(labelwidth, -1), wx.ALIGN_CENTRE)
            self.label.SetFont(textfont)

        #if ostype == 'Mac' and labelwidth < 40: label.SetFont(smalltextfont)
        self.sizer.Add(self.label, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, pad)

        #if type == "textcon": print(f"ParamCon init: {initval}")

        self.numbox = wx.TextCtrl(self, wx.ID_ANY, inittext, wx.DefaultPosition, wx.Size(datawidth, -1), wx.TE_PROCESS_ENTER)
        self.numbox.SetFont(textfont)
        self.sizer.Add(self.numbox, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, pad)

        if type == 'spincon':
            self.spin = wx.SpinButton(self, wx.ID_ANY, wx.DefaultPosition, wx.Size(17, 23), wx.SP_VERTICAL|wx.SP_ARROW_KEYS);  # 21
            self.spin.SetRange(-1000000, 1000000)
            self.sizer.Add(self.spin, 0, wx.ALIGN_CENTER_VERTICAL, 0)
            self.spin.Bind(wx.EVT_SPIN_UP, self.OnSpinUp)
            self.spin.Bind(wx.EVT_SPIN_DOWN, self.OnSpinDown)
            self.spin.Bind(wx.EVT_SPIN, self.OnSpin)

        self.SetInitialSize(wx.DefaultSize)
        self.Move(wx.DefaultPosition)
        self.SetSizer(self.sizer)
        self.Layout()

        self.Bind(wx.EVT_TEXT_ENTER, self.OnEnter)


    def AddButton(self, label, id, width):
        self.buttonwidth = width
        self.button = wx.Button(self, id, label, wx.DefaultPosition, wx.Size(self.buttonwidth, 25))
        self.button.SetFont(self.panel.confont)
        self.sizer.Add(self.button, 0, wx.ALIGN_CENTRE_HORIZONTAL|wx.ALIGN_CENTRE_VERTICAL|wx.TOP|wx.BOTTOM, 2)
        #self.SetInitialSize(wx.DefaultSize)
        self.Layout()
        return self.button
    

    def DoGetBestSize(self):
        return wx.Size(self.labelwidth + self.datawidth + self.buttonwidth, 25)


    def Select(self):
        self.panel.toolbox.TextClick(self.tag)


    def GetValue(self):
        if self.type == 'textcon': return 0
        value = self.numbox.GetValue()
        if not isfloat(value): return 0
        return float(value)


    def GetText(self):
        return self.numbox.GetValue()


    def SetText(self, text):
        return self.numbox.SetValue(text)

    
    def Clear(self):
        return self.numbox.SetValue("")


    def SetPen(self, pen):
        self.numbox.SetForegroundColour(pen)


    def SetValue(self, value):
        if self.type == 'textcon': snum = value
        else: snum = numstring(value, self.decimals)
        self.numbox.SetValue(snum)


    def SetMinMax(self, newmin, newmax, cycle = False):
        self.min = newmin
        self.max = newmax
        self.cycle = cycle


    def DoGetBestSize(self): 
        if GetSystem() == 'Mac':
            if self.type == 'spincon': return wx.Size(self.datawidth + self.labelwidth + self.pad * 2 + 17, 23)
            else: return wx.Size(self.datawidth + self.labelwidth + self.pad * 2, 20)

        if self.type == 'spincon': return wx.Size(self.datawidth + self.labelwidth + 17, 25)
        else: return wx.Size(self.datawidth + self.labelwidth + self.pad * 2, 21 + self.pad * 2)


    def OnSpin(self, event):
        if self.panel.toolbox is not None:
            if self.diagmode: pub.sendMessage("diagbox", message="tool spin click\n")
            self.panel.toolbox.SpinClick(self.tag)
        event.Skip()

    
    def OnEnter(self, event):
        if self.panel.toolbox is not None:
            pub.sendMessage("diagbox", message="tool box enter\n")
            self.panel.toolbox.BoxEnter(self.tag)
        event.Skip()

    
    def OnSpinUp(self, event):
        value = float(self.numbox.GetValue())
        newvalue = value + self.numstep
        if newvalue > self.max:
            if self.cycle: newvalue = self.min + (newvalue - self.max) - 1
            else: return
        snum = numstring(newvalue, self.decimals)
        self.numbox.SetValue(snum)


    def OnSpinDown(self, event):
        value = float(self.numbox.GetValue())
        newvalue = value - self.numstep
        #snum = "SpinDown value {} newvalue {} min {}\n".format(value, newvalue, self.min)
        #pub.sendMessage("diagbox", message=snum)
        if newvalue < self.min: 
            if self.cycle: newvalue = self.max + (newvalue - self.min) + 1
            else: return
        snum = numstring(newvalue, self.decimals)
        self.numbox.SetValue(snum)
        


class ParamSet:
    def __init__(self, panel):
        self.pcons = {}         # parameter controls
        self.paramstore = {}    # parameter value store
        self.panel = panel      # ToolPanel link
        self.currlay = 0        # layout counter for use with multiple ParamLayout calls

        # Default field widths
        self.num_labelwidth = 65
        self.num_numwidth = 40
        self.con_labelwidth = 60
        self.con_numwidth = 60
        self.text_labelwidth = 60
        self.text_numwidth = 150


    def Check(self, tag):
        return tag in self.pcons


    def NumParams(self):
        return len(self.pcons)


    def SetMinMax(self, tag, min, max):
        self.pcons[tag].min = min
        self.pcons[tag].max = max


    def GetCon(self, tag):
        if not tag in self.pcons:
            pub.sendMessage("diagbox", message="ParamSet GetCon " + tag + " not found\n")
            return None
        else: return self.pcons[tag]


    def AddCon(self, tag, label, initval, step, places, labelwidth=-1, numwidth=-1): 
        if labelwidth < 0: labelwidth = self.con_labelwidth
        if numwidth < 0: numwidth = self.con_numwidth
        self.pcons[tag] = ParamCon(self.panel, 'spincon', tag, label, initval, step, places, labelwidth, numwidth);   # number + spin
        return self.pcons[tag]


    def AddNum(self, tag, label, initval, places, labelwidth=-1, numwidth=-1):
        if labelwidth < 0: labelwidth = self.num_labelwidth
        if numwidth < 0: numwidth = self.num_numwidth
        self.pcons[tag] = ParamCon(self.panel, 'numcon', tag, label, initval, 0, places, labelwidth, numwidth);   # number
        return self.pcons[tag]


    def AddText(self, tag, label, initval, labelwidth=-1, textwidth=-1):
        if labelwidth < 0: labelwidth = self.text_labelwidth
        if textwidth < 0: textwidth = self.text_numwidth
        self.pcons[tag] = ParamCon(self.panel, 'textcon', tag, label, initval, labelwidth, textwidth)     # text
        return self.pcons[tag]

    
    def SetValue(self, tag, value):
        self.pcons[tag].SetValue(value)


    def GetValue(self, tag):
        if not tag in self.pcons: return 0
        value = self.pcons[tag].GetValue()
        return float(value)


    def GetText(self, tag):
        text = self.pcons[tag].GetString()
        return text

    
    def GetParams(self):
        for pcon in self.pcons.values():
            value = pcon.GetValue()
            if value < pcon.min:
                value = pcon.oldvalue
                pcon.SetValue(value)

            if value > pcon.max or value > pcon.max:
                value = pcon.oldvalue
                pcon.SetValue(value)

            self.paramstore[pcon.tag] = value
            pcon.oldvalue = value

        return self.paramstore



class ParamBox(ToolBox):
    def __init__(self, model, title, pos, size, tag, type = 0, storemode = 0):
        ToolBox.__init__(self, model.mainwin, tag, title, pos, size, type)
        diag = False

        self.autorun = 0    # auto run model after parameter change
        self.redtag = ""    # store box overwrite warning tag
        self.histmode = 0
        self.storemode = storemode
        self.mod = model   # parent model      
        self.boxtype = type   # 0 - basic panel, 1 - AUI panel
        self.status = None
        #defbutt = 0;
        #defstore = false;
        self.diag = 0   # diagnostic mode
        self.mainwin = model.mainwin  # main window link

        # modmode = 0;
        
        self.activepanel = self.panel
        
        if diag: self.DiagWrite("ParamBox " + self.boxtag + " init\n")

        # Initialise Layout
        self.column = 0     # column mode for parameter controls
        self.buttonwidth = 50
        self.vbox = []
        self.buttonbox = wx.BoxSizer(wx.HORIZONTAL)
        self.panelbuttoncount = 0
        if self.boxtype == 0: self.pconbox = wx.BoxSizer(wx.HORIZONTAL)
        self.defbutt = False

        # Initialise Stores
        self.paramset = ParamSet(self.panel)
        self.modparams = {}
        self.modflags = {}
        self.conflags = {}
        self.checktags = {}
        self.checkboxes = {}
        self.flagtags = {}
        self.flagIDs = {}
        self.panelrefs = {}
        
        print("ParamBox " + self.mod.path)

        # Store Tag
        self.storetag = None
        if self.storemode:
            if diag: self.DiagWrite("Store Box initialise " + self.tag + "\n")
            self.storetag = TagBox(self.activepanel, "", wx.Size(120, 20), self.tag, self.mod.path)
            self.storetag.Show(False)
            self.storetag.SetFont(self.confont)

        self.Bind(wx.EVT_MENU, self.OnAutoRun, ID_AutoRun)
        self.Bind(wx.EVT_BUTTON, self.OnRun, ID_Run)
        self.Bind(wx.EVT_BUTTON, self.OnDefault, ID_Default)
        self.Bind(wx.EVT_TEXT_ENTER, self.OnRun)
        self.Bind(wx.EVT_SPIN, self.OnSpin)

        #self.Bind(wx.EVT_MENU, self.OnQuit, fileItem)


    # string formatting examples
    #
    # Box mpos x {} y {} shift x {} y {}".format(self.mpos.x, self.mpos.y, shift.x, shift.y)
    # "{:.0f}".format(xval + plot.xdis)


    def ParamStore(self, filetag = ""):
        newtag = False
        if filetag == "": newtag = True
        parampath = self.mod.path + "/Params"
        if os.path.exists(parampath) == False: 
            os.mkdir(parampath)

        if self.storetag is not None:
            if filetag == "": filetag = self.storetag.GetValue()
            else: self.storetag.SetValue(filetag)

        # Param data file
        filepath = parampath + "/" + filetag + "-" + self.boxtag + "param.dat";

        # Param file history
        if self.storetag is not None and filetag != "default": 
            tagpos = self.storetag.FindString(filetag)
            if tagpos != wx.NOT_FOUND: self.storetag.Delete(tagpos)
            self.storetag.Insert(filetag, 0)
            print("tag inserted " + filetag)

        # Overwrite Warning
        paramfile = TextFile(filepath)
        if paramfile.Exists() and newtag and self.redtag != filetag: 
            if self.storetag is not None:
                self.storetag.SetForegroundColour(self.redpen)
                self.storetag.SetValue("")
                self.storetag.SetValue(filetag)
            self.redtag = filetag
            return

        # Clear Overwrite Warning
        self.redtag = ""
        if self.storetag is not None:
            self.storetag.SetForegroundColour(self.blackpen)
            self.storetag.SetValue("")
            self.storetag.SetValue(filetag)

        # Open File
        paramfile.Open('w')

        # Write Parameter Values
        for con in self.paramset.pcons.values():
            if con.type != "textcon":
                outline = "{:.8f}".format(con.GetValue())
            else: outline = con.GetString()
            paramfile.WriteLine(con.tag + " " + outline)

        # Write Flag Values
        paramfile.WriteLine("")
        for flagtag in self.flagtags.values():
            outline = f"{self.modflags[flagtag]}"
            #outline = "%.0f".format(self.modflags[flagtag])
            paramfile.WriteLine(flagtag + " " + outline)

        # Write Check Values
        paramfile.WriteLine("")
        for checktag in self.checktags.values():
            outline = f"{self.modflags[checktag]}"
            #outline = "%.0f".format(self.modflags[checktag])
            paramfile.WriteLine(checktag + " " + outline)
  
        # Close File
        paramfile.Close()
        self.DiagWrite("Param File OK\n")


    def ParamLoad(self, filetag = "", compmode = False):
        diagmode = False
        if diagmode: DiagWrite("param load {}\n".format(self.boxtag))

        oldparams = self.GetParams()
        parampath = self.mod.path + "/Params"

        # Param data file
        if self.storetag is not None:
            if filetag == "": filetag = self.storetag.GetValue()
            elif filetag != "default": self.storetag.SetValue(filetag)

        filepath = parampath + "/" + filetag + "-" + self.tag + "param.dat"
        if diagmode: DiagWrite("paramload " + filepath + "\n")
        paramfile = TextFile(filepath)

        if not paramfile.Exists():
            if self.storetag: self.storetag.SetValue("Not found")
            return

        # Param file history
        if self.storetag is not None and filetag != "default": 
            tagpos = self.storetag.FindString(filetag)
            if tagpos != wx.NOT_FOUND: self.storetag.Delete(tagpos)
            self.storetag.Insert(filetag, 0)
            print("tag inserted " + filetag)

        # Clear Overwrite Warning
        self.redtag = ""
        if self.storetag is not None:
            self.storetag.SetForegroundColour(self.blackpen)
            self.storetag.SetValue("")
            self.storetag.SetValue(filetag)

        # Open File
        paramfile.Open('r')

        # Read Parameter Values
        mode = "param"
        filetext = paramfile.ReadLines()
        
        for readline in filetext:

            # parse line
            readline = readline.strip()
            if readline == "":
                if mode == "param": mode = "flag"
                elif mode == "flag": mode = "check"
                continue
            readdata = readline.split(' ')
            tag = readdata[0]
            data = readdata[1]

            # read parameter
            if mode == "param":
                if self.paramset.Check(tag):
                    paramcon = self.paramset.pcons[tag]
                    if paramcon.type != "textcon":
                        datval = float(data)
                        paramcon.SetPen(self.blackpen)
                        if compmode and datval != oldparams[tag]:
                            paramcon.SetPen(self.greenpen)
                            DiagWrite(tag + " param change\n")
                        paramcon.Clear()
                        paramcon.SetValue(datval)
                    else: paramcon.SetValue(data)
                    if diagmode: DiagWrite("Model Param Tag {}, Value {:.4f}\n".format(tag, datval)) 

            # read flag
            if mode == "flag":
                if tag in self.flagIDs:
                    flagval = int(data)
                    self.modflags[tag] = flagval
                    id = self.flagIDs[tag]
                    self.menuModel.Check(id, flagval)
                    if diagmode: DiagWrite("Model flag ID {}, Tag {}, Set %d\n".format(id, tag, flagval)) 

            # read check
            if mode == "check":
                if tag in self.checkIDs:
                    checkval = int(data)
                    self.modflags[tag] = checkval
                    checkbox = self.checkboxes[tag]
                    checkbox.SetValue(checkval)
                    if diagmode: DiagWrite("Model check Tag {}, Set {}\n".format(tag, flagval)) 

        paramfile.Close()


    def GetParams(self, pstore = None):
        if pstore is None: pstore = self.modparams
        for con in self.paramset.pcons.values():
            value = con.GetValue()
            if value < con.min:
                value = con.oldvalue
                con.SetValue(value)
                if con.label is not None:
                    self.SetStatus("Parameter \'{}\' out of range".format(con.label.GetLabel()))
                    self.DiagWrite("Parameter \'{}\' out of range, min {:.2f} max {:.2f}\n".format(con.label.GetLabel(), con.min, con.max))

            if value > con.max:
                value = con.oldvalue
                con.SetValue(value)
                if con.label is not None:
                    self.SetStatus("Parameter {} out of range".format(con.label.GetLabel()))

            pstore[con.tag] = value
            con.oldvalue = value

        return pstore


    def RunBox(self):
        runbox = wx.BoxSizer(wx.HORIZONTAL)
        self.runcount = self.NumPanel(50, wx.ALIGN_CENTRE, "---")
        if GetSystem() == "Mac": self.AddButton(ID_Run, "RUN", 50, runbox)
        else: self.AddButton(ID_Run, "RUN", 70, runbox)
        runbox.AddSpacer(5)
        runbox.Add(self.runcount, 0, wx.ALIGN_CENTRE_HORIZONTAL|wx.ALIGN_CENTRE_VERTICAL)

        if self.defbutt:
            runbox.AddSpacer(5)
            if GetSystem() == "Mac": self.AddButton(ID_Default, "RESET", 50, runbox)
            else: self.AddButton(ID_Default, "RESET", 70, runbox)
	
        return runbox
    

    def SetCount(self, value):
        if self.runcount is not None:
            self.runcount.SetLabel(f"{value} %")
            #DiagWrite(f"Set count, value {value}\n\n")


    def InitMenu(self, type = "menu_model"):
        if type == "menu_model":
            self.menuControls = wx.Menu()
            self.menuControls.Append(ID_AutoRun, "Auto Run", "Toggle Autorun", wx.ITEM_CHECK)
            self.menuControls.Check(ID_AutoRun, self.autorun)
            self.menuModel = wx.Menu()
            menuBar = wx.MenuBar()
            menuBar.Append(self.menuControls, "Controls")
            menuBar.Append(self.menuModel, "Model")

        if type == "menu_gridbox":
            self.menuMode = wx.Menu()
            menuBar = wx.MenuBar()
            menuBar.Append(self.menuMode, "Mode")

        self.SetMenuBar(menuBar)
       

    def AddPanelButton(self, id, label, toolbox):
        if self.panelbuttoncount > 0:
            self.buttonbox.AddSpacer(5)
            self.buttonbox.AddStretchSpacer()
        self.panelrefs[id] = toolbox
        button = self.AddButton(id, label, self.buttonwidth, self.buttonbox)
        button.Bind(wx.EVT_BUTTON, self.OnPanel)
        self.panelbuttoncount += 1
       

    def OnPanel(self, event):
        id = event.GetId()
        toolbox = self.panelrefs[id]
        if toolbox.IsShown(): toolbox.Show(False)
        else: toolbox.Show(True)


    """ def AddFlag(self, id, flagtag, flagtext, state = False, menu = None):
        if menu is None: menu = self.menuModel
        self.modflags[flagtag] = state
        self.flagtags[id] = flagtag
        self.flagIDs[flagtag] = id
        menu.Append(id, flagtext, "Toggle " + flagtext, wx.ITEM_CHECK)
        menu.Check(id, state)
        self.Bind(wx.EVT_MENU, self.OnFlag, id) """


    """ def AddFlag(self, flagtag, flagtext, state=False, menu=None):
        if menu is None: menu = self.menuModel
        item = menu.AppendCheckItem(wx.ID_ANY, flagtext, "Toggle " + flagtext)
        item.Check(bool(state))
        self.modflags[flagtag] = bool(state)
        item.flagtag = flagtag
        self.Bind(wx.EVT_MENU, self.OnFlag, item)
        return item """
    

    """ def AddFlag(self, flagtag, flagtext, state=False, menu=None):
        if menu is None: menu = self.menuModel
        item = menu.AppendCheckItem(wx.ID_ANY, flagtext, "Toggle " + flagtext)
        item.Check(bool(state))
        self.modflags[flagtag] = bool(state)
        self.Bind(wx.EVT_MENU, lambda evt, tag=flagtag: self.OnFlag(evt, tag), source=item)
        return item """
    

    def AddFlag(self, flagtag, flagtext, state=False, menu=None):
        if menu is None: menu = self.menuModel
        item = menu.AppendCheckItem(wx.ID_ANY, flagtext, "Toggle " + flagtext)
        item.Check(bool(state))
        id = item.GetId()
        self.modflags[flagtag] = bool(state)
        self.flagtags[id] = flagtag
        self.Bind(wx.EVT_MENU, self.OnFlag, id=id)
        return item


    def AddCheck(self, id, checktag, checktext, state):
        self.modflags[checktag] = state
        self.checktags[id] = checktag
        newcheck = wx.CheckBox(self.activepanel, id, checktext)
        newcheck.SetFont(self.confont)
        newcheck.SetValue(state)
        newcheck.Bind(wx.EVT_CHECKBOX, self.OnCheck)
        self.checkboxes[checktag] = newcheck
        return newcheck


    """ def OnFlag(self, event):
        id = event.GetId()
        flagtag = self.flagtags[id]
        if self.modflags[flagtag] == 0: self.modflags[flagtag] = 1
        else: self.modflags[flagtag] = 0
        if self.autorun: self.OnRun(event)


    def OnFlag(self, event):
        item = event.GetEventObject()
        flagtag = getattr(item, "flagtag", None)

        if flagtag is None:
            event.Skip()
            return

        self.modflags[flagtag] = item.IsChecked() 
        
        
    def OnFlag(self, event, flagtag):
	    item = event.GetEventObject()
	    self.modflags[flagtag] = item.IsChecked()
        
        """


 
    def OnFlag(self, event):
        id = event.GetId()
        flagtag = self.flagtags[id]
        item = self.menuModel.FindItemById(id)
        self.modflags[flagtag] = item.IsChecked()
        

    def OnCheck(self, event):
        id = event.GetId()
        checktag = self.checktags[id]
        if self.modflags[checktag] == 0: self.modflags[checktag] = 1
        else: self.modflags[checktag] = 0
    

    def OnDefault(self, event):
        self.ParamLoad("default")
        if self.autorun: self.OnRun(event)


    def OnSpin(self, event):
        #self.DiagWrite("ParamBox on spin\n") 
        if self.autorun: self.OnRun(event)


    def OnRun(self, event):
        self.countmark = 0
        #self.GetParams()
        self.mod.RunModel()


    def OnAutoRun(self, event):
        self.autorun = 1 - self.autorun
        print(f"AutoRun {self.autorun}")


    def SetStatus(self, text):
        if self.status is not None: self.status.SetLabel(text)


    def WriteVDU(self, text):
        if self.vdu is not None: self.vdu.AppendText(text)


    def VBox(self, num):
        for i in range(num):
            self.vbox[i] = wx.BoxSizer(wx.VERTICAL)
            self.vbox[i].AddSpacer(5)


    def ParamLayout(self, numcols = 1):                  
        colsize = 0
        numparams = self.paramset.NumParams()

        if numcols == 1: colsize = numparams
        if(numcols >= 2):
            colsize = int((numparams + 1) / numcols) 

        pstart = 0
        for col in range(numcols):
            if col == numcols-1: pstop = numparams
            else: pstop = colsize * (col+1)
            vbox = wx.BoxSizer(wx.VERTICAL)
            vbox.AddSpacer(5)
            for p in range(pstart, pstop):
                vbox.Add(list(self.paramset.pcons.values())[p], 1, wx.ALIGN_CENTRE_HORIZONTAL|wx.ALIGN_CENTRE_VERTICAL|wx.RIGHT|wx.LEFT, 5)
                vbox.AddSpacer(5)
            self.pconbox.Add(vbox, 0)
            pstart = pstop


    def StoreBoxSync(self, label="", storepanel=None):
        self.synccheck = wx.CheckBox(self.panel, wx.ID_ANY, "Sync")
        self.synccheck.SetValue(True)
        storebox = self.StoreBox(label, storepanel)
        storebox.Add(self.synccheck, 0, wx.ALIGN_CENTRE_HORIZONTAL|wx.ALIGN_CENTRE_VERTICAL|wx.ALL, 2)
        return storebox


    def StoreBox(self, label="", storepanel=None):
        if self.storetag is None: return
        paramfilebox = wx.BoxSizer(wx.HORIZONTAL)
        parambuttons = wx.BoxSizer(wx.HORIZONTAL)

        if storepanel is None: storepanel = self.panel
        if self.activepanel != self.panel: self.storetag.Reparent(self.activepanel)

        if label != "": self.storetag.SetLabel(label)
        self.storetag.Show(True)

        self.AddButton(wx.ID_ANY, "Store", 38, parambuttons).Bind(wx.EVT_BUTTON, self.OnStore)
        parambuttons.AddSpacer(2)
        self.AddButton(wx.ID_ANY, "Load", 38, parambuttons).Bind(wx.EVT_BUTTON, self.OnLoad)
        
        paramfilebox.Add(self.storetag, 0, wx.ALIGN_CENTRE_HORIZONTAL|wx.ALIGN_CENTRE_VERTICAL|wx.ALL, 2)
        paramfilebox.Add(parambuttons, 0, wx.ALIGN_CENTRE_HORIZONTAL|wx.ALIGN_CENTRE_VERTICAL|wx.ALL, 2)
        return paramfilebox

    
    def OnStore(self, event):
        self.ParamStore()


    def OnLoad(self, event):
        self.ParamLoad()

    