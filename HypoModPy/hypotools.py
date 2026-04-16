
import wx
import os
from HypoModPy.hypobase import *
from pubsub import pub



class ToolText(wx.StaticText):
    def __init__(self, parent, toolbox, tag, label, pos, size, style):
        wx.StaticText.__init__(self, parent, wx.ID_ANY, label, pos, size, style)
        self.toolbox = toolbox
        self.tag = tag

        self.Bind(wx.EVT_LEFT_UP, self.OnLeftClick)
        self.Bind(wx.EVT_LEFT_DCLICK, self.OnLeftDClick)
        self.Bind(wx.EVT_RIGHT_DCLICK, self.OnRightDClick)


    def OnLeftDClick(self, event):
        if self.toolbox: 
            self.toolbox.pinmode = 1 - self.toolbox.pinmode
            pub.sendMessage("diagbox", message=f"LDClick pin {self.toolbox.pinmode}\n")


    def OnRightDClick(self, event):
        if self.toolbox:
            self.toolbox.pinmode = 1 - self.toolbox.pinmode
            pub.sendMessage("diagbox", message=f"RDClick pin {self.toolbox.pinmode}\n")
            

    def OnLeftClick(self, event):
        if self.toolbox:
            pub.sendMessage("diagbox", message="text click\n")
            self.toolbox.activepanel.OnClick(event.GetPosition())
            self.toolbox.TextClick(self.tag)



class ToolPanel(wx.Panel):
    def __init__(self, parent, pos, size, style = wx.TAB_TRAVERSAL | wx.NO_BORDER):
        wx.Panel.__init__(self, parent, wx.ID_ANY, pos, size, style)

        self.parent = parent
        self.toolbox = None
        self.controlborder = 2

        self.blackpen = wx.Colour("#000000")
        self.redpen = wx.Colour("#dd0000")
        self.greenpen = wx.Colour("#009900")
        self.bluepen = wx.Colour("#0000dd")

        self.PanelInit()


    def PanelInit(self):
        if GetSystem() == 'Mac':
            self.buttonheight = 25
            self.boxfont = wx.Font(wx.FontInfo(12).FaceName("Tahoma"))
            self.confont = wx.Font(wx.FontInfo(11).FaceName("Tahoma"))
        else:
            self.buttonheight = 23
            self.boxfont = wx.Font(wx.FontInfo(8).FaceName("Tahoma"))
            self.confont = wx.Font(wx.FontInfo(8).FaceName("Tahoma"))

        self.Bind(wx.EVT_LEFT_UP, self.OnLeftClick)
        #self.Bind(wx.EVT_LEFT_DCLICK, self.OnLeftDClick)
        #self.Bind(wx.EVT_RIGHT_DCLICK, self.OnRightDClick)


    def OnLeftClick(self, event):
        if type(self.parent) is ToolBox:
            pos = self.parent.GetPosition()
            oldpos = self.parent.oldpos
            mpos = self.parent.mpos
            tag = self.parent.tag
            DiagWrite(f"{tag} pos {pos.x} {pos.y} old {oldpos.x} {oldpos.y} mpos {mpos.x} {mpos.y}\n")
            
        event.Skip()


class ToolButton(wx.Button):
    def __init__(self, parent, id, label, pos, size):
        wx.Button.__init__(self, parent, id, label, pos, size)
        self.parent = parent
        self.ID = id
        self.linkID = 0

        self.Bind(wx.EVT_LEFT_UP, self.OnLeftUp)


    def OnLeftUp(self, event):
        linkpress = wx.CommandEvent(wx.wxEVT_COMMAND_BUTTON_CLICKED, self.linkID)
        
        if self.linkID != 0:
            linkpress.SetInt(1)
            self.AddPendingEvent(linkpress)
            #diagbox->Write("ToolButton linkpress\n");
        
        event.Skip()

    
    def Press(self):
        press = wx.CommandEvent(wx.wxEVT_COMMAND_BUTTON_CLICKED, self.ID)
        self.AddPendingEvent(press)
    


# alt style = wx.FRAME_FLOAT_ON_PARENT | wx.FRAME_TOOL_WINDOW | wx.CAPTION | wx.RESIZE_BORDER

class ToolBox(wx.Frame):
    def __init__(self, parent, tag, title, pos, size, type = 0, 
    style = wx.FRAME_FLOAT_ON_PARENT | wx.FRAME_TOOL_WINDOW | wx.RESIZE_BORDER | wx.SYSTEM_MENU | wx.CAPTION | wx.CLOSE_BOX | wx.MINIMIZE_BOX):

        wx.Frame.__init__(self, parent, title = title, pos = pos, size = size, style = style)

        self.diagmode = False

        self.tag = tag
        self.boxtag = tag   # duplicate for backwards compatibility after rename
        self.mpos = pos - parent.GetPosition() 
        self.oldpos = pos
        self.size = size
        self.status = None
        self.canclose = False
        self.visible = True
        self.storetag = None
        self.parent = parent

        self.blackpen = wx.Colour("#000000")
        self.redpen = wx.Colour("#dd0000")
        self.greenpen = wx.Colour("#009900")
        self.bluepen = wx.Colour("#0000dd")

        self.mainbox = wx.BoxSizer(wx.VERTICAL)

        if GetSystem() == 'Mac':
            self.buttonheight = 25
            self.boxfont = wx.Font(wx.FontInfo(10).FaceName("Tahoma"))
            self.confont = wx.Font(wx.FontInfo(12).FaceName("Tahoma"))
        else:
            self.buttonheight = 23
            self.boxfont = wx.Font(wx.FontInfo(8).FaceName("Tahoma"))
            self.confont = wx.Font(wx.FontInfo(8).FaceName("Tahoma"))

        self.panel = ToolPanel(self, wx.DefaultPosition, wx.DefaultSize)
        self.panel.SetFont(self.boxfont)
        self.panel.SetSizer(self.mainbox)
        self.panel.toolbox = self

        self.selfstore = False
        self.activepanel = self.panel
        #self.paramset.panel = self.panel

        self.Bind(wx.EVT_CLOSE, self.OnClose)
        self.Bind(wx.EVT_MOVE, self.OnMove)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_ICONIZE, self.OnIconize)

        self.SetPosition(parent.GetPosition(), parent.GetSize())


    def BoxEnter(self, tag):
        if self.diagmode: self.DiagWrite("toolbox boxenter " + tag + "\n")


    def SpinClick(self, tag):
        if self.diagmode: self.DiagWrite("toolbox spinclick  " + tag + "\n")


    def StatusBar(self):
        textcon = wx.StaticText(self.activepanel, wx.ID_ANY, "", wx.DefaultPosition, wx.DefaultSize, 
        wx.ALIGN_CENTRE|wx.BORDER_DOUBLE|wx.ST_NO_AUTORESIZE)
        textcon.SetFont(self.confont)
        return textcon


    def TextLabel(self, label):
        textcon = wx.StaticText(self.activepanel, wx.ID_ANY, label)
        textcon.SetFont(self.confont)
        return textcon


    def TextInput(self, width=80, height=-1, label= "---"):
        textcon = wx.TextCtrl(self.activepanel, wx.ID_ANY, label, wx.DefaultPosition, wx.Size(width, height))
        textcon.SetFont(self.confont)
        return textcon


    def NumPanel(self, width=80, align=wx.ALIGN_RIGHT, label="0"):
        textcon = wx.StaticText(self.activepanel, wx.ID_ANY, label, wx.DefaultPosition, wx.Size(width, -1), 
        align|wx.BORDER_RAISED|wx.ST_NO_AUTORESIZE)
        textcon.SetFont(self.confont)
        return textcon


    def AddButton(self, id, label, width, box, pad=1, height=0, panel=None):
        if panel is None: panel = self.activepanel
        if height == 0: height = self.buttonheight
        button = ToolButton(panel, id, label, wx.DefaultPosition, wx.Size(width, height))
        button.SetFont(self.confont)
        box.Add(button, 0, wx.ALIGN_CENTRE_HORIZONTAL|wx.ALIGN_CENTRE_VERTICAL|wx.TOP|wx.BOTTOM, pad)
        return button


    def DiagWrite(self, text):
        pub.sendMessage("diagbox", message=text)


    def InitPosition(self, mpos):
        mainpos = self.parent.GetPosition()
        mainsize = self.parent.GetSize()
        self.Move(mainpos.x + mainsize.x + mpos.x, mainpos.y + mpos.y)   # + 5
        self.oldpos = self.GetPosition()
        self.mpos = mpos

        #snum = f"Box {self.tag} mpos x {self.mpos.x} y {self.mpos.y}"
        #DiagWrite(snum + "\n")


    def SetPosition(self, mainpos, mainsize):
        newpos = wx.Point(mainpos.x + mainsize.x + self.mpos.x, mainpos.y + self.mpos.y)

        if self.GetPosition() != newpos: self.Move(newpos)

        self.oldpos = self.GetPosition()
        return newpos
        
        #self.Move(mainpos.x + mainsize.x + self.mpos.x, mainpos.y + self.mpos.y)
        #self.oldpos = self.GetPosition()
        #return wx.Point(mainpos.x + mainsize.x + self.mpos.x, mainpos.y + self.mpos.y)


    def OnMove(self, event):
        if self.IsActive():
            newpos = self.GetPosition()
            #newsize = self.GetSize()
           
            shift = newpos - self.oldpos
        
            self.mpos.x = self.mpos.x + shift.x
            self.mpos.y = self.mpos.y + shift.y
            self.oldpos = newpos

            snum = f"Box {self.tag} mpos x {self.mpos.x} y {self.mpos.y} shift x {shift.x} y {shift.y}"
            pub.sendMessage("status_listener", message=snum)
            #DiagWrite(snum + "\n")


    def OnSize(self, event):
        event.Skip()
        newsize = self.GetSize()
        #pos = self.GetPosition()
        snum = "Box Size X {} Y {}".format(newsize.x, newsize.y)
        pub.sendMessage("status_listener", message=snum)
        self.size = newsize


    def OnClose(self, event):
        if self.canclose == False:
            self.Show(False)
        else:
            pub.sendMessage("toolclose_listener", message=self.boxtag)
            event.Skip()

    
    def OnIconize(self, event):
        DiagWrite("tool iconize call\n")
        event.Skip()

    

# mpos give the position relative to the main plot window

class ToolDat():
    def __init__(self, tag, mpos, size, visible, box=None):
        self.tag = tag
        self.mpos = mpos
        self.size = size
        self.box = box
        self.visible = visible


class ToolSet():
    def __init__(self):
        self.tools = {}


    def AddBox(self, newbox):
        if newbox.tag in self.tools:
            tool = self.GetTool(newbox.tag)
            tool.box = newbox
            newbox.oldpos = newbox.GetPosition()
            tool.box.SetSize(tool.size)
            tool.box.InitPosition(tool.mpos)
        else:
            self.tools[newbox.tag] = ToolDat(newbox.tag, newbox.GetPosition(), newbox.GetSize(), newbox.IsShown(), newbox)
            

    def AddTool(self, tag, pos, size, visible):
        self.tools[tag] = ToolDat(tag, pos, size, visible)


    def GetTool(self, tag):
        if(tag in self.tools):
            return self.tools[tag]
        return None


    def GetBox(self, tag):
        if(tag in self.tools):
             return self.tools[tag].box
        else:
             return False
             


class TextBox(wx.TextCtrl):
    def __init__(self, parent, id, value, pos, size, style):
        wx.TextCtrl.__init__(self, parent, id, value, pos, size, style)
        
    
    def GetNumValue(self):
        return float(self.GetValue())


    def SetNumValue(self, value, valrange=None):
        if valrange is None: valrange = value
        if valrange < 1:
            self.SetValue("{:.3f}".format(value))
        elif valrange < 10:
            self.SetValue("{:.2f}".format(value))
        elif valrange < 100:
            self.SetValue("{:.1f}".format(value))
        else:
            self.SetValue("{:.0f}".format(value))  


diagbox_target = None

def SetDiagBoxTarget(target):
    global diagbox_target
    diagbox_target = target

def DiagWrite(text):
    if diagbox_target is None:
        return
    diagbox_target.DiagWrite(text)


class DiagBox(ToolBox):
    def __init__(self, parent, title, pos, size):

        ToolBox.__init__(self, parent, "DiagBox", title, pos, size)

        self.textbox = wx.TextCtrl(self.panel, -1, "", wx.DefaultPosition, wx.DefaultSize, wx.TE_MULTILINE)
        self.mainbox.Add(self.textbox, 1, wx.EXPAND)
        self.Bind(EVT_DIAG, self.OnDiagEvent)

    def Write(self, text):
        try:
            self.textbox.AppendText(text)
            return True
        except ValueError:
            return False

    def DiagWrite(self, text):
        evt = DiagEvent(text)
        wx.QueueEvent(self, evt)

    def OnDiagEvent(self, event):
        self.textbox.AppendText(event.text)      



class TagBox(wx.ComboBox):
    def __init__(self, parent, label, size, boxtag, path):
        wx.ComboBox.__init__(self, parent, wx.ID_ANY, label, wx.DefaultPosition, size)

        self.boxtag = boxtag
        self.boxpath = path
        self.redtag = ""
        self.diag = False

        self.PathUpdate()

        #if(diagnostic) mainwin->diagbox->Write(text.Format("TagBox tagpath %s boxpath %s\n", tagpath, boxpath));
        if self.diag: print("TagBox " + path)

        if os.path.exists(self.tagpath) == False: 
            os.mkdir(self.tagpath)

        if self.diag: print("tagpath " + self.tagpath)

        # Read fixed location option file, directs to selectable tagfile location
        opfilepath = self.tagpath + "/" + boxtag + "-op.ini"
        opfile = TextFile(opfilepath)
        check = opfile.Open('r')
        if check == False:
            pub.sendMessage("diagbox", message = "TagBox opfile " + opfilepath + "\n")
            pub.sendMessage("diagbox", message = "TagBox " + boxtag + " No tagpath found, setting default\n")
            self.tagfilename = boxtag + "tags.ini"
        else:
            readline = opfile.ReadLine()
            if readline == "": self.tagfilename = boxtag + "tags.ini"
            else: self.tagfilename = readline.strip()
            opfile.Close()

        #mainwin->diagbox->Write("\nTagBox init " + name + "\n");
        self.HistLoad()

        #Connect(wxEVT_RIGHT_UP, wxMouseEventHandler(TagBox::OnRClick));


    def PathUpdate(self):
        #if self.boxpath == "": 
        #    if projectpath == "": self.tagpath = "Tags"
        #    else: self.tagpath = projectpath + "/Tags"
        #else:
        #    if projectpath == "": self.tagpath = self.boxpath + "/Tags"
        #    else: self.tagpath = projectpath + "/" + self.boxpath + "/Tags"

        
        if self.boxpath == "": self.tagpath = "Tags"
        else: self.tagpath = self.boxpath + "/Tags"

        if os.path.exists(self.tagpath) == False: 
            os.mkdir(self.tagpath)

        if self.diag:
            pub.sendMessage("diagbox", message="TagBox PathUpdate() tagpath {}\n".format(self.tagpath))


    def HistStore(self):
        # Tag history
        if self.tagfilename == "": return
        filepath = self.tagpath + "/" + self.tagfilename
        tagfile = TextFile(filepath)
        tagfile.Open('w')
        if self.GetCount() > 0:
            #print("HistStore count {}".format(self.GetCount()))
            for i in range(0, self.GetCount()):
                #print("HistStore i {}".format(self.GetCount() - i - 1))
                outline = "tag {}".format(self.GetString(self.GetCount() - i - 1))
                tagfile.WriteLine(outline)
        tagfile.Close()

        # Fixed location option file, directs to selectable tagfile location
        opfile = TextFile(self.tagpath + "/" + self.boxtag + "-op.ini")
        opfile.Open('w')
        opfile.WriteLine(self.tagfilename)
        opfile.Close()

        if self.diag: print("HistStore tagfile " + filepath)


    def HistLoad(self):
        diag = False

        tag = ""
        if self.tagpath == "":
            DiagWrite("Tag file not set\n")
            return

        filepath = self.tagpath + "/" + self.tagfilename
        tagfile = TextFile(filepath)
        check = tagfile.Open('r')
        if check == False:
            DiagWrite("No tag history\n")
            return

        if diag: 
            DiagWrite("HistLoad ")
            DiagWrite("Reading tag history " + self.tagfilename + "\n")
        
        filetext = tagfile.ReadLines()
        for readline in filetext:
            readdata = readline.split(' ')
            tag = readdata[1].strip()
            # diagbox->Write("Readline " + readline + "\n");
            self.Insert(tag, 0)
            # diagbox->Write("Insert " + tag + "\n");

        tagfile.Close()	
        self.SetLabel(tag)
        if diag: DiagWrite(self.boxtag + " " + tag + "\n")
        if tag != "": self.labelset = True
