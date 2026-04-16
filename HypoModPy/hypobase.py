
from math import sqrt
import numpy as np
import wx
from pathlib import Path
from pubsub import pub

#from hypotools import *


# Preference Flags
basicmode = 0
studentmode = 1


# Paths
#mainpath = ""
#projectpath = "/Users/duncan/Model"
#modpath = "/Users/duncan/Model"
#modpath = ""
#modpathwin = "C:/Users/Duncan/Model"


def GetSystem():
    oslabel = wx.GetOsDescription()
    if oslabel.startswith("Windows"): return 'Windows'
    if oslabel.startswith("Mac") or oslabel.startswith("mac"): return 'Mac'
    if oslabel.startswith("Linux"): return 'Linux'
    return 0


DiagEventType = wx.NewEventType()
EVT_DIAG = wx.PyEventBinder(DiagEventType, 1)

class DiagEvent(wx.PyCommandEvent):
    def __init__(self, text=""):
        super().__init__(DiagEventType)
        self.text = text



# def DiagWrite(text):
#     #pub.sendMessage("diagbox", message=text)
#     test = 1


class TextFile():
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.readonly = True

    def Exists(self):
        return self.filepath.is_file()

    def Open(self, mode):
        if mode == 'r' and self.Exists() == False: 
            return False
        self.file = open(self.filepath, mode)
        self.unread = True
        return self.file

    def WriteLine(self, text):
        self.file.write(text + '\n')

    def WriteLines(self, textlist):
        self.file.writelines(textlist)

    def ReadLine(self):
        return self.file.readline()

    def ReadLines(self):
        return self.file.readlines()

    def Close(self):
        self.file.close()

	# Postscript Writing
	#void MoveTo(double x, double y);
	#void LineTo(double x, double y);
	#void DrawLine(double xf, double yf, double xt, double yt);
	#void DrawText(wxString, double x, double y);
	#void DrawEllipse(double x, double y, double width, double height);
	#void SetColour(wxString);


def DistXY(p1, p2):
    return sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2))


def numstring(number, places=0):
    return f"{number:.{places}f}"


def numplaces(range):
    places = 0
    if range <= 100: places = 1
    if range <= 10: places = 2
    if range <= 1: places = 3
    if range <= 0.1: places = 4	
    return places


def ParseString(readline, chartag, endtag = ' '):
    readline = readline.partition(chartag)[2]
    readline = readline.strip()
    string = readline.partition(endtag)[0]
    readline = readline.partition(endtag)[2]
    return (string, readline)


def ParseInt(readline, chartag = None):
    if chartag: readline = readline.partition(chartag)[2]         # NULL tag just reads next int
    readline = readline.strip()
    numstring = readline.partition(' ')[0]
    numdat = int(float(numstring))
    readline = readline.partition(' ')[2]
    return (numdat, readline)


def ParseFloat(readline, chartag = None):
    if chartag: readline = readline.partition(chartag)[2]         # NULL tag just reads next float
    readline = readline.strip()
    numstring = readline.partition(' ')[0]
    numdat = float(numstring)
    readline = readline.partition(' ')[2]
    return (numdat, readline)


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def CheckFloat(string):
    try:
        float(string)
        return float(string)
    except ValueError:
        return False



# Button IDs
ID_Sync = wx.NewIdRef()
ID_Store = wx.NewIdRef()
ID_Load = wx.NewIdRef()
ID_Run = wx.NewIdRef()
ID_AutoRun = wx.NewIdRef()
ID_Default = wx.NewIdRef()
ID_ModBrowse = wx.NewIdRef()
ID_Print = wx.NewIdRef()
ID_Neuron = wx.NewIdRef()
ID_Overlay = wx.NewIdRef()
ID_Pos = wx.NewIdRef()
ID_Overlay2 = wx.NewIdRef()
ID_Pos2 = wx.NewIdRef()

# Menu IDs
ID_GraphEPS = wx.NewIdRef()
ID_Scale = wx.NewIdRef()
ID_UnZoom = wx.NewIdRef()
ID_MultiEPS = wx.NewIdRef()
ID_MultiCell = wx.NewIdRef()
ID_Output = wx.NewIdRef()
ID_GraphRemove = wx.NewIdRef()
ID_SelectAll = wx.NewIdRef()
ID_Cut  = wx.NewIdRef()
ID_Copy = wx.NewIdRef()
ID_Paste = wx.NewIdRef()
ID_PasteTranspose = wx.NewIdRef()
ID_Undo = wx.NewIdRef()
ID_Insert = wx.NewIdRef()
ID_Bold = wx.NewIdRef()
ID_PlotPanel = wx.NewIdRef()
ID_Delete = wx.NewIdRef()

# Check IDs
ID_ClipMode = wx.NewIdRef()
