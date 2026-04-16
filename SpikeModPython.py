## HypoModPython


import wx

from HypoModPy.hypomain import *


class HypoApp(wx.App):
    def OnInit(self):
        pos = wx.DefaultPosition
        size = wx.Size(400, 500)
        mainpath = ""
        respath = ""
        modname = "Spike"

        self.mainwin = HypoMain("HypoMod", pos, size, respath, mainpath, modname)
        self.SetTopWindow(self.mainwin)
        self.mainwin.Show()
        self.mainwin.SetFocus()
        go_foreground()
        return True

   
    def MacReopenApp(self):
        mainwin = self.GetTopWindow()
        if not mainwin: return

        mainwin.Show(True)
        if mainwin.IsIconized(): mainwin.Iconize(False)
        mainwin.Raise()

        for tool in mainwin.toolset.tools.values():
            if tool.box and tool.visible:
                tool.box.Show(True)
                tool.box.SetPosition(mainwin.GetPosition(), mainwin.GetSize())

    
app = HypoApp(False)
app.MainLoop()



