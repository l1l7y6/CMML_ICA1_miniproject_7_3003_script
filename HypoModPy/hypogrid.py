
import wx.grid
from HypoModPy.hypobase import *
from HypoModPy.hypoparams import *
#from wx.lib.sheet import *
import wx.richtext
from io import StringIO 
import wx.py.buffer
import pyperclip
import math
from HypoModPy.hypospikes import *


class TextGrid(wx.grid.Grid):
    def __init__(self, parent, size):
        wx.grid.Grid.__init__(self, parent, wx.ID_ANY)
        #CSheet.__init__(self, parent)

        self.ostype = GetSystem()

        self.CreateGrid(size.x, size.y)
        #self.SetGridSize(size.x, size.y)
        self.SetRowLabelSize(35)
        self.SetColLabelSize(25)
        self.SetRowLabelAlignment(wx.ALIGN_RIGHT, wx.ALIGN_CENTRE)
        self.SetDefaultCellAlignment(wx.ALIGN_LEFT, wx.ALIGN_CENTRE)    
        self.SetLabelFont(wx.Font(9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))

        self.undogrid = wx.grid.GridStringTable(size.x, size.y)
        self.vdu = None
        self.gridbox = None
        self.mod = None
        self.selectcol = 0
        self.selectrow = 0

        self.rightmenu = wx.Menu()
        self.rightmenu.Append(ID_SelectAll, "Select All", "Grid Select", wx.ITEM_NORMAL)
        self.rightmenu.Append(ID_Copy, "Copy", "Copy Selection", wx.ITEM_NORMAL)
        self.rightmenu.Append(ID_Paste, "Paste", "Paste Clipboard", wx.ITEM_NORMAL)
        self.rightmenu.Append(ID_PasteTranspose, "Paste Transpose", "Paste Clipboard", wx.ITEM_NORMAL)
        self.rightmenu.Append(ID_Undo, "Undo", "Undo", wx.ITEM_NORMAL)
        self.rightmenu.Append(ID_Insert, "Insert Col", "Insert Column", wx.ITEM_NORMAL)
        self.rightmenu.Append(ID_Delete, "Delete", "Delete", wx.ITEM_NORMAL)

        self.Bind(wx.grid.EVT_GRID_CELL_RIGHT_CLICK, self.OnRightClick)
        self.Bind(wx.grid.EVT_GRID_CELL_LEFT_CLICK, self.OnLeftClick)
        self.Bind(wx.grid.EVT_GRID_LABEL_LEFT_CLICK, self.OnLabelClick)

        self.Bind(wx.EVT_MENU, self.OnSelectAll, ID_SelectAll)
        self.Bind(wx.EVT_MENU, self.OnCut, ID_Cut)
        self.Bind(wx.EVT_MENU, self.OnCopy, ID_Copy)
        self.Bind(wx.EVT_MENU, self.OnPaste, ID_Paste)
        self.Bind(wx.EVT_MENU, self.OnPaste, ID_PasteTranspose)
        self.Bind(wx.EVT_MENU, self.OnUndo, ID_Undo)
        self.Bind(wx.EVT_MENU, self.OnBold, ID_Bold)
        self.Bind(wx.EVT_MENU, self.OnInsertColumn, ID_Insert)
        self.Bind(wx.EVT_MENU, self.OnDelete, ID_Delete)

        self.Bind(wx.EVT_KEY_DOWN, self.OnKey)


    def WriteVDU(self, text):
        if self.vdu: self.vdu.AppendText(text)


    def SetGridSize(self, numrows = 0, numcols = 0):
        if numrows: self.SetNumberRows(numrows)
        if numcols: self.SetNumberCols(numcols)
       

    def OnRightClick(self, event):
        pos = event.GetPosition()
        self.PopupMenu(self.rightmenu, pos.x - 20, pos.y)


    def CopyColumn(self, source, dest):
        numrows = self.GetNumberRows()

        for i in range(numrows):
            celltext = self.GetCellValue(i, source)
            self.SetCellValue(i, dest, celltext)


    def InsertColumn(self, col):
        self.InsertCols(col)


    def ReadFloat(self, row, col):
        celltext = self.GetCell(row, col)
        celltext = celltext.strip()
        if not celltext == "": celldata = float(celltext)
        else: return 0
        return celldata

    
    def GetCell(self, row, col):
        numrows = self.GetNumberRows()
        numcols = self.GetNumberCols()

        if row >= numrows or col >= numcols: return "" 
        else: return self.GetCellValue(row, col)


    def SetCell(self, row, col, data):
        numrows = self.GetNumberRows()
        numcols = self.GetNumberCols()

        if row >= numrows: 
            self.AppendRows(row - numrows + 100)
            print(f"Adding row {numrows} {row}")

        if col >= numcols: 
            self.AppendCols(col - numcols + 10)
            print(f"Adding cols {col}")


        #DiagWrite(f"SetCell row {row} col {col} data {data}\n")
        self.SetCellValue(row, col, data)


    def OnKey(self, event):
        if event.ControlDown(): DiagWrite(f"Control {event.GetKeyCode()}\n")

        if event.ControlDown() and event.GetKeyCode() == 67: self.Copy()        # 67 = C
               
        elif event.ControlDown() and event.GetKeyCode() == 86: self.Paste()     # 86 = V

        elif event.ControlDown() and event.GetKeyCode() == 69: self.Paste(2)    # 69 = E    # workaround Excel mode for RTF double \n on Mac

        elif event.ControlDown() and event.GetKeyCode() == 84: self.Paste(1)    # 84 = T

        elif event.ControlDown() and event.GetKeyCode() == 88: self.Cut()       # 88 = X

        elif event.ControlDown() and event.GetKeyCode() == 65: self.SelectAll() # 65 = A

        elif event.ControlDown() and event.GetKeyCode() == 90: self.Undo()      # 90 = Z

        elif event.ControlDown() and event.GetKeyCode() == 8: self.Delete()      # 8 = Mac Delete

        elif event.GetKeyCode() == wx.WXK_DELETE:
            self.Delete()
            return

        else: event.Skip()


    def ClearCol(self, col):
        for i in range(self.GetNumberRows()):
            self.SetCellValue(i, col, "")


    def SetBold(self):
        self.CopyUndo()

        for i in range(self.GetNumberRows()):     
            for j in range(self.GetNumberCols()): 
                if self.IsInSelection(i, j): self.SetCellFont(i, j, wx.Font(9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))

        self.Refresh()


    def OnSelectAll(self, event):
        self.SelectAll()


    def OnInsertColumn(self, event):
        col = self.GetGridCursorCol()
        self.InsertColumn(col)


    def OnCut(self, event):
        self.Cut()

    
    def OnCopy(self, event):
        self.Copy()


    def OnPaste(self, event):
        if event.GetId() == ID_PasteTranspose: self.Paste()
        else: self.Paste()


    def OnBold(self, event):
        self.SetBold()

    
    def OnDelete(self, event):
        self.Delete()


    def OnInsertColumn(self, event):
        col = self.GetGridCursorCol()
        self.InsertColumn(col)


    def Delete(self):
        self.CopyUndo()
        self.SetCellValue(self.GetGridCursorRow(), self.GetGridCursorCol(), "")
        for i in range(self.GetNumberRows()):     
            for j in range(self.GetNumberCols()): 
                if self.IsInSelection(i, j): self.SetCellValue(i, j, "")


    def Cut(self):
        self.Copy()
        self.Delete()


    def OnUndo(self, event):
        self.Undo()


    def OnLabelClick(self, event):
        #if self.vdu: self.vdu.AppendText("Label Click\n")
        c = event.GetCol()
        r = event.GetRow()

        if self.gridbox and c >= 0: 
            self.gridbox.ColumnSelect(c)
            self.selectcol = c

        if self.gridbox and r >= 0: 
            self.gridbox.RowSelect(r)
            self.selectrow = r

        event.Skip()


    def OnLeftClick(self, event):
        pos = event.GetPosition()
        row = event.GetRow()
        col = event.GetCol()

        self.selectrow = row
        self.selectcol = col

        #if(diagbox) diagbox->Write(text.Format("grid click row %d col %d\n", row, col));
        event.Skip()


    def OnRightClick(self, event):
        pos = event.GetPosition()
        self.PopupMenu(self.rightmenu, pos.x - 20, pos.y)


    def CopyUndo(self):
        if self.GetNumberRows() > self.undogrid.GetNumberRows(): self.undogrid.AppendRows(self.GetNumberRows() - self.undogrid.GetNumberRows())
        if self.GetNumberCols() > self.undogrid.GetNumberCols(): self.undogrid.AppendCols(self.GetNumberCols() - self.undogrid.GetNumberCols())

        for x in range(self.GetNumberCols()):
            for y in range(self.GetNumberRows()):
                data = self.GetCellValue(y, x)
                self.undogrid.SetValue(y, x, data)


    def Undo(self):
        if self.GetNumberRows() > self.undogrid.GetNumberRows(): self.undogrid.AppendRows(self.GetNumberRows() - self.undogrid.GetNumberRows())
        if self.GetNumberCols() > self.undogrid.GetNumberCols(): self.undogrid.AppendCols(self.GetNumberCols() - self.undogrid.GetNumberCols())

        for x in range(self.GetNumberCols()):
            for y in range(self.GetNumberRows()):
                data = self.undogrid.GetValue(y, x)
                self.SetCellValue(y, x, data)
    

    def Copy(self):
        # code adapted from jkueng, https://stackoverflow.com/questions/28509629/work-with-ctrl-c-and-ctrl-v-to-copy-and-paste-into-a-wx-grid-in-wxpython
        # Get number of copy rows and cols
        if list(self.GetSelectionBlockTopLeft()) == []:
            rowstart = self.GetGridCursorRow()
            colstart = self.GetGridCursorCol()
            rowend = rowstart
            colend = colstart
        else:
            rowstart = self.GetSelectionBlockTopLeft()[0][0]
            colstart = self.GetSelectionBlockTopLeft()[0][1]
            rowend = self.GetSelectionBlockBottomRight()[0][0]
            colend = self.GetSelectionBlockBottomRight()[0][1]

        numrows = rowend - rowstart + 1
        numcols = colend - colstart + 1

        #DiagWrite(f"TextGrid Copy() start r{rowstart} c{colstart} {numrows} rows  {numcols} cols\n")

        # initialise data string
        data = ""

        # For each cell in selected range append the cell value
        # to the data string, '\t' to separate cols and '\n' to separate rows
        for row in range(numrows):
            for col in range(numcols):
                data += self.GetCellValue(rowstart + row, colstart + col)
                if col < numcols - 1: data += '\t'
            if row < numrows - 1: data += '\n'

        # Create text data object
        clipboard = wx.TextDataObject()
        clipboard.SetText(data)

        # Put the data on the clipboard
        if wx.TheClipboard.Open():
            wx.TheClipboard.SetData(clipboard)
            wx.TheClipboard.Close()
        else:
            wx.MessageBox("Can't open the clipboard", "Error")


        #print(f"linesep {repr(os.linesep)}")
     

    def Paste(self, mode = 0):
        # grid paste code from wxwidgets forum

        self.CopyUndo()

        if self.vdu and mode == 1: self.vdu.AppendText("Transpose Pasting...\n")
        if self.vdu: self.vdu.AppendText("Copy clipboard...")
        
        #if not wx.TheClipboard.Open():
        #    wx.MessageBox("Can't open the clipboard", "Warning")
        #    return False

        #clipboard = wx.TextDataObject()
        #wx.TheClipboard.GetData(clipboard)

        
        #buf = cStringIO.StringIO()
        #buf = wx.py.buffer.Buffer()
        #richclip = wx.richtext.RichTextBufferDataObject()
        #wx.TheClipboard.GetData(richclip)
        #richdata = wx.richtext.RichTextBuffer()
        #s = bytearray(10000)
        #richdata = memoryview(s)
        #check = richclip.GetDataHere(wx.DataFormat(wx.DF_UNICODETEXT), richdata)
        #print(f"RichData {check} {richdata}")

        #wx.TheClipboard.Close()
        #data = clipboard.GetText()
        data = pyperclip.paste()
        if data[-1] == "\n":
            data = data[:-1]

        #DiagWrite(f"TextGrid Paste() data: {clipboard.GetFormat()} end\n")
        #if clipboard.GetFormat() == wx.DF_TEXT: print("Clipboard wxDF_TEXT")
        #elif clipboard.GetFormat() == wx.DF_UNICODETEXT: print("Clipboard wxDF_UNICODETEXT")
        #else: print("Clipboard UNKNOWN")

        #s = pyperclip.paste()
        #print(s)
        
        rowstart = self.GetGridCursorRow()
        colstart = self.GetGridCursorCol()
        numrows = self.GetNumberRows()
        numcols = self.GetNumberCols()

        #print(repr(data))

        if mode == 2: rowstring = "\n\n"
        else: rowstring = "\n"

        newrows = len(data.split(rowstring))
        if newrows > numrows: self.AppendRows(newrows - numrows)
        
        if mode == 0 or mode == 2:
            #print(f"Paste() add rows {numrows} {newrows}")
            #DiagWrite(f"Paste() add rows {rowmax} {newrows}")
            
            for row, line in enumerate(data.split(rowstring)):
                target_row = rowstart + row
                #if target_row > rowmax:
                #    break

                for col, value in enumerate(line.split('\t')):
                    target_col = colstart + col
                    #if target_col > colmax:
                    #    break

                    #if mode == 0 or mode == 2:
                    self.SetCell(target_row, target_col, value.strip())
                    #if mode == 1:        

        if self.vdu: self.vdu.AppendText("OK\n")
    


class GridBox(ParamBox):
    def __init__(self, mod, title, pos, size, rows=100, cols=20, bookmode=True, vdumode=True):      
        ParamBox.__init__(self, mod, title, pos, size, "gridbox", 0, 1)
            
        self.mod = mod
        self.numgrids = 0
        #self.textgrid = []
    
        self.undomode = True
        #self.redtag = ""
        self.gridrows = rows
        self.gridcols = cols
        self.bookmode = bookmode
        self.vdumode = vdumode

        #self.startshift = False   # True
        self.notebook = None
        self.vdu = None
        self.gauge = None
        self.plotbox = None

        #textdata.resize(1000);
        #textdatagrid.grow = 10;

        #numdata.resize(10000);
        #numdatagrid.grow = 100;

        self.grids = {}
        #self.grids["Data"] = None
        #self.grids["Output"] = None
        #self.grids["Params"] = None
        #self.grids["Layout"] = None

        self.gridtags = []
        self.gridindex = {}

        vdubox = wx.BoxSizer(wx.VERTICAL)

        if vdumode:
            self.vdu = wx.TextCtrl(self.panel, wx.ID_ANY, "", wx.DefaultPosition, wx.Size(-1, 50), wx.BORDER_RAISED|wx.TE_MULTILINE)
            self.vdu.SetFont(self.confont)
            self.vdu.SetForegroundColour(wx.Colour(0,255,0)) # set text color
            self.vdu.SetBackgroundColour(wx.Colour(0,0,0)) # set text back color
            self.gauge = wx.Gauge(self.panel, wx.ID_ANY, 20)
            #vdubox.Add(self.vdu, 1, wx.EXPAND)

        if bookmode:
            self.notebook = wx.Notebook(self.panel, -1, wx.Point(-1,-1), wx.Size(-1, 400), wx.NB_TOP)
            self.AddGrid("Data", wx.Size(self.gridrows, self.gridcols))
            self.AddGrid("Output", wx.Size(self.gridrows, self.gridcols))
            #self.AddGrid("Params", wx.Size(20, 20))
            #self.AddGrid("Layout", wx.Size(20, 20))
        else: self.AddGrid("", wx.Size(self.gridrows, self.gridcols))

        self.currgrid = "Data"

        controlbox = wx.BoxSizer(wx.HORIZONTAL)
        storebox = self.StoreBox()

        self.buttonbox = wx.BoxSizer(wx.HORIZONTAL)
        self.buttonbox.Add(storebox, 0, wx.ALIGN_CENTRE_HORIZONTAL|wx.ALIGN_CENTRE_VERTICAL)
        self.AddButton(ID_Undo, "Undo", 40, self.buttonbox)

        leftbox = wx.BoxSizer(wx.VERTICAL)
        leftbox.Add(self.buttonbox, 0) 
        if vdumode: leftbox.Add(self.gauge, 0, wx.EXPAND)

        controlbox.AddSpacer(10)
        controlbox.Add(leftbox, 1, wx.ALIGN_CENTRE_HORIZONTAL|wx.ALIGN_CENTRE_VERTICAL)
        controlbox.AddSpacer(10)
        if vdumode: controlbox.Add(self.vdu, 10, wx.EXPAND)
        controlbox.AddSpacer(10)

        if bookmode: self.mainbox.Add(self.notebook, 1, wx.EXPAND)
        else: self.mainbox.Add(self.textgrid[0], 1, wx.EXPAND)
        self.mainbox.Add(controlbox, 0)
        self.mainbox.AddSpacer(2)

        self.panel.Layout()

        self.Bind(wx.EVT_BUTTON, self.OnUndo, ID_Undo)
        self.Bind(wx.EVT_BUTTON, self.OnCopy, ID_Copy)
        #self.Bind(wx.EVT_RIGHT_DOWN, self.OnRightClick)
        #self.Bind(wx.grid.EVT_GRID_CELL_CHANGED, self.OnCellChange)
        self.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.OnGridSelect)


    def OnGridSelect(self, event):
        newpageindex = event.GetSelection()
        self.currgrid = self.gridtags[newpageindex]
        DiagWrite(f"OnGridSelect {newpageindex} {self.currgrid}\n")


    # AddGrid() in (now default) notebook mode adds a new TextGrid and wxNotebook page
    # initialises grid and links to output controls

    def AddGrid(self, label, size):
        # Initialise
        if self.notebook: 
            newgrid = TextGrid(self.notebook, size)
            self.notebook.AddPage(newgrid, label)
        else: newgrid = TextGrid(self.panel, size)

        self.grids[label] = newgrid
        self.gridtags.append(label)     # store grid tags by page index, for page selection
        self.gridindex[label] = len(self.gridtags) - 1   # store grid index, for page selection
        newgrid.tag = label
          
        # Set Links
        newgrid.vdu = self.vdu
        newgrid.gauge = self.gauge
        newgrid.gridbox = self

        # Format
        newgrid.SetDefaultRowSize(20, True)
        newgrid.SetDefaultColSize(60, True)
        newgrid.SetRowLabelSize(50)


    def NeuroButton(self):
        if GetSystem() == "Mac": buttwidth = 45
        else: buttwidth = 40
        self.buttonbox.AddSpacer(2)
        self.AddButton(ID_Neuron, "Neuro", buttwidth, self.buttonbox)
        self.Bind(wx.EVT_BUTTON, self.OnNeuroScan, ID_Neuron)


    def OnUndo(self, event):
        self.grids[self.currgrid].Undo()


    def OnCopy(self, event):
        self.grids[self.currgrid].Copy()


    def OnStore(self, event):
        #print("GridStore")
        self.GridStore()


    def OnLoad(self, event):
        self.GridLoad()


    def GridStore(self):

        storeversion = 2     # first version under Python is v2, v1 is indexed grid format in C++

        gridpath = self.mod.path + "/Grids"
        if os.path.exists(gridpath) == False: 
            os.mkdir(gridpath)

        filetag = self.storetag.GetValue()
        if filetag == "": 
            print("GridStore() Bad file name")
            return

        # Grid data file
        filepath = gridpath + "/" + filetag + "-grid.txt"

        # Graph file history
        tagpos = self.storetag.FindString(filetag)
        if tagpos != wx.NOT_FOUND: self.storetag.Delete(tagpos)
        self.storetag.Insert(filetag, 0)

         # Overwrite Warning
        outfile = TextFile(filepath)
        if outfile.Exists() and self.redtag != filetag: 
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
        outfile.Open('w')
        outtext = []
    
        self.WriteVDU("Writing file...")

        outtext.append(f"gsv {storeversion}\n")
        #outtext.append(f"num {len(self.grids)}\n")    # number of grids not used any more

        for tag in self.grids:
            outtext.append(f"g {tag} r {self.grids[tag].GetNumberRows()} c {self.grids[tag].GetNumberCols()}\n")

        for tag in self.grids:
            for row in range(self.grids[tag].GetNumberRows()):
                if self.gauge: self.gauge.SetValue(int(100 * (row + 1) / self.grids[tag].GetNumberRows()))
                for col in range(self.grids[tag].GetNumberCols()):
                    celltext = self.grids[tag].GetCellValue(row, col)
                    celltext = celltext.strip()
                    if not celltext == "": 
                        outtext.append(f"{tag} {row} {col} {celltext}\n")

        outtext.append("\n")
        outfile.WriteLines(outtext)
        outfile.Close()
        if self.gauge: self.gauge.SetValue(0)
        self.WriteVDU("OK\n")

        # Grid size file
        filepath = gridpath + "/" + filetag + "-gridsize.txt"
        outfile = TextFile(filepath)
        outfile.Open('w')

        for tag in self.grids:
            for col in range(self.grids[tag].GetNumberCols()):
                outfile.WriteLine(f"grid {tag} col {col} {self.grids[tag].GetColSize(col)}")
    
        outfile.Close()


    def GridLoad(self, tag = ""):
        diag = False

        # File path
        gridpath = self.mod.path + "/Grids"
        if tag == "": filetag = self.storetag.GetValue()
        else: filetag = tag

        # Grid data file
        filepath = gridpath + "/" + filetag + "-grid.txt"
        infile = TextFile(filepath)
        if not infile.Exists():
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
        if not infile.Open('r'): 
            DiagWrite("GridLoad file error\n")
            return
        filetext = infile.ReadLines()
        linecount = 0
        numlines = len(filetext)

        self.WriteVDU("Reading grid data...")

        for readline in filetext:
            if readline.strip() == "": break
            readdata = readline.split()
            # read file version
            if readdata[0] == "gsv": storeversion = int(readdata[1])
            # read and set grids
            elif readdata[0] == "g":    
                tag = readdata[1]
                numrows = int(readdata[3])
                numcols = int(readdata[5])
                if not tag in self.grids: self.AddGrid(tag, wx.Size(numrows, numcols))
                else: 
                    if numrows > self.grids[tag].GetNumberRows(): self.grids[tag].AppendRows(numrows - self.grids[tag].GetNumberRows())
                    if numcols > self.grids[tag].GetNumberCols(): self.grids[tag].AppendCols(numcols - self.grids[tag].GetNumberCols())
                    self.grids[tag].ClearGrid()
                    #print(f"GridLoad {tag} setsize {numrows} {numcols}")
            # read grid cells
            else:   
                if diag: DiagWrite(readline)
                tag = readdata[0]
                row = int(readdata[1])
                col = int(readdata[2])
                celldata = readdata[3].strip()
                #print(f"Grid {tag} {self.grids[tag].GetNumberRows()} {self.grids[tag].GetNumberCols()}")
                self.grids[tag].SetCellValue(row, col, celldata)
                if diag: DiagWrite(f"row {row} col {col} data: {celldata} end\n")
            linecount += 1
            if self.gauge: self.gauge.SetValue(int(100 * linecount / numlines))

        self.WriteVDU("OK\n")
        if self.gauge: self.gauge.SetValue(0)
        infile.Close()

        # Read column sizes file
        filepath = gridpath + "/" + filetag + "-gridsize.txt"
        infile = TextFile(filepath)
        infile.Open('r')

        if not infile.Open('r'): 
            DiagWrite("GridLoad file error\n")
            return
        filetext = infile.ReadLines()

        for readline in filetext:
            readdata = readline.split()
            tag = readdata[1]
            col = int(readdata[3])
            width = int(readdata[4])
            self.grids[tag].SetColSize(col, width)

        infile.Close()


    def ColumnSelect(self, col):
        self.WriteVDU(f"Column Select {col}\n")
        if self.mod: self.mod.GridColumn(col)
        if self.plotbox: self.plotbox.SetColumn(col)


    def RowSelect(self, row):
        self.WriteVDU(f"Row Select {row}\n")
        if self.mod: self.mod.GridRow(row)


    def OnNeuroScan(self, event):
        self.NeuroScan()


    def NeuroScan(self):
        self.WriteVDU("Neural data scan...")
        #params = self.neurobox.GetParams()
        #filterthresh = params["filterthresh"]
        filterthresh = 10
        timescale = 1000    # 1000 to convert seconds to millseconds
        grid = self.grids[self.currgrid]
        spikedata = self.mod.spikedata
        cellpanel = self.mod.spikedatabox.cellpanel

        
        cellcount = 0
        col = 0
        startthresh = 10000 / timescale
        startshift = True

        celltext = grid.GetCell(0, col)
        celltext.strip()

        while not celltext == "":
            #if(celldata->size() <= cellcount) celldata->resize(cellcount + 10);
            #diagbox->Write(text.Format("cellcount %d  cell data size %d\n", cellcount, (int)mod->celldata.size()));
            spikedata.append(NeuroDat())
            spikedata[cellcount].name = celltext
            spikedata[cellcount].filter = 0
            spikecount = 0
            spikestart = 0
            row = 1

            celltext = grid.GetCell(row, col)
            celltext.strip()
            
            # Skip non-spike time rows
            while not isfloat(celltext):
                # diagbox->Write(text.Format("col %d row %d %s\n", col, row, celltext));
                row += 1
                celltext = grid.GetCell(row, col);
                celltext.strip()

            # Read and filter spike time data
            while not celltext == "":
                # DiagWrite(f"neuroscan celltext {celltext}\n")
                cellval = float(celltext)
                if startshift and spikecount == 0 and cellval > startthresh: 
                    spikestart = math.floor(cellval);   # shift spike times when there's a long initial silent period
                spiketime = (cellval - spikestart) * timescale
                #DiagWrite(f"neuroscan spike {spikecount}  cellval {cellval}  spiketime {spiketime}\n")
                if spikecount > 0:    
                    spikeint = spiketime - spikedata[cellcount].times[spikecount - 1]
                    # if(spikecount < 10) diagbox->Write(text.Format("col %d spikeint %.2f filter %d\n", col, spikeint, filterthresh));
                if spikecount >= spikedata[cellcount].maxspikes: spikedata[cellcount].SetSize(spikecount + 10000)
                if spikecount == 0 or spikeint > filterthresh:
                    try:
                        spikedata[cellcount].times[spikecount] = spiketime
                    except Exception:
                        DiagWrite(f"NeuroScan spikecount {spikecount} spikedata max {spikedata[cellcount].maxspikes} array size {spikedata[cellcount].times.size}\n")
                    spikecount += 1
                #else: DiagWrite(f"neuroscan filter interval\n")
                row += 1 
                celltext = grid.GetCell(row, col)
                celltext.strip()

            # Record spike count and initialise next column
            spikedata[cellcount].spikecount = spikecount
            spikedata[cellcount].gridcol = col
            spikedata[cellcount].index = cellcount
            cellcount += 1
            col += 1
            celltext = grid.GetCell(0, col)
            celltext.strip()

        if cellcount == 0:
            DiagWrite("Neuro scan: NO DATA\n")
            #mod->cellbox->datneuron->SetLabel("NO DATA");
        else:
            DiagWrite(f"Neuro scan: {cellcount} cells read OK\n")
            cellpanel.SetDataCount(cellcount)

        # multi cell analysis
        # neurobox->cellpanel->MultiCellAnalysis();

        #DiagWrite("Neuro data....")
        # neurobox->cellpanel->NeuroData()
        self.mod.NeuroData()
        #DiagWrite("OK\n")

        self.WriteVDU("OK\n")

