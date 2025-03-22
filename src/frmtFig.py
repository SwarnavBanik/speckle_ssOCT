"""
###############################################################################
Code for formatting figures
###############################################################################
Created:    Swarnav Banik  on  May 12, 2023 
"""

def setFigureSize(plt, figNo, figSize, journal = False):
    if journal:
        scaleFactor  = 2.6/figSize[0]
        figSize = [a*scaleFactor for a in figSize]
        
    fig = plt.figure(figNo,figsize = figSize, dpi = 300)
    return fig

def frmtFig(mpl, plt, FS_title = 20, FS_tickLabel = 20, FS_axisLabel = 20, journal = False):
    
    if journal:
        # Sizes ###############################################################
        FS_axisLabel = 5
        FS_tickLabel = 5
        FS_title     = 5
        linewidth    = 0.7
        framewidth   = 0.5
        gridwidth    = 0.4
        markersize   = 3
        
        # Colors ##############################################################
        clrBckg = (100/100, 100/100, 100/100)
        clrText = (0,0,0)
        clrPts1 = (90/100, 47/100, 27/100)
        clrPts2 = (0/100, 50/100, 99/100)
        clrPts3 = (74/100, 76/100, 75/100)
        clrPts4 = (31/100, 74/100, 55/100)
        clrPts  = [clrPts1, clrPts2, clrPts3, clrPts4]
        
        
    else:
        # Sizes ###############################################################
        linewidth    = 2
        framewidth   = 1.5
        gridwidth    = 1
        markersize   = 20
        
        # Colors ##################################################################
        clrBckg = (8/100, 11/100, 17.6/100)
        clrText = (1,1,1)
        clrPts1 = (90/100, 47/100, 27/100)
        clrPts2 = (0/100, 50/100, 99/100)
        clrPts3 = (74/100, 76/100, 75/100)
        clrPts4 = (31/100, 74/100, 55/100)
        clrPts  = [clrPts1, clrPts2, clrPts3, clrPts4]
      
    mpl.rcdefaults()
    # Set figure font sizes ###############################################
    mpl.rcParams['axes.labelsize']  = FS_axisLabel
    mpl.rcParams['xtick.labelsize'] = FS_tickLabel
    mpl.rcParams['ytick.labelsize'] = FS_tickLabel
    mpl.rcParams['legend.fontsize'] = FS_axisLabel
    mpl.rcParams['axes.titlesize']  = FS_title
    plt.rc('figure', titlesize = FS_title)
    plt.rcParams['text.color']      = clrText
    # Set the figure color ################################################
    mpl.rcParams['axes.facecolor']    = clrBckg
    mpl.rcParams['axes.edgecolor']    = clrText
    mpl.rcParams['axes.labelcolor']   = clrText
    mpl.rcParams['xtick.color']       = clrText
    mpl.rcParams['ytick.color']       = clrText
    mpl.rcParams['figure.facecolor']  = clrBckg
    # Set the grid ########################################################
    mpl.rcParams['grid.color']       = clrText
    mpl.rcParams['grid.alpha']       = 0.4
    mpl.rcParams['grid.linewidth']   = gridwidth
    # Set Legend properties ###############################################
    mpl.rcParams['legend.frameon']        = False
    mpl.rcParams['legend.title_fontsize'] = FS_axisLabel        
    # Set marker and line sizes ###########################################
    mpl.rcParams['lines.linewidth']  = linewidth
    mpl.rcParams['patch.linewidth']  = linewidth
    mpl.rcParams['lines.markersize'] = markersize
    # Set the border sizes ################################################
    mpl.rcParams['axes.linewidth']    = framewidth
    mpl.rcParams['xtick.major.width'] = gridwidth
    mpl.rcParams['ytick.major.width'] = gridwidth
    mpl.rcParams['xtick.minor.width'] = gridwidth/2
    mpl.rcParams['ytick.minor.width'] = gridwidth/2      
    mpl.rcParams.update(mpl.rcParams)
    
    return clrPts, mpl, plt


    