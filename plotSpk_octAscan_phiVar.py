"""
###############################################################################
Code for estimating speckle patterns for OCT applications
###############################################################################
Created:    Swarnav Banik  on  Nov 27, 2024
"""
CODENAME = 'plotSpk_octAscan_phiVar'
import sys
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.append('../')
from src.frmtFig import frmtFig 
from src.frmtFig import setFigureSize
from src.speckle import octSignal_aScan

# %% Common Functions #########################################################
def getAvgTrace(field):
    hist, bin_edges = np.histogram(field, bins = 70, density = True)
    bin_centers   = (bin_edges[:-1] + bin_edges[1:]) / 2
    averageVal    = np.mean(field)
    Sfrac         = bin_centers/averageVal
    PtimesavgS    = averageVal*hist
    PtimesavgS_theory = np.pi/2 * Sfrac * np.exp( -np.pi/4 * Sfrac**2 )
    return Sfrac, PtimesavgS, PtimesavgS_theory

def getContrast(signal):
    return np.std(signal)/np.mean(signal)

def getSNR(signal):
    return np.mean(signal)/np.std(signal)

# %% Inputs ###################################################################
wA                  = 10E-6        # beam waist [m]
z0A                 = 50E-3        # beam waist position wrt collimator [m]
zTarget             = 50E-3        # target position wrt collimator [m]
deltaZ_col2target   = 500E-6       # axial length of sample [m]
wavelength          = 1050E-9      # wavelength [m]
rxColDiameter       = 12.7E-3      # Collimator Diameter [m]
lensOffset          = [0,0]        # Lens offset wrt target [m] - descan/ pitch catch
Ngrid_target        = 30           # No. of target grid points 
Ngrid_rxCol         = 100          # No. of rxCol grid points
Nz                  = 1            # No. of z grid points
Nruns               = 1000         # No. of averaging points
journal             = True         # Figure type

# %% Initialize ###############################################################
clrPts, mpl, plt = frmtFig(mpl, plt, FS_title = 24, FS_tickLabel = 18, FS_axisLabel = 18, journal = journal)

# %% Generate a single Speckle Pattern and evaluate it's characteristics ######
start_time = time.time()
signals_b = []
signals_c = []
signals_d = []
for ii in range(Nruns):
    b = octSignal_aScan(wA, z0A, zTarget, deltaZ_col2target = deltaZ_col2target, Nz = Nz,\
                        wavelength = wavelength, Ngrid_target = Ngrid_target, phaseScale = 0.5,\
                        lensDiameter = rxColDiameter, Ngrid_rxCol = Ngrid_rxCol, lensOffset = lensOffset)
    c = octSignal_aScan(wA, z0A, zTarget, deltaZ_col2target = deltaZ_col2target, Nz = Nz,\
                        wavelength = wavelength, Ngrid_target = Ngrid_target, phaseScale = 0.7,\
                        lensDiameter = rxColDiameter, Ngrid_rxCol = Ngrid_rxCol, lensOffset = lensOffset)
    d = octSignal_aScan(wA, z0A, zTarget, deltaZ_col2target = deltaZ_col2target, Nz = Nz,\
                        wavelength = wavelength, Ngrid_target = Ngrid_target, phaseScale = 1,\
                        lensDiameter = rxColDiameter, Ngrid_rxCol = Ngrid_rxCol, lensOffset = lensOffset)
    signals_b.append( b.aScan )
    signals_c.append( c.aScan )
    signals_d.append( d.aScan )
    if ii == int(Nruns/2):   
        d.plotAscan(0, journal = journal)
        d.plotSpkPatterns_z(1, journal = journal)
end_time = time.time()
print(f"Time taken for {Nruns:d} run(s) of {Nz:d} points: {end_time-start_time:.4f} seconds")
# %% Plot the a scan data #####################################################
fig = setFigureSize(plt, 4, (12, 6), journal = journal)
gs = GridSpec(1,1)
fig.clf()
axs1 = fig.add_subplot(gs[0,0]) 
axs1.plot((b.z-b.zPos_target)*1E6, b.aScan/b.G_noSpk, '.', label = '$\delta \phi \in 0.3~[-\pi, \pi]$')
axs1.plot((c.z-c.zPos_target)*1E6, c.aScan/c.G_noSpk, '.', label = '$\delta \phi \in 0.7~[-\pi, \pi]$')
axs1.plot((d.z-d.zPos_target)*1E6, d.aScan/d.G_noSpk, '.', label = '$\delta \phi \in 1.0~[-\pi, \pi]$')
axs1.grid('major')
axs1.set_xlabel('z - z$_{\\rm obj}$ ($\\mu$m)')
axs1.set_ylabel('I$_{\\rm a}(\phi) / I_{\\rm a}(\phi=0)$')
axs1.legend()
title = 'OCT A scan Signal'
fig.suptitle(title)
axs1.set(ylim=( -0.1, 1.1 ))


# %% Plot Rayleigh Density statistics of the RX col field #####################
Sfrac_b, PtimesavgS_b, _ = getAvgTrace(signals_b)
Sfrac_c, PtimesavgS_c, _ = getAvgTrace(signals_c)
Sfrac_d, PtimesavgS_d, PtimesavgS_theory = getAvgTrace(signals_d)

fig = setFigureSize(plt, 5, (12,6), journal = journal)
gs = GridSpec(1,1)
axs1 = fig.add_subplot(gs[0,0]) 
axs1.plot(Sfrac_b , PtimesavgS_b,'.', color = clrPts[1], label = '$\delta \phi \in 0.3~[-\pi, \pi]$')
axs1.plot(Sfrac_c , PtimesavgS_c,'.', color = clrPts[2], label = '$\delta \phi \in 0.7~[-\pi, \pi]$')
axs1.plot(Sfrac_d , PtimesavgS_d,'.', color = clrPts[3], label = '$\delta \phi \in 1.0~[-\pi, \pi]$')
axs1.plot(Sfrac_d , PtimesavgS_theory,'-', color = clrPts[3], label = 'Theory')
axs1.legend()
axs1.grid('major')
axs1.set_ylabel('$\\left<S\\right>$ P(S)')
axs1.set_xlabel('S/$\\left<S\\right>$')
axs1.legend()
axs1.set(ylim=( 0, 6 ))
# axs1.set(xlim=( 0, 3 ))
plt.savefig(CODENAME+'_dist.svg') 
plt.savefig(CODENAME+'_dist.png') 

# %% Save the variables #######################################################
import pickle

with open(CODENAME+'.pkl', 'wb') as f:
    pickle.dump({'PtimesavgS_theory': PtimesavgS_theory,\
                 'Sfrac_b': Sfrac_b, 'PtimesavgS_b': PtimesavgS_b, 'signals_b': signals_b,\
                 'Sfrac_c': Sfrac_c, 'PtimesavgS_c': PtimesavgS_c, 'signals_c': signals_c,\
                 'Sfrac_d': Sfrac_d, 'PtimesavgS_d': PtimesavgS_d, 'signals_d': signals_d,\
                      }, f)



