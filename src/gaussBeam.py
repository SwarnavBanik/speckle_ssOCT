"""
###############################################################################
Code for gaussian beam propagation
###############################################################################
Created:    Swarnav Banik  on  Apr 02, 2024
"""
import logging
import numpy as np

# %% Constants ################################################################
c = 299792458

# %% Simple Gaussian beam #####################################################
class gaussBeam:
    """
    Single Gaussian Beam
    """
    def __init__(self, w0, z0, wavelength = 1331E-9, M2 = 1):      
        self.w0           = w0            # Min. Beam Waist [m]
        self.z0           = z0            # Min Beam Waist Position [m] wrt lens of interest
        self.wavelength   = wavelength
        self.M2           = M2
        self.zR           = np.pi * self.w0**2 /self.wavelength/ self.M2
        self.divergence   = 2*self.wavelength / (self.w0 * np.pi)
        self.k            = 2*np.pi/self.wavelength

        
    def gaussBeamWaist(self, z):
        w = self.w0 * np.sqrt( 1 + ((z-self.z0)/self.zR)**2 )
        return w
    
    def gaussBeamROC(self, z):
        if z == self.z0:
            ROC = np.inf
        else:
            ROC = (z-self.z0) * (1 + ( self.zR/(z-self.z0) )**2 )       
        return ROC

        
    def spkWidth(self, z, scanSpeed):
        beamWidth = self.gaussBeamWaist(z)
        width = 2 * scanSpeed*2*np.pi*z/(2*beamWidth)
        return width

    def gaussBeamLensPropagation(self, f):
        z0_img = ( 1/f + 1/(self.z0 + self.zR**2/(self.z0+f) ) )**-1
        w0_img = self.w0 * np.sqrt( abs( (z0_img-f)/(abs(self.z0)-f) ) )
        return z0_img, w0_img
    
    def gaussBeam_q(self, z):
        q = 1j*self.zR + (z-self.z0)
        return q
    
    # def gaussBeamField(self, r, z, r0 = 0):
    #     w = self.gaussBeamWaist(z)
    #     e = self.w0/w * np.exp(-(r-r0*np.ones(np.shape(r)))**2/w**2) \
    #         * np.exp(-1j * self.k * (r-r0*np.ones(np.shape(r)))**2 / (2*self.gaussBeamROC(z)) )\
    #         * np.exp(-1j * self.k * z)
    #     return e
    
    def gaussBeamField(self, x, y, z, x0, y0):
        w = self.gaussBeamWaist(z)
        r = np.sqrt( (x - x0*np.ones(np.shape(x)))**2 + (y - y0*np.ones(np.shape(y)))**2 )
        e = self.w0/w * np.exp(-r**2/w**2)\
            * np.exp(-1j * self.k * r**2 / (2*self.gaussBeamROC(z)) )\
            * np.exp(-1j * self.k * z)\
            # * np.exp( 1j * (np.random.rand()*2*np.pi - np.pi) )
        return e
        
      
# %% Gaussian beam propagation using q and ABCD ###############################
class gaussBeamPropagation:
    """
    Single Gaussian Beam
    """
    def __init__(self, qi, ri = 0, thetai = 0, wavelength = 1331E-9, M2 = 1):  
        self.wavelength = wavelength
        self.M2         = M2      
        self.qi         = qi 
        self.zRi        = np.imag(self.qi)
        self.w0i        = np.sqrt( self.zRi * self.wavelength * self.M2 / np.pi )
        self.wi         = self.w0i * np.sqrt( 1 + (np.real(self.qi)/self.zRi)**2 )
        self.M          = np.array([[1, 0],[0, 1]]) 
    
    def propM(self):
        if not np.shape(self.M) == (2,2):
            logging.error("gaussBeam:::gaussBeamPropagation::propM: Optical element matrix need to 2X2.")
        if hasattr(self, 'qf'):
            self.qf = ( self.M[0,0]*self.qf + self.M[0,1] )/( self.M[1,0]*self.qf + self.M[1,1] ) 
        else:
            self.qf = ( self.M[0,0]*self.qi + self.M[0,1] )/( self.M[1,0]*self.qi + self.M[1,1] ) 
        self.zRf        = np.imag(self.qf)
        self.w0f        = np.sqrt( self.zRf * self.wavelength * self.M2 / np.pi )
        self.wf         = self.w0f * np.sqrt( 1 + (np.real(self.qf)/self.zRf)**2 )
    
    def actionLens(self, f):
        self.M = np.dot( np.array( [[1, 0],[-1/f, 1]] ), self.M )
        
    def actionFSProp(self, d):
        self.M = np.dot(np.array( [[1, d],[0, 1]] ), self.M )
  
# %% Ray propagation using ABCD ###############################################        
class rayPropagation:
    """
    Single Gaussian Beam
    """
    def __init__(self, ri = 0, thetai = 0):            
        self.ri         = ri 
        self.thetai     = thetai 

    def actionLens(self, f, dr = 0):
        M = np.array( [[1, 0],[-1/f, 1]] )
        if hasattr(self, 'rf') and hasattr(self, 'thetaf'):
            self.thetaf =  M[1,0]*(self.rf-dr) + M[1,1]*self.thetaf
        else:
            self.thetaf =  M[1,0]*(self.ri-dr) + M[1,1]*self.thetai
            self.rf     = self.ri
        
    def actionFSProp(self, d):
        M = np.array( [[1, d],[0, 1]] )
        if hasattr(self, 'rf') and hasattr(self, 'thetaf'):
            self.rf     =  M[0,0]*self.rf + M[0,1]*self.thetaf
        else:
            self.rf     =  M[0,0]*self.ri + M[0,1]*self.thetai
            self.thetaf =  self.thetai
            
       
    def actionPlaneMirror(self, alpha):
        if hasattr(self, 'rf') and hasattr(self, 'thetaf'):
            self.thetaf = -self.thetaf + 2 * alpha 
        else:
            self.thetaf = -self.thetai + 2 * alpha 
            self.rf     = self.ri
        
    
        

        

    