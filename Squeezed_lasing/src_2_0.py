'''

This module includes classes with functions which are tailored for simulating and visualizing results for my masters thesis on squeezed lasing systems, particularly interesting in quantum optics. 
Below I present a summary of each class and its main functionalities.

Author: Rodrigo Grande de Diego
Version: 2.0
Date: 21/12/2024
Github repository:

Classes:

    - spec_prop: This class provides tools to analyze the dynamics and spectral properties of quantum systems described by given Hamiltonian and Lindblad operators.

            - corr_2: calculates the second-order correlation function for a range of delay times evaluated in t-->inf, i.e. in the steady state.
            
            - emis_spec: calculates the emission spectrum of the system by obtaining the first-order correlation function and applying a Fourier transform to yield the frequency spectrum.
            
            - squeezed_spec: calculates the squeezing spectrum of a quantum system for a certain quadrature.
            
            - Liouv_eigval: calculates the Liouvillian eigenvalues for the system.


    - wigner_rep: this class is designed for calculating and visualizing the Wigner function of a quantum system density matrix, offering static plots and animated visualizations to depict quantum states in phase space.

        Functions:

            - wigner_calc: calculates the Wigner function for a given density matrix rho over specified symmetric limits intervals in the x and p dimensions.

            - wigner_plot: generates a static plot of the Wigner function for the system's density matrix.

            - wigner_animation: creates an animated gif of the Wigner function's evolution as a system parameter changes.
'''

from qutip import *
import numpy as np
import scipy as scipy
import os
from tqdm import tqdm
import sys

class spec_prop:
    '''
    This class provides tools to analyze quantum systems described by given Hamiltonian and Lindblad operators. It includes methods to study its dynamics 
    and spectral properties, such as correlation functions, emission spectra, squeezing spectra, or Liouvillian eigenvalues.

    '''
    def __init__(self,a,H,Lops):
        '''
        Initializes the spec_prop class.

        Parameters:
        -----------
            a (Qobj): annihilation operator of the cavity mode of the system
            H (Qobj): hamiltonian of the quantum system
            Lops (list of Qobj): list of collapse (Lindblad) operators present in the master equation that describes the dynamics of the quantum system. 
        '''
        self.a = a # Annihilation operator
        self.H = H # Hamiltonian
        self.Lops = Lops # Collapse operators
    
    def corr_2(self,rho,k,tau_f=25):
        '''
        Calculates the second-order correlation function for a range of delay times normalising the result with the square of the expected photon number.

        Parameters:
        -----------
            rho (Qobj): density matrix for which the correlation is calculated.
            k (float): scaling factor for time, representing the system's characteristic damping or evolution rate.
            tau_f (int, optional, default = 25): maximum time delay in units of 1/k for which correlations are calculated.

        Returns:
        -----------
            Vector of time delays (array): array representing time delays over which correlations are evaluated.
            Normalized second-order correlation function (array): representing photon correlation at different time delays.
        '''
        a = self.a; H = self.H; Lops = self.Lops;

        self.tau_vec = np.insert(np.logspace(np.log10(0.005/k),np.log10(tau_f/k),10000), 0, 0) # Time vector of delays
        
        self.corr2 = correlation_3op_1t(H, None, self.tau_vec, Lops, a.dag(), a.dag()*a, a, options=Options(nsteps=10000))

        return self.tau_vec, self.corr2/(expect(a.dag()*a,rho))**2
    
    def emis_spec(self,k,tau_f=100):
        '''
        Calculates the emission spectrum of the system by obtaining the first-order correlation function and applying a Fourier transform to yield the frequency spectrum.

        Parameters:
        -----------
            k (float): scaling factor for time, representing the system's characteristic damping or evolution rate.
            tau_f (int, optional, default = 100): maximum time delay in units of 1/k for which the correlator used in the calculation of the emission spectrum is computed.

        Returns:
        -----------
            wlist (array): array of frequencies corresponding to the emission spectrum.
            spec (array): computed two-sided emission spectrum over the frequency range specified by wlist.
        '''
        a = self.a; H = self.H; Lops = self.Lops;

        self.tau_vec = np.linspace(0,tau_f/k,2000) # Time vector of delays

        corr = correlation_2op_1t(H, None, self.tau_vec, Lops, a.dag(), a, options=Options(nsteps=10000))
        wlist, spec = spectrum_correlation_fft(self.tau_vec,corr)

        return wlist, spec

    def Liouv_eigval(self,gap:str = None):
        '''
        Calculates the Liouvillian eigenvalues for the system. The smallest non-zero eigenvalue's real part represents the "Liouvillian gap," which is inversely proportional to the timescale of relaxation towards the steady state.

        Parameters:
        -----------
            gap (bool, optional): select 'True' when interested just in the smallest non-zero eigenvalue (Liouvillian gap).

        Returns:
        -----------
            If gap is True, the real part of the smallest non-zero eigenvalue with opposite sign is returned (float). Otherwise, an array of all eigenvalues of the Liouvillian operator is returned (array).

        '''
        H = self.H; Lops = self.Lops;
       
        eigs = liouvillian(H,Lops).eigenenergies()
        if gap == 'True':
            return -np.real(eigs[-2])
        else:
            return eigs
        
    def squeezed_spec(self,rho_0,k,quad_ang,t_f=10,tau_f=150):
        '''
        Computes the squeezing spectrum of a quantum system for a certain quadrature. The calculation involves two-time correlation functions and the integration over
        specified frequencies to derive the spectrum.

        Parameters:
        ----------
            rho_0 (Qobj): the initial density matrix of the quantum system.
            k (float): scaling factor for time, representing the system's characteristic damping or evolution rate.
            quad_ang (float): quadrature angle (in radians) for which the squeezed spectrum is calculated.
            t_f (float, optional, default = 10): final time for the system's evolution, in units of 1/k.
            tau_f (float, optional, default = 150): maximum time delay in units of 1/k for which the correlator used in the calculation of the squeezing spectrum is computed.

        Returns:
        -------
            Array of absolute times of the system's evolution at which the spectrum is evaluated.
            Array of frequency values for the spectrum.
            2D array of the squeezing spectrum with axis corresponding to the evolution time and the frequency.

        '''

        a = self.a; H = self.H; Lops = self.Lops;

        self.tau_vec = np.linspace(0,tau_f/k,1000) # Time vector of delays
        self.t_vec = np.linspace(0,t_f/k,5) # Time vector of the evolution of the system

        corr_aa = correlation_2op_2t(H, rho_0, self.t_vec, self.tau_vec, Lops, a, a, options=Options(nsteps=10000)) # Calculation of the required correlators
        corr_adad = correlation_2op_2t(H, rho_0, self.t_vec, self.tau_vec, Lops, a.dag(), a.dag(), options=Options(nsteps=10000), reverse = 'True')
        corr_ada = correlation_2op_2t(H, rho_0, self.t_vec, self.tau_vec, Lops, a.dag(), a, options=Options(nsteps=10000))
        corr_aad = correlation_2op_2t(H, rho_0, self.t_vec, self.tau_vec, Lops, a.dag(), a, options=Options(nsteps=10000), reverse = 'True')

        mean_ad = np.zeros([np.size(self.t_vec),np.size(self.tau_vec)]) # Calculation of mean values in order to substract fluctuations from the correlator
        mean_a = np.zeros([np.size(self.t_vec),np.size(self.tau_vec)])

        for i in tqdm(range(np.size(self.t_vec))):
            no_fluc = mesolve(H, rho_0, self.t_vec[i]*np.ones(np.size(self.tau_vec))+self.tau_vec, Lops, [a,a.dag()], options=Options(nsteps=10000))
            mean_a[i,:] = no_fluc.expect[0]
            mean_ad[i,:] = no_fluc.expect[1]

        corr = np.zeros([np.size(self.t_vec),np.size(self.tau_vec)])

        for i in tqdm(range(np.size(self.t_vec))):
            corr[i,:] = np.exp(-1j*2*quad_ang)*(corr_aa[i,:]-mean_a[0,:]*mean_a[i,:])+(corr_aad[i,:]-mean_ad[0,:]*mean_a[i,:])+(corr_ada[i,:]-mean_a[0,:]*mean_ad[i,:])+np.exp(1j*2*quad_ang)*(corr_adad[i,:]-mean_ad[0,:]*mean_ad[i,:])
        
        freqs = np.sort(np.fft.fftfreq(np.size(self.tau_vec), d=1)) # Frequency vector

        spec = np.zeros([np.size(self.t_vec),np.size(freqs)])
        for i in tqdm(range(np.size(self.t_vec))):
            for j in range(np.size(freqs)):
                spec[i,j] = 2*k*scipy.integrate.simpson(np.cos(freqs[j]*self.tau_vec)*corr[i,:],x=self.tau_vec)
        return self.t_vec, freqs, spec

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['cmr10']
mpl.rcParams['mathtext.fontset'] = 'cm'  # Use matching math font
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 16
mpl.rc('axes', unicode_minus=False)

class wigner_rep:

    '''
    This class is designed for calculating and visualizing the Wigner function of a quantum system's density matrix, offering static plots and animated visualizations 
    to depict quantum states in phase space. It works with states which must have been previously calculated.

    '''

    def __init__(self,rhos):

        '''
        Initializes the wigner_rep class. It takes a predefined list of density matrices.

        Parameters:
        -----------
            rhos (Qobj or list of Qobjs): density matrix or list of density matrices from which we want to obtain the wigner function.
        '''
        self.rhos = rhos
            
    def wigner_calc(self,rho,x_lim,p_lim):

        '''
        Calculates the Wigner function for a given density matrix rho over symmetric intervals specified by x_lim and p_lim in the x and p dimensions.

        Parameters:
        -----------
            rho (Qobj): density matrix upon which the wigner funtion is calculated.
            x_lim (float or int): upper limit for the x-axis in the phase space plot. Controls the range of x values over which the Wigner function is evaluated, influencing the width of the plotted region.
            p_lim (float or int): upper limit for the p-axis in the phase space plot. Controls the range of p values, determining the height of the region shown in the plot.

        Returns:
        -----------
            Arrays of the x and p coordinates at which the Wigner function has been obtained.
            Array of the values representing the Wigner function calculated over the specified range.
        '''
        wig_xvec = np.linspace(-x_lim,x_lim,100) # Values of x in the phase space
        wig_pvec = np.linspace(-p_lim,p_lim,100) # Values of p in the phase space
        Wigner = wigner(rho.ptrace(0),wig_xvec,wig_pvec)

        return wig_xvec, wig_pvec, Wigner

    def wigner_plot(self,x_lim,p_lim):

        '''
        Generates a static plot of the Wigner function for a system's density matrix over a specified range of values in phase space.

        Parameters:
        -----------
            x_lim (float or int): upper limit for the x-axis in the phase space plot. Controls the range of x values over which the Wigner function is evaluated, influencing the width of the plotted region.
            p_lim (float or int): upper limit for the p-axis in the phase space plot. Controls the range of p values, determining the height of the region shown in the plot.

        Returns:
        -----------
            The function displays a contour plot of the Wigner function in a new window using matplotlib.
        '''
        rho = self.rhos;

        wig_xvec, wig_pvec, Wigner = self.wigner_calc(rho,x_lim,p_lim)

        if np.isnan(Wigner.max()).any()==True or np.isnan(Wigner.min()).any()==True:
            sys.exit('Warning: NaN encountered in the calculation of Wigner. Change axis limits.')
        else:
            nrm_lim = Wigner.max()

        nrm = mpl.colors.Normalize(-nrm_lim,nrm_lim)

        fig, ax = plt.subplots()
        
        cbar = fig.colorbar(cm.ScalarMappable(norm=nrm, cmap=cm.RdBu),cax = ax.inset_axes([1.05, 0.1, 0.05, 0.8]))
        cbar.set_label('$W(x,p)$',fontsize=16)
        ax.contourf(wig_xvec, wig_pvec, Wigner, 100, cmap=cm.RdBu, norm=nrm)

        plt.xlim([np.min(wig_xvec),np.max(wig_xvec)]) 
        plt.ylim([np.min(wig_pvec),np.max(wig_pvec)])
        # plt.title('Steady state Wigner function',fontsize=14)
        plt.axis('scaled')
        plt.xlabel('$x$',fontsize=16)
        plt.ylabel('$p$',fontsize=16)
        plt.show()
        fig.savefig('Wigner.png',dpi=500, bbox_inches = 'tight')

    def wigner_animation(self,x_lim,p_lim,moving_param: np.array,figname: str = 'Wig',speed = 100, mrname: str = 'Param'):
    
        '''
        Creates an animated gif of the Wigner function's evolution as a system parameter changes.

        Parameters:
        -----------
            x_lim (float or int): the limit for the x-axis in the phase space plot. This parameter sets the range of x values over which the Wigner function is calculated, controlling the width of the plotted region.
            p_lim (float or int): the limit for the p-axis in the phase space plot. Similarly to x_lim, this defines the range of p values, affecting the height of the region shown in the plot.
            moving_param (array): an array containing values of a parameter that changes across the animation frames.
            figname (str, optional, default: 'Wig'): the name used for the saved animation file. The function will save the animation as a GIF with this name, appending '_WIGNER.gif' to create the final filename.
            speed (int, optional, default: 100): the interval speed for each frame in milliseconds. Determines the delay between frames in the GIF, effectively setting the playback speed of the animation.
            mrname (str, optional, default: 'Param'): the name of the parameter being varied, shown in the animation title.

        Returns:
        -----------
            The function creates and saves an animated GIF file that shows the evolution of the Wigner function, with each frame representing the Wigner function at a different parameter value.
        '''
        rhos = self.rhos;

        def update_Wigner(num,x_lim,p_lim,mrname,moving_param):
            wig_xvec, wig_pvec, Wigner = self.wigner_calc(rhos[num],x_lim,p_lim)

            ax.cla()

            if np.isnan(Wigner.max()).any()==True or np.isnan(Wigner.min()).any()==True:
                sys.exit('Warning: NaN encountered in the calculation of Wigner. Change axis limits.')
            else:
                nrm_lim = Wigner.max()
            nrm = mpl.colors.Normalize(-nrm_lim,nrm_lim) 

            cbar = plt.colorbar(cm.ScalarMappable(norm=nrm, cmap=cm.RdBu),cax = ax.inset_axes([1.05, 0.1, 0.05, 0.8])) 
            cbar.set_label(label='$W(x,p)$',fontsize = 16)
            
            ax.contourf(wig_xvec, wig_pvec, Wigner, 100, cmap=cm.RdBu, norm=nrm)

            ax.set_xlim([np.min(wig_xvec),np.max(wig_xvec)])
            ax.set_ylim([np.min(wig_pvec),np.max(wig_pvec)])
            ax.set_title('Evolution of the Wigner function ('+mrname+' = '+str(format(moving_param[num],'.3f'))+')')
            ax.set_xlabel('$x$',fontsize=16)
            ax.set_ylabel('$p$',fontsize=16)
            ax.axis('scaled')

        im,ax = plt.subplots(figsize=(x_lim/2+1.6,p_lim/2))

        wig_xvec, wig_pvec, Wigner = self.wigner_calc(rhos[0],x_lim,p_lim)

        if np.isnan(Wigner.max()).any()==True or np.isnan(Wigner.min()).any()==True:
            sys.exit('Warning: NaN encountered in the calculation of Wigner. Change axis limits.')
        else:
            nrm_lim = Wigner.max()
        nrm = mpl.colors.Normalize(-nrm_lim,nrm_lim)
        
        ax.contourf(wig_xvec, wig_pvec, Wigner, 100, cmap=cm.RdBu, norm=nrm)

        cbar = im.colorbar(cm.ScalarMappable(norm=nrm, cmap=cm.RdBu),cax = ax.inset_axes([1.05, 0.1, 0.05, 0.8])) 
        cbar.set_label(label='$W(x,p)$',fontsize = 16)

        animW = animation.FuncAnimation(im, update_Wigner, range(np.ma.size(rhos,0)), fargs = (x_lim,p_lim, mrname, moving_param), interval = speed)
        animW.save(str(figname)+'_WIGNER.mp4')
        animW.save(str(figname)+'_WIGNER.gif') 