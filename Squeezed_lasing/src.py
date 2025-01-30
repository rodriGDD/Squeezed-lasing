'''

This module includes classes with functions which are tailored for simulating and visualizing squeezed lasing systems, particularly interesting in quantum optics. 
Below we present a summary of each class and its main functionalities.

Author: Rodrigo Grande de Diego
Date: 10/11/2024
Github repository:

Classes:

    - squeezed_lasing: this class models a laser system that includes squeezing effects in its dynamic evolution. It enables the calculation of steady states and related properties such as the second-order correlation function, emission spectrum, photon distributions, and more.

        Funtions:

            - def_H: defines the Hamiltonian of the system in the squeezed basis. It considers a coupling between a photonic cavity and a one-atom subsystem.

            - def_Lops: defines the system's collapse (Lindblad) operators, one that represents incoherent pumping and other that describes photon loss.

            - calc_rho_sss: calculates the steady-state density matrix in the squeezed basis using the Hamiltonian (def_H) and collapse operators (def_Lops).

            - calc_rho_bss: converts the steady state from the squeezed basis to the bare basis (non-squeezed basis) by applying a unitary tranformation given by the squeezing operator.
            
            - no_squeezed_ev: calculates the expectation value of a normally ordered combination of annihilation and creation operators of the squeezed basis.

            - no_bare_ev: calculates the expectation value of a normally ordered combination of annihilation and creation operators of the bare basis.

            - corr_2: calculates the second-order correlation function for a range of delay times. It uses the defined Hamiltonian and collapse operators and normalizes the result with the square of the expected photon number.
            
            - emis_spec: calculates the emission spectrum of the system by obtaining the first-order correlation function and applying a Fourier transform to yield the frequency spectrum.
            
            - phot_dist: calculates the photon distribution of a state.
            
            - quad_unc: calculates the quadrature fluctuations of a state based on a chosen quadrature angle.
            
            - Liouv_eigval: calculates the Liouvillian eigenvalues for the system.

    - wigner_rep: this class is designed for calculating and visualizing the Wigner function of a quantum system's density matrix, offering static plots and animated visualizations to depict quantum states in phase space. The class primarily works with states from a separate squeezed_lasing class.

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

class squeezed_lasing:
    '''
    This class models a laser system that includes squeezing effects in its dynamic evolution. It enables the calculation of the steady state and related properties
    such as the second-order correlation function, emission spectrum, photon distributions, and more.

    '''
    def __init__(self,delta_s=None,delta_sigma=None,g_0=None,P=None,k=None,N=300,theta=np.pi,r=0,eta=0,omega=0,phi=0):
        '''
        Initializes the squeezed_lasing class, mainly the parameters and operators necessary for describing the squeezed lasing system.

        Parameters:
        -----------
            delta_s (float): Frequency detuning of the squeezed field mode.
            delta_sigma (float): Frequency detuning of the two-level system.
            g_0 (float): Coupling strength between the field mode and the two-level system.
            P (float): Pumping rate for the lasing system.
            k (float): Decay rate of the field mode.
            N (int, default = 300): Dimension of the Hilbert space for the field mode.
            theta (float, default = np.pi): Phase angle associated with squeezing.
            r (float, default = 0): Squeezing parameter, with larger values indicating more squeezing.
            eta (float, default = 0): Photon loss through secondary channels (e.g. intracavity losses)9.
            omega (float, default = 0): Coherent drive strength for the field mode.
            phi (float, default = 0): Well-defined phase of the driving term.
        '''
        self.a_s = tensor(destroy(N),identity(2)) # Annihilation operator of the squeezed basis
        self.a = self.a_s*np.cosh(r)-np.exp(-1j*theta)*self.a_s.dag()*np.sinh(r) # Annihilation operator of the bare basis
        self.sigma = tensor(identity(N),basis(2,0)*basis(2,1).dag()) # Lowering operator of the two-level system
        self.delta_s = delta_s # System parameters are saved to self to be used in subsequent calculations
        self.delta_sigma = delta_sigma
        self.g_0 = g_0
        self.P = P
        self.k = k
        self.theta = theta
        self.r = r
        self.eta = eta
        self.omega = omega
        self.phi = phi

    def def_H(self):
        '''
        Constructs and returns the Hamiltonian of the system in the squeezed basis. It considers a coupling between a photonic cavity and a one-atom subsystem as well as a possible coherent drive.

        Parameters:
        -----------
            This method does not take any additional parameters, but uses attributes defined during initialization.

        Returns:
        -----------
            Hamiltonian (Qobj): a complex operator that describes the dynamics of the squeezed lasing system.
        '''
        a_s = self.a_s; a = self.a; sigma = self.sigma; delta_s = self.delta_s; delta_sigma = self.delta_sigma; g_0 = self.g_0; r = self.r; theta = self.theta; omega = self.omega; phi = self.phi;

        return delta_s*a*a.dag()+delta_sigma*sigma.dag()*sigma+g_0*np.cosh(r)*(a_s.dag()*sigma+a_s*sigma.dag())+omega*(a_s*np.exp(1j*phi)+a_s.dag()*np.exp(-1j*phi))
        

    def def_Lops(self):
        '''
        It defines the system's collapse (Lindblad) operators, one that represents incoherent pumping and other that describes photon loss.

        Parameters:
        -----------
            This method does not take any additional parameters but uses attributes defined during initialization.

        Returns:
        -----------
            Collapse operators (two-element list): two collapse pperators are returned in a list, each representing a distinct dissipative process (incoherent pumping and photon decay).
        '''
        a_s = self.a_s; sigma = self.sigma; P = self.P; k = self.k; eta = self.eta;
       
        return [np.sqrt(P)*sigma.dag(),np.sqrt(k*(1+eta))*a_s]

    def calc_rho_sss(self):
        '''
        It calculates and returns the steady-state density matrix in the squeezed basis using the Hamiltonian (def_H) and collapse operators (def_Lops).

        Parameters:
        -----------
            This method does not take any additional parameters but uses attributes defined during initialization.

        Returns:
        -----------
            Steady-State density matrix (Qobj): a density matrix representing the steady-state of the squeezed lasing system.
        '''
        delta_s = self.delta_s; delta_sigma = self.delta_sigma; g_0 = self.g_0; P = self.P; k = self.k; r = self.r; eta = self.eta; omega = self.omega; phi = self.phi;

        H = self.def_H()
        Lops = self.def_Lops()

        return steadystate(H,Lops) 
    
    def calc_rho_bss(self,rho=None):
        '''
        Converts the steady state from the squeezed basis to the bare basis (non-squeezed basis) by applying a unitary tranformation given by the squeezing operator.

        Parameters:
        -----------
            rho (Qobj, optional): if provided, it will be transformed to the bare basis. If not provided, the function calculates a steady-state density matrix with calc_rho_sss and the parameters given in initialization and then applies the squeezing transformation.

        Returns:
        -----------
            Transformed density matrix in bare basis (Qobj): either of the matrix given as input or of the matrix calculated with the initialization parameters.
        '''
        a_s = self.a_s; delta_s = self.delta_s; delta_sigma = self.delta_sigma; g_0 = self.g_0; P = self.P; k = self.k; theta = self.theta; r = self.r; eta = self.eta; omega = self.omega; phi = self.phi;
        self.chi = r*np.exp(-1j*theta) 
        self.S = (0.5*(self.chi*a_s*a_s-np.conj(self.chi)*a_s.dag()*a_s.dag())).expm() # Squeezing operator

        if rho is None:
            return self.S*calc_rho_sss()*self.S.dag()
        else:
            return self.S*rho*self.S.dag()
    
    def no_squeezed_ev(self,rho,n_dag,n):
        '''
        Calculates the expectation value of a normally ordered combination of annihilation and creation operators of the squeezed basis.

        Parameters:
        -----------
            rho (Qobj): density matrix for which the expectation value is calculated.
            n_dag (int): power of the creation operator.
            n (int): power of the annihilation operator.

        Returns:
        -----------
            Expectation Value (float): expected value of the normally ordered combination of annihilation and creation operators of the squeezed basis.
        '''
        a_s = self.a_s;

        return expect(a_s.dag()**n_dag*a_s**n,rho)
    
    def no_bare_ev(self,rho,n_dag,n):
        '''
        Calculates the expectation value of a normally ordered combination of annihilation and creation operators of the bare basis.

        Parameters:
        -----------
            rho (Qobj): density matrix for which the expectation value is calculated.
            n_dag (int): power of the creation operator.
            n (int): power of the annihilation operator.

        Returns:
        -----------
            Expectation Value (float): expected value of the normally ordered combination of annihilation and creation operators of the bare basis.
        '''
        a = self.a;

        return expect(a.dag()**n_dag*a**n,rho)
    
    def corr_2(self,rho,tau_f=25):
        '''
        Calculates the second-order correlation function for a range of delay times. It uses the defined Hamiltonian and collapse operators and normalizes the result with the square of the expected photon number.

        Parameters:
        -----------
            rho (Qobj): density matrix for which the correlation is calculated.
            tau_f (default = 25): maximum time delay in units of 1/k for which correlations are calculated.
            n (int): power of the annihilation operator.

        Returns:
        -----------
            Vector of time delays (np.array): array representing time delays over which correlations are evaluated.
            Normalized second-order correlation function (np.array): representing photon correlation at different time delays.
        '''
        a = self.a; k = self.k;

        self.tau_vec = np.linspace(0,tau_f/k,10000) # Time vector of delays
        H = self.def_H()
        Lops = self.def_Lops()
        
        self.corr2 = correlation_3op_1t(H, None, self.tau_vec, Lops, a.dag(), a.dag()*a, a)

        return self.tau_vec, self.corr2/(self.no_bare_ev(rho,1,1))**2
    
    def emis_spec(self,tau_f=25):
        '''
        Calculates the emission spectrum of the system by obtaining the first-order correlation function and applying a Fourier transform to yield the frequency spectrum.

        Parameters:
        -----------
            tau_f (default = 25): maximum time in units of 1/k for which the emission spectrum is calculated.
            n (int): power of the annihilation operator.

        Returns:
        -----------
            wlist (np.array): array of frequencies corresponding to the emission spectrum.
            spec (np.array): computed two-sided emission spectrum over the frequency range specified by wlist.
        '''
        a = self.a; k = self.k;

        self.tau_vec = np.linspace(0,tau_f/k,100) # Time vector
        H = self.def_H()
        Lops = self.def_Lops()
        corr = correlation_2op_1t(H, None, self.tau_vec, Lops, a.dag(), a, options=Options(nsteps=10000))
        wlist, spec = spectrum_correlation_fft(self.tau_vec,corr)

        return wlist, spec
    
    def phot_dist(self,rho):
        '''
        Calculates the photon number distribution of a state.

        Parameters:
        -----------
            rho (Qobj): density matrix for which the photon distribution is calculated.

        Returns:
        -----------
            Photon distribution (np.array): array with probabilities corresponding to each photon number state in the reduced density matrix of the field mode.
        '''
        return rho.ptrace(0).diag()
    
    def quad_unc(self,quad_ang,rho):
        '''
        Calculates the quadrature fluctuations (uncertainty) of a state based on a chosen quadrature angle.

        Parameters:
        -----------
            quad_ang (float): quadrature angle that specifies the field's quadrature to measure.
            rho (Qobj): density matrix for which the quadrature uncertainty is calculated.

        Returns:
        -----------
        Returns
            Squared quadrature uncertainty (float): variance of the chosen quadrature operator 
        '''
        a = self.a;
        X_ang = a*np.exp(-1j*quad_ang)+a.dag()*np.exp(1j*quad_ang)

        return expect(X_ang**2,rho)-expect(X_ang,rho)**2

    def Liouv_eigval(self,gap:str = None):
        '''
        Calculates the Liouvillian eigenvalues for the system. The smallest non-zero eigenvalue's real part represents the "Liouvillian gap," which is inversely proportional to the timescale of relaxation towards the steady state.

        Parameters:
        -----------
            gap (bool, optional): select 'True' when interested just in the smallest non-zero eigenvalue (Liouvillian gap).

        Returns:
        -----------
            If gap is True, the real part of the smallest non-zero eigenvalue with opposite sign is returned (float). Otherwise, an array of all eigenvalues of the Liouvillian operator is returned (np.array).

        '''
        H = self.def_H()
        Lops = self.def_Lops()
        eigs = liouvillian(H,Lops).eigenenergies()
        if gap == 'True':
            return -np.real(eigs[-2])
        else:
            return eigs

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec

class wigner_rep:

    '''
    A class designed for calculating and visualizing the Wigner function of a quantum system's density matrix, offering static plots and animated visualizations 
    to depict quantum states in phase space. The class primarily works with states from a separate squeezed_lasing class.

    '''

    def __init__(self,rhos = None,params: list = None):

        '''
        Initializes the wigner_rep class. It either takes a predefined list of density matrices or a list of system parameters to set up the class for 
        Wigner function calculation. If params are provided, it creates an instance of squeezed_lasing to compute the system's steady-state density
        matrix in the bare basis.

        Parameters:
        -----------
            rhos (Qobj or list of Qobjs, optional): density matrix or list of density matrices from which we want to obtain the wigner function.
            params (list, optional): parameters of the system, from which the density matrix of the steady state in the bare basis is obtained.
        '''
        if rhos is None:
            self.delta_s = params[0]
            self.delta_sigma = params[1]
            self.g_0 = params[2]
            self.P = params[3]
            self.k = params[4]
            self.N = params[5]
            self.theta = params[6]
            self.r = params[7]
            self.eta = params[8]
            self.omega = params[9]
            self.phi = params[10]   

            sl = squeezed_lasing(self.delta_s,self.delta_sigma,self.g_0,self.P,self.k,self.N,self.theta,self.r,self.eta,self.omega,self.phi)
            self.H = sl.def_H()
            self.Lops = sl.def_Lops()
            self.rhos = sl.calc_rho_bss()
        else:
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
            np.arrays of the x and p coordinates at which the Wigner function has been obtained.
            np.array of the values representing the Wigner function calculated over the specified range.
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
        cbar.set_label('W(x,p)',fontsize=14)
        ax.contourf(wig_xvec, wig_pvec, Wigner, 100, cmap=cm.RdBu, norm=nrm)

        plt.xlim([np.min(wig_xvec),np.max(wig_xvec)]) 
        plt.ylim([np.min(wig_pvec),np.max(wig_pvec)])
        # plt.title('Steady state Wigner function',fontsize=14)
        plt.axis('scaled')
        plt.xlabel('x',fontsize=14)
        plt.ylabel('p',fontsize=14)
        plt.show()

    def wigner_animation(self,x_lim,p_lim,moving_param: np.array,figname: str = 'Wig',speed = 100, mrname: str = 'Param'):
    
        '''
        Creates an animated gif of the Wigner function's evolution as a system parameter changes.

        Parameters:
        -----------
        x_lim (float or int): the limit for the x-axis in the phase space plot. This parameter sets the range of x values over which the Wigner function is calculated, controlling the width of the plotted region.
        p_lim (float or int): the limit for the p-axis in the phase space plot. Similar to x_lim, this defines the range of p values, affecting the height of the region shown in the plot.
        moving_param (np.array): an array containing values of a parameter that changes across the animation frames.
        figname (str, default: 'Wig'): the name used for the saved animation file. The function will save the animation as a GIF with this name, appending '_WIGNER.gif' to create the final filename.
        speed (int, default: 100): the interval speed for each frame in milliseconds. Determines the delay between frames in the GIF, effectively setting the playback speed of the animation.
        mrname (str, default: 'Param'): the name of the parameter being varied, shown in the animation title.

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
            cbar.set_label(label='$W(x,p)$',fontsize = 10)
            
            ax.contourf(wig_xvec, wig_pvec, Wigner, 100, cmap=cm.RdBu, norm=nrm)

            ax.set_xlim([np.min(wig_xvec),np.max(wig_xvec)])
            ax.set_ylim([np.min(wig_pvec),np.max(wig_pvec)])
            ax.set_title('Evolution of the Wigner function ('+mrname+' = '+str(format(moving_param[num],'.3f'))+')')
            ax.set_xlabel('x')
            ax.set_ylabel('p')
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
        cbar.set_label(label='$W(x,p)$',fontsize = 10)

        animW = animation.FuncAnimation(im, update_Wigner, range(np.ma.size(rhos,0)), fargs = (x_lim,p_lim, mrname, moving_param), interval = speed)
        animW.save(str(figname)+'_WIGNER.mp4')
        animW.save(str(figname)+'_WIGNER.gif') 