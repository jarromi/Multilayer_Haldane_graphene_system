#=====================================
#
#   Bulk topological proximity effect
#    Code for diagonalizing Hamiltonian of a multilayer systems.
#    System composed of multiple AA-stacked hexagonal lattice layers.
#    Each layer described by the tight-binding model of free, spinless fermions.
#    Working in momentum space.
#
#   v2.0 -- 12.03.2020
#
#    Author: Jaromir Panas
#
#=====================================

import numpy as np
from cmath import *
from scipy import optimize
from random import random
import os

#
#    Here begins the main class of the code, PhysicalSystem, that specifies the system and methods of solving.
#

class PhysicalSystem(object):
    """
    This class contains the variables and methods for defining the physical system under consideration.
    
    Here we specify:
    - number and type of layers,
    - parameters for each layer,
    - coupling strength between layers,
    - grid in momentum space,
    - methods for diagonalizing Hamiltonian, finding direct gap, and for calculating topological invariants.
    """
    
    def __init__(self,NL,kS=None):
        """
        Default initializer of the system. Requires number of layers as input.
            NL - number of layers
            kS - object of kSpace type
        """
        assert isinstance(NL,int), "Wrong data type for number of layers.\n\n"
        assert NL>0, "At least one layer is needed.\n"
        self.NL=NL
        self.layers=[]    # List self.layers will store object of type Layer
        for i in range(NL):
            self.layers+=[Layer(),]    # Initialize with blank layers
        
        self.couplings=[]    # List self.couplings will store coupling strengths between layers
        for i in range(NL-1):
            self.couplings+=[0.,]    # Initialize with decoupled layers
        
        if kS:
            assert isinstance(kS,kSpace), "Wrong data type of momentum space.\n\n"
            self.kS=kS
        else:
            self.kS=kSpace()    # System requires some momentum space discretization. If none provided taken default.
        
    def __str__(self):
        """
        Method for printing info about system.
        """
        rep="This system has "+str(self.NL)+" layers.\n"
        rep+="The parameters for the each layers are:\n"
        for i in range(self.NL-1):
            rep+="Layer no. "+str(i)+":\t "+str(self.layers[i])
            rep+="Coupled to the next layer with strength:\t"+str(self.couplings[i])+"\n"
        rep+="Layer no. "+str(self.NL-1)+":\t "+str(self.layers[self.NL-1])
        
        return rep
    
    def update_all_layers(self,t1=0.0,t2=0.0,phi=0.0,m=0.0,t31=0.0, t32=0.0, randomly=False,sigma=0.03):
        """
        This function updates parameters of all layers with the same values.
        If option randomly=True than small Gaussian noise (with sigma mean squared error) is added to each parameter.
        """
        if randomly:
            for i in range(self.NL):
                self.layers[i].update_values( t1*(1.+np.random.randn(1)*sigma) ,t2*(1.+np.random.randn(1)*sigma) ,phi*(1.+np.random.randn(1)*sigma) ,m*(1.+np.random.randn(1)*sigma) ,t31*(1.+np.random.randn(1)*sigma), t32*(1.+np.random.randn(1)*sigma) )
        else:
            for i in range(self.NL):
                self.layers[i].update_values( t1 ,t2 ,phi ,m ,t31 ,t32 )
                
    def update_all_couplings(self,r=0.0, randomly=False,sigma=0.03):
        """
        This function updates all couplings with the same values.
        If option randomly=True than small Gaussian noise (with sigma mean squared error) is added to each parameter.
        """
        assert self.NL>1, "At least one coupling is needed!\n"
        if randomly:
            for i in range(self.NL-1):
                self.couplings[i]=r*(1.+np.random.randn(1)*sigma)
        else:
            for i in range(self.NL-1):
                self.couplings[i]=r
                
    
    def update_layer(self,LI=0,t1=0.0,t2=0.0,phi=0.0,m=0.0,t31=0.0, t32=0.0, randomly=False,sigma=0.03):
        """
        This function updates parameters of a single layer with specified values.
        If option randomly=True than small Gaussian noise (with sigma mean squared error) is added to each parameter.
        """
        if randomly:
            self.layers[IL].update_values( t1*(1.+np.random.randn(1)*sigma) ,t2*(1.+np.random.randn(1)*sigma) ,phi*(1.+np.random.randn(1)*sigma) ,m*(1.+np.random.randn(1)*sigma) ,t31*(1.+np.random.randn(1)*sigma), t32*(1.+np.random.randn(1)*sigma) )
        else:
            self.layers[IL].update_values( t1 ,t2 ,phi ,m ,t31 ,t32 )
                
    def update_coupling(self,IC=0,r=0.0, randomly=False,sigma=0.03):
        """
        This function updates a single coupling with specified value.
        If option randomly=True than small Gaussian noise (with sigma mean squared error) is added to each parameter.
        """
        assert self.NL>1, "At least one coupling is needed!\n"
        if randomly:
            self.couplings[IC]=r*(1.+np.random.randn(1)*sigma)
        else:
            self.couplings[IC]=r
            
    def reset_kSpace(self,Nx=24,Ny=24,kx0=-pi/3.,ky0=0.,kxmax=pi,kymax=2.*pi/np.sqrt(3.)):
        """
        See description of reset_kSpace in kSpace class.
        """
        self.kS.reset_kSpace(Nx,Ny,kx0,ky0,kxmax,kymax)
                
    def Ham_gen(self,kx,ky):
        """
        This method generates a Hamiltonian matrix for given point in k-space.
        By default it is a 2NLx2NL matrix, since we are working with tight-binding model on hexagonal lattice.
        """
        temp=np.zeros((self.NL*2,self.NL*2),dtype=complex)    # for storage of Hamiltonian matrix
        for i in range(self.NL):
            #Diagonal terms are purely layer specific.
            # DIAG A
            temp[2*i  ,2*i  ]=self.layers[i].H1(kx,ky) + self.layers[i].Hz(kx,ky)
            # LOWER OFF-DIAG BA
            temp[2*i+1,2*i  ]=self.layers[i].Hx(kx,ky) + 1.j*self.layers[i].Hy(kx,ky)
            # UPPER OFF-DIAG AB
            temp[2*i  ,2*i+1]=self.layers[i].Hx(kx,ky) - 1.j*self.layers[i].Hy(kx,ky)
            # DIAG B
            temp[2*i+1,2*i+1]=self.layers[i].H1(kx,ky) - self.layers[i].Hz(kx,ky)

            # Next update the couplings between the layers.
            if i<self.NL-1:
                temp[2*i  ,2*i+2]=self.couplings[i]
                temp[2*i+1,2*i+3]=self.couplings[i]
                temp[2*i+2,2*i  ]=self.couplings[i]
                temp[2*i+3,2*i+1]=self.couplings[i]

        return temp

    def init_eigdata(self,LI=0):
        """
        This method diagonalizes the Hamiltonian for each point in the Brillouin zone and stores its eigenvalues and eigenvectors as well as density matrix assuming half-filling and T=0.
        These results are needed for calculation of various topological invariants.
        Updates class specific variables.
        Returns direct gap at the K and K' points in the Brillouin zone.
        LI represents index of layer excluded in calculation of ALDM (density matrix of a subsystem consisting of NL-1 layers)
        """
        # For shorter notation we locally redefine variables.
        NL=self.NL
        Nx=self.kS.Nx
        Ny=self.kS.Ny
        kx0=self.kS.kx0
        ky0=self.kS.ky0
        dkx=self.kS.dkx
        dky=self.kS.dky
        
        self.LDM=np.zeros((NL,Nx+1,Ny+1,2,2),dtype=complex)                # Local density matrix for each layer at each point in th grid of the Brillouin zone.
        self.alleigvals=np.zeros((2*NL,Nx+1,Ny+1),dtype=complex)        # Eigenvalues at each point of the grid in the BZ. (Sorted)
        self.alleigvecs=np.zeros((2*NL,2*NL,Nx+1,Ny+1),dtype=complex)    # Eigenvectors at each point of the grid in the BZ. (Sorted with eigenvalues)
        self.ALDM=np.zeros((Nx+1,Ny+1,2*(NL-1),2*(NL-1)),dtype=complex) # Storage of density matrix of subsystem with NL-1 layers

        diagdmat=np.zeros((2*NL,2*NL))        # Density matrix in eigenspace. (Assumes half-filling and T=0.
        for i in range(NL):
            diagdmat[i,i]=1.

        # Diagonalization and sorting
        for ix in range(Nx+1):
            kx=kx0+float(ix)*dkx
            for iy in range(Ny+1):
                ky=ky0+float(iy)*dky
                tHam=self.Ham_gen(kx,ky)
                eigval,eigvec=np.linalg.eig(tHam)
                sidc=eigval.argsort()
                eigval=eigval[sidc]
                eigvec=eigvec[:,sidc]

                if eigval[NL-1]>eigval[NL]:
                    print("Error in sorting.")
                    exit()
                
                # Density matrix of the system
                dmat=np.dot( np.dot(eigvec,diagdmat) ,np.conj(eigvec.T) )
                
                # Density matrices of each layer
                for i in range(NL):
                    locdmat=dmat[2*i:2*i+2,2*i:2*i+2]
                    eigval_loc,eigvec_loc=np.linalg.eig(locdmat)
                    sidc_loc=eigval_loc.argsort()
                    eigval_loc=eigval_loc[sidc_loc]
                    eigvec_loc=eigvec_loc[:,sidc_loc]
                    self.LDM[i,ix,iy,:,:]=eigvec_locc
                
                # Remove layer LI and calculate the density matrix.
                alocdmat=np.delete(dmat,2*LI,0)                                             #remove layer LI and project the density matrix
                alocdmat=np.delete(alocdmat,2*LI,0)
                alocdmat=np.delete(alocdmat,2*LI,1)
                alocdmat=np.delete(alocdmat,2*LI,1)
                
                eigval_aloc,eigvec_aloc=np.linalg.eig(alocdmat)                             #diagonalize projected density matrix
                sidc_aloc=eigval_aloc.argsort()                                               #sort by eigenvalue
                eigval_aloc=eigval_aloc[sidc_aloc]
                eigvec_aloc=eigvec_aloc[:,sidc_aloc]
                self.ALDM[ix,iy,:,:]=eigvec_aloc

                self.alleigvecs[:,:,ix,iy]  = eigvec
                self.alleigvals[:,ix,iy]    = eigval
        
        #Check band gap in K point in BZ
        kx=2.*pi/3.
        ky=2.*pi/3./np.sqrt(3.)
        tHam=self.Ham_gen(kx,ky)
        eigval,eigvec=np.linalg.eig(tHam)
        sidc=eigval.argsort()
        eigval=eigval[sidc]
        bndgp1=(eigval[NL]-eigval[NL-1]).real
        
        #Check band gap in K' point in BZ
        kx=0.
        ky=4.*pi/3./np.sqrt(3.)
        tHam=self.Ham_gen(kx,ky)
        eigval,eigvec=np.linalg.eig(tHam)
        sidc=eigval.argsort()
        eigval=eigval[sidc]
        bndgp2=(eigval[NL]-eigval[NL-1]).real
        
        return (bndgp1,bndgp2)
    
    def find_indirect_gap(self,rpts=5):
        """
        This method allows to find an indirect gap in the model.
        The algorithm initiates a rondom point in the BZ and performs optimization algorithm to find minimal energy of the lowest empty band and highest energy of the highest empty band.
        Calculation is repeated several times to avoid geting stuck in local minimum.        
        Returns gap size and positions in the BZ of minimum and maximum.    
        """
        # First find the miniumu of the upper band.
        # Start with a random point in the BZ.
        x0up=[self.kS.kx0+random()*(self.kS.kxmax-self.kS.kx0),self.kS.ky0+random()*(self.kS.kymax-self.kS.ky0)]
        # Define functions to minimize
        fun1= lambda x: self.Ham.gen(x[0],x[1])[self.NL]
        # Optimize initial guess.
        x1up=optimize.minimize(fun1,x0up).x
        valup=fun1(x1up)
        # Reiterate to check for local minima.
        for ix in range(rpts):
            for iy in range(rpts):
                x0up=[self.kS.kx0+random()*(self.kS.kxmax-self.kS.kx0),self.kS.ky0+random()*(self.kS.kymax-self.kS.ky0)]
                xnew1up=optimize.minimize(fun1,x0up).x
                if fun1(xnew1up)<valup:
                    x1up=xnew1up
                    valp=fun1(x1up)
        # Also always check special points in the BZ
        x0up=[0.,(4.*pi/3.)/np.sqrt(3.)]
        xnew1up=optimize.minimize(fun1,x0up).x
        if fun1(xnew1up)<valup:
            x1up=xnew1up
            valup=fun1(x1up)
        x0up=[2.*pi/3.,(2.*pi/3.)/np.sqrt(3.)]
        xnew1up=optimize.minimize(fun1,x0up).x
        if fun1(xnew1up)<valup:
            x1up=xnew1up
            valup=fun1(x1up)
            
        # Repeat the same for the lower band
        x0dn=[self.kS.kx0+random()*(self.kS.kxmax-self.kS.kx0),self.kS.ky0+random()*(self.kS.kymax-self.kS.ky0)]
        # Define functions to minimize
        fun2= lambda x: -self.Ham.gen(x[0],x[1])[self.NL-1]
        # Optimize initial guess.
        x1dn=optimize.minimize(fun2,x0dn).x
        valdn=fun2(x1dn)
        # Reiterate to check for local minima.
        for ix in range(rpts):
            for iy in range(rpts):
                x0dn=[self.kS.kx0+random()*(self.kS.kxmax-self.kS.kx0),self.kS.ky0+random()*(self.kS.kymax-self.kS.ky0)]
                xnew1dn=optimize.minimize(fun2,x0dn).x
                if fun2(xnew1dn)<valdn:
                    x1dn=xnew1dn
                    valdn=fun2(x1dn)
        # Also always check special points in the BZ
        x0dn=[0.,(4.*pi/3.)/np.sqrt(3.)]
        xnew1dnoptimize.minimize(fun2,x0dn).x
        if fun2(xnew1dn)<valdn:
            x1dn=xnew1dn
            valdn=fun2(x1dn)
        x0dn=[2.*pi/3.,(2.*pi/3.)/np.sqrt(3.)]
        xnew1dn=optimize.minimize(fun2,x0dn).x
        if fun2(xnew1dn)<valdn:
            x1dn=xnew1dn
            valdn=fun2(x1dn)
            
        return valup+valdn,x1up,x1dn
        
    
    def find_direct_gap(self,rpts=5):
        """
        This method allows to find an direct gap in the model.
        The algorithm initiates a rondom point in the BZ and performs optimization algorithm to find minimal gap.
        Calculation is repeated several times to avoid geting stuck in local minimum.        
        Returns gap size and position in the BZ.
        """
        # Start with a random point in the BZ.
        x0up=[self.kS.kx0+random()*(self.kS.kxmax-self.kS.kx0),self.kS.ky0+random()*(self.kS.kymax-self.kS.ky0)]
        # Define functions to minimize
        fun1= lambda x: self.Ham.gen(x[0],x[1])[self.NL]-self.Ham.gen(x[0],x[1])[self.NL-1]
        # Optimize initial guess.
        x1up=optimize.minimize(fun1,x0up).x
        valup=fun1(x1up)
        # Reiterate to check for local minima.
        for ix in range(rpts):
            for iy in range(rpts):
                x0up=[self.kS.kx0+random()*(self.kS.kxmax-self.kS.kx0),self.kS.ky0+random()*(self.kS.kymax-self.kS.ky0)]
                xnew1up=optimize.minimize(fun1,x0up).x
                if fun1(xnew1up)<valup:
                    x1up=xnew1up
                    valp=fun1(x1up)
        # Also always check special points in the BZ
        x0up=[0.,(4.*pi/3.)/np.sqrt(3.)]
        xnew1up=optimize.minimize(fun1,x0up).x
        if fun1(xnew1up)<valup:
            x1up=xnew1up
            valup=fun1(x1up)
        x0up=[2.*pi/3.,(2.*pi/3.)/np.sqrt(3.)]
        xnew1up=optimize.minimize(fun1,x0up).x
        if fun1(xnew1up)<valup:
            x1up=xnew1up
            valup=fun1(x1up)
            
        return valup,x1up
        
    def method1(self):
        """
        This represents the first method of calculationg topological invariant.
        Calcualtes TKNN invariant, that is the Chern number of the entire system.
        Implements Fukui's method.
        Returns the Chern number and a map of estimates of Berry curvature.
        """
        cres=0.    # Variable for storing Chern number.
        # The U matrices from Fukui's method; storage...
        Ux=np.zeros((self.kS.Nx+1,self.kS.Ny+1),dtype=complex)
        Uy=np.zeros((self.kS.Nx+1,self.kS.Ny+1),dtype=complex)
        
        # ... and calculation of U matrices
        for ix in range(self.kS.Nx+1):
            for iy in range(self.kS.Ny+1):
                mat1=self.alleigvecs[:,:,ix  ,iy  ]
                if ix<self.kS.Nx:
                    mat2=self.alleigvecs[:,:,ix+1,iy  ]
                else:
                    mat2=self.alleigvecs[:,:,1   ,iy  ]
                if iy<self.kS.Ny:
                    mat3=self.alleigvecs[:,:,ix  ,iy+1]
                else:
                    mat3=self.alleigvecs[:,:,ix  ,1   ]
                Ux[ix,iy]=np.linalg.det(np.dot(np.conj(mat1.T),mat2)[:self.NL,:self.NL])
                Uy[ix,iy]=np.linalg.det(np.dot(np.conj(mat1.T),mat3)[:self.NL,:self.NL])
                    
        # Local estimates of Berry curvature; storage ...
        ftempall=np.zeros((self.kS.Nx,self.kS.Ny),complex)
        # ... and calculation
        for ix in range(self.kS.Nx):
            for iy in range(self.kS.Ny):
                ftemp=np.log(Ux[ix,iy]*Uy[ix+1,iy]/Ux[ix,iy+1]/Uy[ix,iy])
                ftempall[ix,iy]=ftemp    # ... of local Berry curvature ...
                cres+=ftemp/2./pi/1j    # ... and of Berry phase (Chern number).

        return cres.real, ftempall
        
    def method2(self):
        """
        This represents the second method of calculationg topological invariant.
        Calcualtes layer specific invariant with a method defined by Junhui Zheng.
        Also uses Fukui's method for discretization.
        Returns the invariants for each layer as a list.
        """
        cres=np.zeros(FS.NL,dtype=float)    # List of invariants
        # The U matrices from Fukui's method; storage...
        Ux_loc=np.zeros((self.kS.Nx+1,self.kS.Ny+1),dtype=complex)
        Uy_loc=np.zeros((self.kS.Nx+1,self.kS.Ny+1),dtype=complex)
        
        for il in range(self.NL):
            # ... and calculation of U matrices for each layer
            for ix in range(self.kS.Nx+1):
                for iy in range(self.kS.Ny+1):
                    mat1=self.LDM[il,ix                 ,iy                 ,:,:]
                    mat2=self.LDM[il,(ix%self.kS.Nx)+1    ,iy                 ,:,:]
                    mat3=self.LDM[il,ix                 ,(iy%self.kS.Ny)+1    ,:,:]
                    
                    Ux_loc[ix,iy]=np.dot(np.conj(mat1.T),mat2)[1,1]
                    Uy_loc[ix,iy]=np.dot(np.conj(mat1.T),mat3)[1,1]
            
            for ix in range(self.kS.Nx):
                for iy in range(self.kS.Ny):
                    ftemp=np.log(Ux_loc[ix,iy]*Uy_loc[ix+1,iy]/Ux_loc[ix,iy+1]/Uy_loc[ix,iy])
                    cres[il]+=(ftemp/2./pi/1j).real    # Layer specific topological invariant
        
        return cres
        
        
    def method3(self):
        """
        This represents the second method of calculationg topological invariant.
        Calcualtes subsystem specific invariant with a method defined by Junhui Zheng.
        However, unlike in method2 it calculates invariant of subsystem with NL-1 layer (LI-layer excluded).
        Also uses Fukui's method for discretization.
        Returns the invariant.
        """
        cres=0.
        Ux_aloc=np.zeros((self.kS.Nx+1,self.kS.Ny+1),dtype=complex)
        Uy_aloc=np.zeros((self.kS.Nx+1,self.kS.Ny+1),dtype=complex)
        for ix in range(self.kS.Nx+1):
            for iy in range(self.kS.Ny+1):
                mat1=self.ALDM[ix  ,iy,                  : , : ]
                mat2=self.ALDM[(ix%self.kS.Nx)+1, iy,   : , : ]
                mat3=self.ALDM[ix   ,(iy%self.kS.Ny)+1, : , : ]
                
                Ux_aloc[ix,iy]=np.linalg.det(np.dot(np.conj(mat1.T),mat2)[self.NL-1:,self.NL-1:])
                Uy_aloc[ix,iy]=np.linalg.det(np.dot(np.conj(mat1.T),mat3)[self.NL-1:,self.NL-1:])

        for ix in range(self.kS.Nx):
            for iy in range(self.kS.Ny):
                ftemp=np.log(Ux_aloc[ix,iy]*Uy_aloc[ix+1,iy]/Ux_aloc[ix,iy+1]/Uy_aloc[ix,iy])
                cres+=ftemp/2./pi/1j
        
        return cres.real
        #End of method3
        
#
#    Here begins support class Layer that PhysicalSystem uses.
#

class Layer(object):
    """
    The class that defines the Layer: parameters, Hamiltonian in k-space
    Parameters are:
        - t1 - nearest-neighbor hopping,
        - t2 - next-nearest-neighbor hopping,
        - phi - phase shift associated with t2,
        - m - staggered potential,
        - t31 - next-next-nearest-neighbor hopping across the hexagon,
        - t32 - next-next-nearest-neighbor hopping not across the hexagon,
    """
    
    def __init__(self,t1=0.,t2=0.,phi=0.,m=0., t31=0., t32=0.):
        """
        Class constructor. Specifies all parameters.
        """
        try:
            self.t1=t1
            self.t2=t2
            self.phi=phi
            self.m=m
            self.t31=t31
            self.t32=t32
        except:
            print("Wrong data type")
            exit()

    def update_values(self,t1,t2,phi,m,t31,t32):
        """
        Method for modifying parameter values.
        """
        try:
            self.t1=t1
            self.t2=t2
            self.phi=phi
            self.m=m
            self.t31=t31
            self.t32=t32
        except:
            print("Wrong data type")
            exit()

    def __str__(self):
        """
        Method for printing parameter values.
        """
        return str(self.t1)+"<-->t1, \t"+str(self.t2)+"<-->t2, \t"+str(self.phi)+"<-->phi, \t"+str(self.m)+"<-->m, \t"+str(self.t31)+"<-->t31, \t"+str(self.t32)+"<-->t32, \n"
    
    def Hx(kx,ky):
        """
        For a given layer calculate Hx component of the Hamiltonian in the momentum space based on the parameters.
        """
        Hxr  = -self.t1*(1.+np.cos(-3.*kx/2.+ky*np.sqrt(3.)/2.)+np.cos(-3.*kx/2.-ky*np.sqrt(3.)/2.)) # NN hopping
        Hxr += -self.t31*(2.*np.cos(-np.sqrt(3.)*ky) + np.cos(-3.*kx) )      # NNNN hopping across hexagon
        Hxr += -self.t32*( np.cos(-3.*kx/2.+np.sqrt(3.)*ky/2.) + np.cos(-3.*kx+np.sqrt(3.)*ky) + np.cos(-3.*kx/2.+3.*np.sqrt(3.)*ky/2.)\
            + np.cos(3.*kx/2.+np.sqrt(3.)*ky/2.) + np.cos(3.*kx+np.sqrt(3.)*ky) + np.cos(-3.*kx/2.-3.*np.sqrt(3.)*ky/2.) )
        return Hxr
    
    def Hy(kx,ky):
        """
        For a given layer calculate Hy component of the Hamiltonian in the momentum space based on the parameters.
        """
        Hyr  = -self.t1*(np.sin(-3.*kx/2.+ky*np.sqrt(3.)/2.)+np.sin(-3.*kx/2.-ky*np.sqrt(3.)/2.))
        Hyr += -self.t31*(np.sin(-3.*kx))
        Hyr += -self.t32*( -np.sin(-3.*kx/2.+np.sqrt(3.)*ky/2.) + np.sin(-3.*kx+np.sqrt(3.)*ky) + np.sin(-3.*kx/2.+3.*np.sqrt(3.)*ky/2.)\
            + np.sin(3.*kx/2.+np.sqrt(3.)*ky/2.) - np.sin(3.*kx+np.sqrt(3.)*ky) + np.sin(-3.*kx/2.-3.*np.sqrt(3.)*ky/2.) )
        return Hyr
    
    def Hz(kx,ky):
        """
        For a given layer calculate Hz component of the Hamiltonian in the momentum space based on the parameters.
        """
        return self.m-2.*self.t2*np.sin(self.phi)*(np.sin(3.*kx/2.+np.sqrt(3.)*ky/2.)+np.sin(-3.*kx/2.+np.sqrt(3.)*ky/2.)+np.sin(-np.sqrt(3.)*ky))
    
    def H1(kx,ky):
        """
        For a given layer calculate Hz component of the Hamiltonian in the momentum space based on the parameters.
        """
        return -2.*self.t2*np.cos(self.phi)*(np.cos(3.*kx/2.+np.sqrt(3.)*ky/2.)+np.cos(-3.*kx/2.+np.sqrt(3.)*ky/2.)+np.cos(-np.sqrt(3.)*ky))


#
#    Here begins support class kSpace that PhysicalSystem uses.
#

class kSpace(object):
    """
    This class specifies the discretization of the (quasi-) momentum space.
    """
    
    def __init__(self,Nx=24,Ny=24,kx0=-pi/3.,ky0=0.,kxmax=pi,kymax=2.*pi/np.sqrt(3.)):
        """
        Default constructor.
        It is best to use multiples of 24 for Nx and Ny, in order for grid of discretization to "hit" the K and K' points in Brillouin zone.
        Otherwise accuracy might go down close to the topological phase transitions due to the discretization of quite sharp features in the Berry curvature of the graphene.
        Here we use rectangular version of the Brillouin zone. The default borders are chosen such as to ensure coverage of the entire BZ. Hamiltonian is periodic with respect to shift by reciprocal lattice vectors.
        """
        self.Nx=Nx
        self.Ny=Ny
        self.kx0=kx0
        self.ky0=ky0
        self.kxmax=kxmax
        self.kymax=kymax
        self.dkx=(kxmax-kx0)/float(Nx)
        self.dky=(kymax-ky0)/float(Ny)
        
            
    def reset_kSpace(self,Nx,Ny,kx0=-pi/3.,ky0=0.,kxmax=pi,kymax=2.*pi/np.sqrt(3.)):
        """
        Reset discretization of the Brillouin zone.
        If changing borders, make sure that the selected surface has proper area.
        """
        self.Nx=Nx
        self.Ny=Ny
        self.kx0=kx0
        self.ky0=ky0
        self.kxmax=kxmax
        self.kymax=kymax
        self.dkx=(kxmax-kx0)/float(Nx)
        self.dky=(kymax-ky0)/float(Ny)