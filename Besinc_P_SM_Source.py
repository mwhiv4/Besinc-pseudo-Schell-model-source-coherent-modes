import numpy as np 
import matplotlib.pyplot as plt
from scipy.special import j1,jn_zeros
from scipy.linalg import eig,inv
from scipy.interpolate import PchipInterpolator

def jinc(x):
    x = np.where(x==0,1.0e-20,x)
    y = 2*j1(x)/x
    return y

def Besinc_P_SM_S_CSD(I0,w0,dc,m,zeta,x1,y1,x2,y2):
    r1 = np.sqrt(x1**2+y1**2)
    r2 = np.sqrt(x2**2+y2**2)
    W = I0*(2*r1*r2/w0**2)**m*np.exp(-(r1**2+r2**2)/w0**2)*jinc(zeta*(r1-r2)/dc)
    return W

# ----- Besinc pseudo-Schell model source ----- #
I0 = 1
w0 = 0.6e-3
dc = w0/4
m = 1
zeta = jn_zeros(1,1)
# --------------------------------------------- #

# ----- CSD ----- #
L = 7*w0
ds = dc/10
N = np.ceil(L/ds)
if np.remainder(N,2) != 0:
    N = N+1
ds = L/N
r1,r2 = np.meshgrid(np.arange(N)*ds,np.arange(N)*ds,indexing='ij')
F = I0*(2*r1*r2/w0**2)**m*np.exp(-(r1**2+r2**2)/w0**2) \
    *jinc(zeta*(r1-r2)/dc)
LAM,Y = eig(r2*F,right=True);
LAM = np.diag(np.real(inv(Y)@F@Y)) 
CohModes_pchip = PchipInterpolator(r1[:,1],np.real(Y),axis=0)
Starikov_criterion = np.ceil(np.sum(LAM)**2/np.sum(LAM**2)).astype('int')
# --------------- #

# ----- Grid ----- #
M = 2*N
ds = ds/2
x,y = np.meshgrid(np.arange(-M/2,M/2)*ds,np.arange(-M/2,M/2)*ds)
rho = np.sqrt(x**2+y**2)
CohModes2D = CohModes_pchip(rho)
# ---------------- #

# ----- Coherent Modes ----- #
Wsim = np.zeros(np.shape(rho))
Ssim = np.zeros(np.shape(rho))
mm = 0
while np.abs(LAM[mm])/np.max(LAM) > 1e-3: 
    U = np.sqrt(LAM[mm])*CohModes2D[:,:,mm]
    Wsim = Wsim+np.outer(U[int(M/2),:],np.conj(U[int(M/2),:]))
    Ssim = Ssim+np.abs(U)**2
    mm = mm+1
Wthy = Besinc_P_SM_S_CSD(I0,w0,dc,m,zeta,x,0,y,0)
Sthy = Besinc_P_SM_S_CSD(I0,w0,dc,m,zeta,x,y,x,y)
# -------------------------- #

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Computer Modern'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}' 

plt.semilogy(np.arange(1,len(LAM)+1),np.abs(LAM)/np.max(LAM),'o',markerfacecolor='none')
plt.semilogy(np.arange(1,len(LAM)+1),1e-3*np.ones(np.shape(LAM)),'--k')
plt.xlabel(r'$N$')
plt.ylabel(r'$\lambda_n/\max\left(\lambda_n\right)$')
plt.xlim([0.5,50])
plt.ylim([10**(-17),5])
plt.grid()
plt.savefig('eig.pdf',bbox_inches='tight')
plt.show()

fig,axs = plt.subplots(nrows=2,ncols=3,layout='compressed',figsize=(7.5,5))

p=axs[0,0].imshow(Sthy,extent=[x[0,0]*1e3,x[0,-1]*1e3,y[0,0]*1e3,y[-1,0]*1e3])
axs[0,0].set_xlabel(r'$x$ (mm)')
axs[0,0].set_ylabel(r'$y$ (mm)')
axs[0,0].set_xlim([-1.5,1.5])
axs[0,0].set_ylim([-1.5,1.5])
plt.colorbar(p,ax=axs[0,0])
axs[0,0].set_title(r'$S^{\mathrm{thy}}$')

p=axs[0,1].imshow(Ssim,extent=[x[0,0]*1e3,x[0,-1]*1e3,y[0,0]*1e3,y[-1,0]*1e3])
axs[0,1].set_xlabel(r'$x$ (mm)')
axs[0,1].set_xlim([-1.5,1.5])
axs[0,1].set_ylim([-1.5,1.5])
plt.colorbar(p,ax=axs[0,1])
axs[0,1].set_title(r'$S^{\mathrm{sim}}$')

axs[0,2].plot(x[0,:]*1e3,Sthy[int(M/2),:],'k',label='Theory')
axs[0,2].plot(x[0,:]*1e3,Ssim[int(M/2),:],'--r',label='{} Modes'.format(mm))
axs[0,2].set_xlabel(r'$x$ (mm)')
axs[0,2].set_ylabel(r'$S\left(x,0\right)$')
axs[0,2].set_xlim([-1.5,1.5])
axs[0,2].grid()
axs[0,2].legend(fancybox=False,edgecolor='black',framealpha=1, \
    bbox_to_anchor=(0.95,0.54),bbox_transform=fig.transFigure)

p=axs[1,0].imshow(Wthy,extent=[x[0,0]*1e3,x[0,-1]*1e3,y[0,0]*1e3,y[-1,0]*1e3])
axs[1,0].set_xlabel(r'$x_1$ (mm)')
axs[1,0].set_ylabel(r'$x_2$ (mm)')
axs[1,0].set_xlim([-1.5,1.5])
axs[1,0].set_ylim([-1.5,1.5])
axs[1,0].set_title(r'$W^{\mathrm{thy}}\left(x_1,0,x_2,0\right)$')
plt.colorbar(p,ax=axs[1,0])

p=axs[1,1].imshow(Wsim,extent=[x[0,0]*1e3,x[0,-1]*1e3,y[0,0]*1e3,y[-1,0]*1e3])
axs[1,1].set_xlabel(r'$x_1$ (mm)')
axs[1,1].set_xlim([-1.5,1.5])
axs[1,1].set_ylim([-1.5,1.5])
plt.colorbar(p,ax=axs[1,1])
axs[1,1].set_title(r'$W^{\mathrm{sim}}\left(x_1,0,x_2,0\right)$')

inds = (x==-y)
axs[1,2].plot(x[inds]*1e3,Wthy[inds],'k',label='Theory')
axs[1,2].plot(x[inds]*1e3,Wsim[inds],'--r',label='{} Modes'.format(mm-1))
axs[1,2].set_xlabel(r'$x_1$ (mm)')
axs[1,2].set_ylabel(r'$W\left(x_1,0,-x_1,0\right)$')
axs[1,2].grid()
axs[1,2].set_xlim([-1.5,1.5])

fig.savefig('Besinc.pdf',bbox_inches='tight')
plt.show()
