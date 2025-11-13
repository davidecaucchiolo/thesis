import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size':12})
__labelFontSize=15


def fai(N,M):
    ra=np.zeros((2*M,2*M))
    for i in range(0,M-1):
        for j in range(0,M-1):
            ra[i,j]=1

    for i in range(M-1,2*M-2):
        for j in range(M-1,2*M-2):
            ra[i,j]=1

    for i in range(2*M-2):
        ra[i,2*M-2]=1
        ra[i,2*M-1]=1
        ra[2*M-2,i]=1
        ra[2*M-1,i]=1
    ra[2*M-2,2*M-2]=1+N-M
    ra[2*M-1,2*M-1]=1+N-M
    ra[2*M-2,2*M-1]=2
    ra[2*M-1,2*M-2]=2
    ra/=2*N
    ev=list(np.linalg.eig(ra).eigenvalues)
    ev.sort(reverse=True, key=lambda a: a*a)
    ev=ev[:4]
    ev.sort()
    return ev

M=lambda N : N/2

vals=np.array([fai(N,int(M(N))) for N in range(10,1000,11)])
vals=vals.transpose()

fig, (axs1, axs2) = plt.subplots(1,2,figsize=(10,4))

for v in vals:
    axs1.plot(list(range(10,1000,11)),v,".-")
axs1.plot([10,1000],[M(1)/(2)]*2,"--",label=r"$\frac{1}{4}$",color='grey')
axs1.set_xlabel("N",fontsize=__labelFontSize)
axs1.set_ylabel(r"Eigenvalues of $\rho_{A,\tilde{w}}$",fontsize=__labelFontSize)
axs1.set_xscale('log')
axs1.legend()


############################


M=lambda N : N/4

vals=np.array([fai(N,int(M(N))) for N in range(10,1000,11)])
vals=vals.transpose()

for v in vals:
    axs2.plot(list(range(10,1000,11)),v,".-")
axs2.plot([10,1000],[(1-M(1))/(2)]*2,"--",label=r"$\frac{3}{8}$",color='grey')
axs2.plot([10,1000],[M(1)/(2)]*2,"--",label=r"$\frac{1}{8}$",color='purple')
axs2.set_xlabel("N",fontsize=__labelFontSize)
axs2.set_xscale('log')
axs2.set_ylabel(r"Eigenvalues of $\rho_{A,\tilde{w}}$",fontsize=__labelFontSize)
axs2.legend()




axs1.set_title("(A)",loc='left')
axs2.set_title("(B)",loc='left')



fig.subplots_adjust(left=.07, bottom=.15, right=.97, top=.88, wspace=.35)


plt.show()
