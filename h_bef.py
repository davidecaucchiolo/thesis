import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({'font.size':12})
__labelFontSize=15


__iCurrShape=-1
def nextShape():
    listaShapes=['o','v','^','s','P','p']
    global __iCurrShape
    __iCurrShape= (__iCurrShape+1)%len(listaShapes)
    return listaShapes[__iCurrShape]



hvalues=np.arange(0.05,1.55,.05) 
sizes=[20,50,100,200]
for n in sizes:
    dist=np.arange(1,n+1,1)
    r=np.where(dist==int(n/2))[0][0]
    fid=np.load('RedFid_afm_pbc_N%d.npy'%(n),allow_pickle=True)
    plt.plot(hvalues,np.sqrt(abs(1-fid[:,r])),nextShape()+'-',label="N=%d"%(n+1))
plt.plot([0.05,1-.0001,1+.0001,1.5],[np.sqrt(1-1/np.sqrt(2))]*2 + [0]*2,'k--',linewidth=1.5,label=r'Asymptotic curve')
plt.legend()
plt.xlabel("h",fontsize=__labelFontSize)
plt.ylabel(r"$\mathcal{B}_{N,\frac{N-1}{2}}$",fontsize=__labelFontSize)


plt.show()

