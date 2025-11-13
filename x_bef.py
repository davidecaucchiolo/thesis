import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({'font.size':12})
__labelFontSize=15


fig, (axs) = plt.subplots(1,1,figsize=(6,4))




__iCurrShape=-1
def nextShape():
    listaShapes=['o','v','^','.','p']
    global __iCurrShape
    __iCurrShape= (__iCurrShape+1)%len(listaShapes)
    return listaShapes[__iCurrShape]



hvalues=np.arange(0.05,1.55,.05) 
ih=0
h=hvalues[ih]
hvalues=[h]
sizes=[20,50,100,200]
for n in sizes:
    dist=np.arange(1,n+1,1)
    r=np.where(dist==int(n/2))[0][0]
    fid=np.load('RedFid_afm_pbc_N%d.npy'%(n),allow_pickle=True)
    plt.plot(np.linspace(0,1,n),np.sqrt(abs(1-fid[0,:])),nextShape()+'-',label="N=%d"%(n+1))

xValues=np.linspace(0,1,1000)
plt.plot(xValues,(1-(1-xValues)**.5)**.5,'k--',linewidth=1.5,label=r'Asym=$\sqrt{1 - {\sqrt{1-x}}}$')
plt.legend()
plt.xlabel(r"$x = \frac{M}{N-1}$",fontsize=__labelFontSize)
plt.ylabel(r"$\mathcal{B}_{N,x(N-1)}$",fontsize=__labelFontSize)

fig.subplots_adjust(left=.125, bottom=.16, right=.9, top=.95)

plt.show()

