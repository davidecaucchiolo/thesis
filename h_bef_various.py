import numpy as np
import matplotlib.pyplot as plt



plt.rcParams.update({'font.size':12})
__labelFontSize=15



def typeToDescr(t):
    if t=="afm_pbc":
        return "AFM periodic boundary conditions"
    elif t=="afm_obc":
        return "AFM open boundary conditions"
    else:
        return "FM periodic boundary conditions"

__iCurrShape=-1
def nextShape():
    listaShapes=['o','v','^','s','P','p']
    global __iCurrShape
    __iCurrShape= (__iCurrShape+1)%len(listaShapes)
    return listaShapes[__iCurrShape]
    


hvalues=np.arange(0.05,1.55,.05) 
params=[{"N":200, "type":"afm_pbc", "style":"-"},
        {"N":100, "type":"afm_obc", "style":"-"},
        {"N":100, "type":"fm_pbc", "style":"--"}]
for p in params:
    n=p['N']
    type=p['type']
    dist=np.arange(1,n+1,1)
    r=np.where(dist==int(n/2))[0][0]
    fid=np.load('RedFid_%s_N%d.npy'%(type,n),allow_pickle=True)
    plt.plot(hvalues,np.sqrt(abs(1-fid[:,r])),nextShape()+p['style'],label=typeToDescr(type))
plt.legend()
plt.xlabel("h",fontsize=__labelFontSize)
plt.ylabel(r"$\mathcal{B}_{N,\frac{N-1}{2}}$",fontsize=__labelFontSize)


plt.show()

