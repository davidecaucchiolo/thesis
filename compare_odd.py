import numpy as np
import numpy.linalg as nla

import qtealeaves as qtl
from qtealeaves.models import get_quantum_ising_1d
import qtealeaves.emulator as emulator
import matplotlib.pyplot as plt
from qtealeaves import modeling, operators
from itertools import product
import torch as tc

from qredtea.torchapi import default_pytorch_backend
from qredtea.torchapi.qteatorchtensor import QteaTorchTensor

import io, os, sys
from contextlib import redirect_stdout, redirect_stderr
import time

__positive=False
plt.rcParams.update({'font.size':12})
__labelFontSize=15



__iCurrentShape=-1
def nextShape():
    shapesList=['v','^','s','P','p']
    global __iCurrentShape
    __iCurrentShape= (__iCurrentShape+1)%len(shapesList)
    return shapesList[__iCurrentShape]



def get_quantum_ising_1d_closed():
    model_name = lambda params: "QIsing_g%2.4f" % (params["h"])

    model = modeling.QuantumModel(1, "L", name=model_name)
    model += modeling.LocalTerm("sx", strength="h", prefactor=(1 if __positive else -1))
    model += modeling.TwoBodyTerm1D(
        ["sz", "sz"], 1, strength="J", prefactor=1, has_obc=False
    )

    ising_ops = operators.TNSpin12Operators()

    return model, ising_ops
def der2(f,h):
    r=[0.]*len(f)
    for i in range(1,len(f)-1):
        r[i]=(f[i+1]-2*f[i]+f[i-1])/h**2
    r[0]=r[1]
    r[-1]=r[-2]
    return r

fig, (axs1,axs2) = plt.subplots(1,2,figsize=(10,4))
jValues=[-1.,1.]
hValues=np.linspace(0,2,41)
lValues=[101]

tn_type=6; ## MPS
tensor_backend=default_pytorch_backend()
statics_method=1
input_folder=None
output_folder=None

if input_folder is None:
    input_folder = lambda params: ("QI1d/input_L%d_h%f" % (params["L"],params["h"]))+("_FM" if params["J"]==-1. else "")+("" if __positive else "_neg")
if output_folder is None:
    output_folder = lambda params: ("QI1d/output_L%d_h%f" % (params["L"],params["h"]))+("_FM" if params["J"]==-1. else "")+("" if __positive else "_neg")

save_folder= lambda elem: 'QI1d/psi_L_%d_h_%.2f_FM'%(elem["L"],elem["h"])


model, my_ops = get_quantum_ising_1d_closed()

my_conv = qtl.convergence_parameters.TNConvergenceParameters(
    max_iter=40, max_bond_dimension=128, statics_method=statics_method,
    abs_deviation=1e-10,krylov_maxiter=15)

my_obs = qtl.observables.TNObservables()
my_obs += qtl.observables.TNObsLocal("<sz>", "sz")
my_obs += qtl.observables.TNObsBondEntropy()
my_obs += qtl.observables.TNState2File(save_folder,'U')

simulation = qtl.QuantumGreenTeaSimulation(
    model,
    my_ops,
    my_conv,
    my_obs,
    tn_type=tn_type,
    tensor_backend=2,
    py_tensor_backend=tensor_backend,
    folder_name_input=input_folder,
    folder_name_output=output_folder,
    has_log_file=False,
    store_checkpoints=False,
    )
t0=time.time()
params = [{"L":L,"J":J,"h":float(h)} for L,h,J in product(lValues,hValues,jValues)]

simulation.run(params, delete_existing_folder=False)

print(time.time()-t0)
xVals={}
yVals={}
for elem in params:
    results = simulation.get_static_obs(elem)
    if (elem["L"], elem["J"]) not in xVals:
        xVals[elem["L"], elem["J"]]=[elem["h"]]
        yVals[elem["L"], elem["J"]]=[results["energy"]]
    else:
        xVals[elem["L"], elem["J"]].append(elem["h"])
        yVals[elem["L"], elem["J"]].append(results["energy"])


for l,j in xVals:
    hVals=xVals[l,j][1]-xVals[l,j][0]
    axs1.plot(xVals[l,j],[v/l for v in der2(yVals[l,j],hVals)],nextShape()+"-",label="FM" if j==-1. else "AFM")

axs1.set_xlabel("h",fontsize=__labelFontSize)
axs1.set_ylabel(r"$\frac{1}{N}\ \frac{\partial ^2E}{\partial h^2}$",fontsize=__labelFontSize)


axs1.legend()





__iCurrentShape=-1


##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################


jValues=[-1., 1.]
lValues= [101]
hValues=np.linspace(0.05,2, 40)

params = [{"L":L,"J":J,"h":float(h)} for L,h,J in product(lValues,hValues,jValues)]
hValues=[float(h) for h in hValues]
mps_list={}

σx=tc.tensor([[0.,1.],[1.,0.]])
σy=tc.tensor([[0,-1.j],[1.j,0]])
σz=tc.tensor([[1.,0.],[0,-1.]])
I2=tc.tensor([[1.,0.],[0,1.]])

def from_qtea_to_torch(mps):
    a=mps.to_tensor_list()
    return [x.elem for x in a]

def from_torch_to_qtea(mps):
    tlist=[QteaTorchTensor.from_elem_array(x) for x in mps]
    return emulator.MPS.from_tensor_list(tlist,tensor_backend=tensor_backend)

def PxProjector_MPO(N,p,dtype=tc.float64):
    Hl = tc.zeros((2,2,2,2),dtype=dtype)
    H1 = tc.zeros((1,2,2,2),dtype=dtype);
    Hn = tc.zeros((2,2,1,2),dtype=dtype)
    Hl[0,:,0,:] = I2; Hl[1,:,1,:] = σx;
    H1[0,:,0,:]=I2; H1[0,:,1,:]=σx
    Hn[0,:,0,:]=.5*I2; Hn[1,:,0,:]=.5*p*σx
    H = [Hl for l in range(N)]
    H[0] = H1; H[N-1] = Hn
    return H

def Product(mpo,mps):
    O=mpo.copy(); S=mps.copy()
    M=[]
    for i in range(len(mps)):
        t=tc.einsum('ijk,lmnj->ilmkn',S[i],O[i])
        s=t.shape
        M.append(tc.reshape(t,(s[0]*s[1],s[2],s[3]*s[4])))
    return M

proj=PxProjector_MPO(lValues[0],+1,tc.complex128) 

for elem in params:
    ttn_filename = ttn_file(elem)+".pklmps"
    if mps_list.get((elem["L"],elem["J"])) is None:
        mps_list[elem["L"],elem["J"]]=[None]*len(hValues)
    mps_list[elem["L"],elem["J"]][hValues.index(float(elem["h"]))]=emulator.MPS.read_pickle(ttn_filename)
        
for J,i in [(+1.,i) for i in range(40)] + [(-1.,i) for i in range(40)]:
    gs=mps_list[lValues[0],J][i]
    gs_torch=from_qtea_to_torch(gs)
    new_gs=from_torch_to_qtea(Product(proj,gs_torch)) 
    new_gs/=new_gs.norm().item() 
    mps_list[lValues[0],J][i] = new_gs


for l,J in product(lValues,jValues):
    fidelity=[abs(mps_list[l,J][i].dot(mps_list[l,J][i-1]))**2 for i in range(1,len(hValues))]
    axs2.plot(hValues[1:],fidelity,nextShape()+"-", label="FM" if J==-1. else "AFM")

mps_list=None


axs2.set_xlabel("h",fontsize=__labelFontSize)
axs2.set_ylabel(r"$\mathcal{F}$",fontsize=__labelFontSize)

axs2.legend()

fig.subplots_adjust(left=.11, bottom=.13, right=.962, top=.9, wspace=0.4)

axs1.set_title("(A)", loc="left")
axs2.set_title("(B)", loc="left")

plt.show()

