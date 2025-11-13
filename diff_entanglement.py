import numpy as np
import numpy.linalg as nla

import qtealeaves as qtl
from qtealeaves.models import get_quantum_ising_1d
import matplotlib.pyplot as plt
from qtealeaves import modeling, operators, emulator
from itertools import product
import torch as tc

from qredtea.torchapi import default_pytorch_backend
from qredtea.torchapi.qteatorchtensor import QteaTorchTensor

import io, os, sys
from contextlib import redirect_stdout, redirect_stderr
import time

plt.rcParams.update({'font.size':12})
__labelFontSize=15



def get_quantum_ising_1d_closed():
    model_name = lambda params: "QIsing_g%2.4f" % (params["h"])

    model = modeling.QuantumModel(1, "L", name=model_name)
    model += modeling.LocalTerm("sx", strength="h", prefactor=-1)
    model += modeling.TwoBodyTerm1D(
        ["sz", "sz"], 1, strength="J", prefactor=1, has_obc=False
    )

    ising_ops = operators.TNSpin12Operators()

    return model, ising_ops

fig, axs = plt.subplots(1,1,figsize=(8,5))
h= 0.1
hValues=[h]
lValues=[L for L in range(21,102,10)]
JValues=[-1.,1.]

tn_type=6; ## MPS
tensor_backend=default_pytorch_backend() 
statics_method=1
input_folder=None
output_folder=None

if input_folder is None:
    input_folder = lambda params: ("QI1d/input_L%d_h%f" % (params["L"],params["h"]))+("" if params["J"]==1. else "_FM")+"_neg"
if output_folder is None:
    output_folder = lambda params: ("QI1d/output_L%d_h%f" % (params["L"],params["h"]))+("" if params["J"]==1. else "_FM")+"_neg"
save_folder= lambda elem: 'QI1d/psi_L_%d_h_%.2f'%(elem["L"],elem["h"])+"_neg"

model, my_ops = get_quantum_ising_1d_closed()

my_conv = qtl.convergence_parameters.TNConvergenceParameters(
    max_iter= 40, max_bond_dimension=128, statics_method=statics_method,
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
params = [{"L":L,"J":J,"h":h} for L,J in product(lValues,JValues)]
simulation.run(params, delete_existing_folder=False)

σx=tc.tensor([[0.,1.],[1.,0.]])
σy=tc.tensor([[0,-1.j],[1.j,0]])
σz=tc.tensor([[1.,0.],[0,-1.]])
I2=tc.tensor([[1.,0.],[0,1.]])

def from_qtea_to_torch(mps):
    ##### here we convert a qtealeaves mps into a list of torch tensors
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


p=1  ### choose a parity sector


print(time.time()-t0)
xVals={}
yValsFM={}
yValsAFM={}
for elem in params:
    results = simulation.get_static_obs(elem)
    if elem["J"]==-1.:
        mps_list=[emulator.MPS.read_pickle(save_folder(elem)+'.pklmps')]
        fidelity=[abs(mps_list[i].dot(mps_list[i-1]))**2 for i in range(1,len(mps_list))]

        new_mps_list=[]
        for gs in mps_list:
            gs_torch=from_qtea_to_torch(gs)
            proj=PxProjector_MPO(gs.num_sites,p,gs_torch[0].dtype) 
            new_gs=from_torch_to_qtea(Product(proj,gs_torch)) 
            new_gs/=new_gs.norm().item() 
            new_mps_list.append(new_gs)

        gs=new_mps_list[0]
        entanglement=gs.meas_bond_entropy()
        if (yValsFM.get(elem["L"]) is None):
            yValsFM[elem["L"]]=[-42]*int(elem["L"])
        for x in entanglement.keys():
            yValsFM[elem["L"]][int(x[1]-x[0])] = entanglement[x]/np.log(2)


    else:
        for i in results["bond_entropy0"]:
            if (xVals.get(elem["L"]) is None):
                xVals[elem["L"]]=[-16]*int(elem["L"])
            if (yValsAFM.get(elem["L"]) is None):
                yValsAFM[elem["L"]]=[-15]*int(elem["L"])

            xVals[elem["L"]][int(i[1]-i[0])] = (i[1]-i[0])/elem["L"]
            yValsAFM[elem["L"]][int(i[1]-i[0])] = results["bond_entropy0"][i]/np.log(2)


for L in lValues:
    print("L :",L)
    axs.plot(xVals[L][1:],np.array(yValsAFM[L][1:]) - np.array(yValsFM[L][1:]),"o-",label="N = %d"%L)


x=np.linspace(0.001,1,1000)
plt.plot(x, -x*np.log2(x) - (1-x)*np.log2(1-x), label=r"$S_W(x)$")
plt.xlabel("x = M/N",fontsize=__labelFontSize)
plt.ylabel("S",fontsize=__labelFontSize)

plt.legend()

fig.subplots_adjust(left=.08, bottom=.11, right=.962, top=.96)


plt.show()

