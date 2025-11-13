from qutip import *
import numpy as np
import matplotlib.pyplot as plt

__labelFontSize=12




#operator that applies only to i-th qbit
def opOnIth (op,i,N):
    l=[identity(2) for x in range(N)]
    l[i]=op
    return tensor(l)

def numToQbits(x,N):
    binRepr=bin(x)[2:]
    binRepr='0'*(N-len(binRepr))+binRepr
    return tensor([basis(2,int(b)) for b in binRepr])

rendiVisibili=False
jVals=[-1.,1.]

gap={}
hvals=np.linspace(0,.8,1000)
def calcola(N,J,ax):
    H0=J*sum([opOnIth(sigmaz(),i,N)*opOnIth(sigmaz(),(i+1)%N,N) for i in range(N)])
    V=sum([opOnIth(sigmax(),i,N) for i in range(N)])

    

    #Autovalori di H al variare di h
    eeVals=np.array([(H0-h*V).eigenenergies() for h in hvals]).transpose()

    aVals,aVett= H0.eigenstates()
    eBande=[aVals[0]]
    vettBande=[[aVett[0]]]
    for i in range(1,len(aVals)):
        if aVals[i]==aVals[i-1]:
            vettBande[-1].append(aVett[i])
        else:
            eBande.append(aVals[i])
            vettBande.append([aVett[i]])

    gap[N]=eeVals[len(vettBande[0])]-eeVals[len(vettBande[0])-1]

    indexList=list(range(len(eeVals)))
    new_eeVals=[eeVals[indexList,0],eeVals[indexList,1]]
    for t in range(2,len(eeVals[0])):
        protetti=[False]*len(indexList)
        for ii in range(1,len(indexList)-1):


            if t>100 and abs(2*new_eeVals[-1][ii]-new_eeVals[-2][ii] - eeVals[indexList[ii],t]) >\
                    abs(2*new_eeVals[-1][ii]-new_eeVals[-2][ii] - eeVals[indexList[ii-1],t]) and\
                    abs(eeVals[indexList[ii],t] - eeVals[indexList[ii-1],t]) > .00003:
                        j=2
                        while not protetti[ii-j] and abs(eeVals[indexList[ii-1],t]-eeVals[indexList[ii-j],t]) < .00003:
                            j+=1
                        j-=1
                        indexList[ii-j],indexList[ii]= indexList[ii],indexList[ii-j]
                        protetti[ii]=True

        new_eeVals.append(eeVals[indexList,t])

    eeVals=np.array(new_eeVals).transpose()



    i=0

    nBandeDaMostrare =len(vettBande[0]) if  N==7 and J==1. else len(vettBande[0])+len(vettBande[1])
    for e in eeVals[:nBandeDaMostrare]:
        ax.plot(hvals,e+.05*(-2)**(N-1)+i if rendiVisibili else e)
        i+=1

    ax.set_xlabel("h",fontsize=__labelFontSize)
    ax.set_ylabel("Energy",fontsize=__labelFontSize)
    props = dict(boxstyle='round', facecolor='wheat', alpha=1)
    if N!=7 or J!=1.:
        ax.text(0.,eBande[0]+.7,"Fundamental band",bbox=props)
        ax.text(0.,eBande[1]+.7,"First excited band",bbox=props)
    else:
        ax.annotate("Ground state", xytext=(0, eBande[0]-1), 
            xy=(hvals[550], eeVals[0][550]),
            arrowprops=dict(arrowstyle="->"),bbox=props)
        # ax.text(0.,eBande[0]-1,"Ground state",bbox=props)
        ax.text(0.,eBande[0]+.4,"First excited band",bbox=props)

fig, axs = plt.subplots(1,2, figsize=(8,3))
calcola(6,-1.,axs[0])
calcola(6,1.,axs[1])
axs[0].set_title("(A)",loc="left")
axs[1].set_title("(B)",loc="left")
fig.subplots_adjust(left=.08, bottom=.17, right=.96, top=.91, wspace=.3)
plt.show()

fig, axs = plt.subplots(1,2, figsize=(8,3))
calcola(7,-1.,axs[0])
calcola(7,1.,axs[1])
axs[0].set_title("(A)",loc="left")
axs[1].set_title("(B)",loc="left")
fig.subplots_adjust(left=.08, bottom=.17, right=.96, top=.91, wspace=.3)
plt.show()
