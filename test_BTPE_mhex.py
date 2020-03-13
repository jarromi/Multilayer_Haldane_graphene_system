# BTPE_mhex.py test

import BTPE_mhex as BTPE
from cmath import *

PS = BTPE.PhysicalSystem(3)
PS.update_all_layers(4.,0.,0.,1.)
PS.update_all_couplings(1.)
PS.update_layer(0,4.,1.,pi/2,0.)
PS.reset_kSpace()
print(PS)

(bg1,bg2)=PS.init_eigdata()
print("Gap at K: "+str(bg1)+"; and gap at K': "+str(bg2)+"\n")

ChN,BCtemp=PS.method1()
IsL=PS.method2()
I2G=PS.method3()

print("Chern number is: "+str(ChN)+"\n")
print("Topological invariant of layer 0 is: "+str(IsL[0])+"\n")
print("Topological invariant of layer 1 is: "+str(IsL[1])+"\n")
print("Topological invariant of layer 2 is: "+str(IsL[2])+"\n")
print("Topological invariant of layers 1 and 2 is: "+str(I2G)+"\n")

IdGp,kv1,kv2=PS.find_indirect_gap(5)
print("Indirect gap is: "+str(IdGp)+" at positions "+str(kv1)+" "+str(kv2)+"\n")
DiGp,kv=PS.find_direct_gap(5)
print("Direct gap is: "+str(DiGp)+" at position "+str(kv)+"\n")