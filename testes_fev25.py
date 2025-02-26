from qsp_tree_reduction_22fev25 import *
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile

import numpy as np
from qclib.util import get_state
from datetime import datetime
from threading import Thread
import time
import multiprocessing
from multiprocessing import Process
import random
import sys
import os
#import winsound

dc = -9
Atol = 0.0001 # absolute tolerance (atol): absolute(a - b) <= atol
Rtol = 0.01

num_qubits = 5
density = 0.3
 
#m = 1*num_qubits**3  #number of non-zero amplitudes [10n, n², 2n², 8n², n³]


show_graphs = True
show_circuits = True


"""
State 4 qubits problems with subtract subtress:
0.3735+0.0603j 	 0.0000+0.0000j 	 0.0000+0.0000j 	 0.0000+0.0000j 	 
0.1101+0.2282j 	 0.0000+0.0000j 	 0.3812+0.1968j 	 0.0000+0.0000j 	 
0.1807+0.3931j 	 0.0000+0.0000j 	 0.0000+0.0000j 	 0.0000+0.0000j 	 
0.3677+0.3781j 	 0.3322+0.1816j 	 0.0000+0.0000j 	 0.0000+0.0000j 
"""

"""
QuantumState = np.array(0.2772+0.5056j, 	 0.2805+0.2758j, 	 0.0000+0.0000j, 	 0.2485+0.3891j, 	 
0.0000+0.0000j, 	 0.0000+0.0000j, 	 0.0000+0.0000j, 	 0.0000+0.0000j, 	 
0.0000+0.0000j, 	 0.0000+0.0000j, 	 0.0000+0.0000j, 	 0.0000+0.0000j, 	 
0.0000+0.0000j, 	 0.2585+0.0090j, 	 0.4696+0.1106j, 	 0.0000+0.0000j )
"""


#"""
#state1 = np.random.rand(2**num_qubits)+np.random.rand(2**num_qubits)*1j
#state2 = np.random.rand(2**num_qubits)+np.random.rand(2**num_qubits)*1j
#"""
#num_qubits = 2*num_qubits
#m = int(density*2**num_qubits)
#m =360#2**num_qubits
#print("m: ", m, "\n")
m = 16

QuantumState = generate_random_state_n_m(num_qubits, m)

#QuantumState = np.array([0.0000+0.0000j, 	 0.0000+0.0000j, 	 0.0000+0.0000j, 	 0.3861+0.3612j, 	 
#0.0000+0.0000j, 	 0.0000+0.0000j, 	 0.4456+0.0123j, 	 0.0000+0.0000j, 	 
#0.0000+0.0000j, 	 0.0000+0.0000j, 	 0.0000+0.0000j, 	 0.1549+0.3567j, 	 
#0.1439+0.5349j, 	 0.0000+0.0000j, 	 0.1095+0.2274j, 	 0.0000+0.0000j ])


#QuantumState = np.array([0.0000+0.0000j, 	 0.0000+0.0000j, 	 0.0000+0.0000j, 	 0.4317+0.2022j, 	 
#0.0000+0.0000j, 	 0.1776+0.4137j, 	 0.0000+0.0000j, 	 0.0000+0.0000j, 	 
#0.0000+0.0000j, 	 0.0000+0.0000j, 	 0.1877+0.3430j, 	 0.3253+0.3777j, 	 
#0.2578+0.2024j, 	 0.0000+0.0000j, 	 0.0831+0.2334j, 	 0.0000+0.0000j])

#QuantumState = np.array([0.0000+0.0000j ,	 0.0087+0.0712j, 	 0.0000+0.0000j, 	 0.3147+0.2814j, 	 
#0.0000+0.0000j, 	 0.3744+0.1977j, 	 0.0000+0.0000j, 	 0.0000+0.0000j, 	 
#0.0000+0.0000j, 	 0.0000+0.0000j, 	 0.0000+0.0000j, 	 0.0108+0.1192j, 	 
#0.0000+0.0000j, 	 0.1496+0.4747j, 	 0.1771+0.5865j, 	 0.0000+0.0000j])


#QuantumState = np.array([0.4456+0.2074j, 	 0.0000+0.0000j, 	 0.2794+0.4736j, 	 0.0000+0.0000j, 	 
#0.0000+0.0000j, 	 0.0000+0.0000j, 	 0.4412+0.5113j, 	 0.0000+0.0000j])

"""
numcnots_o_mc 20
numcnots_o_mx 12
numcnots_p_mc 18
numcnots_p_mx 16
numcnots_m_mc 18
numcnots_m_mx 10
numcnots_sg_mc 12
numcnots_sg_mx 10
numcnots_subtr_paths 10

Para este caso, subtr paths se saiu melhor que mx original

"""

"""
Quanto a sparsidade é elevada, subtract paths tem desempenho melhor ou igual a multiplexor (mx)
Caso: 6 qubits, m = 3:
    QS[28] = 0.2589+0.0245j,
    QS[30] = 0.2373+0.3839j, 
    QS[32] = 0.5099+0.6846j
    
numcnots_o_mc 282
numcnots_o_mx 124
numcnots_p_mc 36
numcnots_p_mx 28
numcnots_m_mc 32
numcnots_m_mx 20
numcnots_sg_mc 22
numcnots_sg_mx 20
numcnots_subtr_paths 20    
"""

"""

caso 7 qubits, m=3:
    numcnots_o_mc 638
    numcnots_o_mx 252
    numcnots_p_mc 64
    numcnots_p_mx 38
    numcnots_m_mc 64
    numcnots_m_mx 38
    numcnots_sg_mc 50
    numcnots_sg_mx 38
    numcnots_subtr_paths 46
"""





#QuantumState = np.kron(state1, state2)
#QuantumState = QuantumState/np.linalg.norm(QuantumState)

num_qubits = int(np.log2(len(QuantumState)))

print("State:")
printvector4fn(QuantumState,4)
printvector4fn(abs(QuantumState),4)


state_tree = get_state_tree(QuantumState, num_qubits)
if state_tree.arg != 0:
    global_phase = state_tree.arg
else:
    global_phase = 0

if dontcare_condition:
    angle_tree = create_angles_tree_with_dont_cares(state_tree, dc)
else:
    angle_tree = create_angles_tree(state_tree)


Gy = tree_to_graph_weighted_edges(angle_tree, 'y')
Gz = tree_to_graph_weighted_edges(angle_tree, 'z')
display_all_graph(Gy, Gz, show_graphs, 'originals')
#Gy = remove_subtrees_zero(Gy)
#Gz = remove_subtrees_zero(Gz)

Gyo=Gy.copy()
Gzo=Gz.copy()

Gyo = remove_subtrees_zero(Gyo)
Gzo = remove_subtrees_zero(Gzo)

print("originals mc")
state_o_mc, numcnots_o_mc = CNOTS_QC_MC_with_params(Gyo, Gzo, global_phase, num_qubits, show_circuits)
state_o_mx, numcnots_o_mx = CNOTS_QC_MX_with_params(Gyo, Gzo, global_phase, num_qubits, show_circuits)


#if show_graphs:
#    plt.figure("Gy orig")
#    display_graph(Gy)
#    plt.figure("Gz orig")
#    display_graph(Gz)

if dontcare_condition(QuantumState):
    Gy = prune_graph_dontcares_one(Gy, dc)
    Gz = prune_graph_dontcares_one(Gz, dc)
    
    Gyp = remove_subtrees_zero(Gy)
    Gzp = remove_subtrees_zero(Gz)
    
    
    display_all_graph(Gy, Gz, show_graphs, 'prune')
        
    print("Prune")
    state_p_mc, numcnots_p_mc = CNOTS_QC_MC_with_params(Gyp, Gzp, global_phase, num_qubits, show_circuits)
    state_p_mx, numcnots_p_mx = CNOTS_QC_MX_with_params(Gyp, Gzp, global_phase, num_qubits, show_circuits)

#Gyp = Gy.copy()
#Gzp = Gz.copy()

#Gy = remove_subtrees_zero(Gy)
#Gz = remove_subtrees_zero(Gz)

#if show_graphs:
#    display_graph(Gy)
#    display_graph(Gz)

#Gy = remove_first_zero_node(Gy)
#Gz = remove_first_zero_node(Gz)

#if show_graphs:
    #display_graph(Gy)
    #display_graph(Gz)

Gy = merge_equal_sibling_subtrees(Gy)
Gz = merge_equal_sibling_subtrees(Gz)

Gym = remove_subtrees_zero(Gy)
Gzm = remove_subtrees_zero(Gz)

Gyc = Gy.copy()
Gzc = Gz.copy()


display_all_graph(Gy, Gz, show_graphs, 'merge')


print("merge mc")
state_m_mc, numcnots_m_mc = CNOTS_QC_MC_with_params(Gym, Gzm, global_phase, num_qubits, show_circuits)
state_m_mx, numcnots_m_mx = CNOTS_QC_MX_with_params(Gym, Gzm, global_phase, num_qubits, show_circuits)



Gy = remove_subtrees_zero(Gy)
Gz = remove_subtrees_zero(Gz)

Gy = subtract_subgraphs_brothers_down_top_10fev(Gy)
#Gz = subtract_subgraphs_brothers_down_top(Gz)
Gz = subtract_subgraphs_brothers_down_top_10fev(Gz)




display_all_graph(Gy, Gz, show_graphs, 'subtract_graphs')

#Gy, Gz = remove_zero_root_node_Gy_and_Gz(Gy, Gz)
print("subtract subgraphs") 
state_sg_mc, numcnots_sg_mc = CNOTS_QC_MC_with_params(Gy, Gz, global_phase, num_qubits, show_circuits)
state_sg_mx, numcnots_sg_mx = CNOTS_QC_MX_with_params(Gy, Gz, global_phase, num_qubits, show_circuits)

plt.show()


Gyc = remove_subtrees_zero(Gyc)
Gzc = remove_subtrees_zero(Gzc)

print("\n \n after subtract paths \n \n")
qcy_0 = BuildQuantumCircuit_From_Paths(num_qubits, Gyc, 'y')
qcz_0 = BuildQuantumCircuit_From_Paths(num_qubits, Gzc, 'z')
qc_qsp_subtract_paths_0 = QuantumCircuit.compose(qcy_0, qcz_0)
qc_qsp_subtract_paths_0.global_phase = global_phase

#qc_qsp = qc_qsp_subtract_paths_0.decompose()

if show_circuits:
  plt.figure()
  qc_qsp_subtract_paths_0.draw()
  
  #display(qc_qsp_subtract_paths_0.draw('mpl'))
  qc_qsp_subtract_paths_0.draw('mpl')
  
  plt.show()
  #print(qc_qsp_subtract_paths_0)
  
qc_qsp_subtract_paths_0.draw()
qc_qsp_subtract_paths_0.draw(output='mpl')
#display(qc_qsp_subtract_paths_0.draw('mpl'))

print("subtrpaths")
state_sp = get_state(qc_qsp_subtract_paths_0.reverse_bits())
printvector4fn(state_sp,4)

printvector4fn(abs(state_sp),4)

qc_subtr_paths = transpile(qc_qsp_subtract_paths_0, basis_gates=['u', 'cx'], optimization_level=0)
numcnots_subtr_paths = qc_subtr_paths.count_ops().get('cx', 0)



print("numcnots_o_mc", numcnots_o_mc)
print("numcnots_o_mx", numcnots_o_mx)

if dontcare_condition(QuantumState):
  print("numcnots_p_mc", numcnots_p_mc)
  print("numcnots_p_mx", numcnots_p_mx)

print("numcnots_m_mc", numcnots_m_mc)
print("numcnots_m_mx", numcnots_m_mx)

print("numcnots_sg_mc", numcnots_sg_mc)
print("numcnots_sg_mx", numcnots_sg_mx)

print("numcnots_subtr_paths", numcnots_subtr_paths)

print()

print("isclose original:")
print("mc:",np.isclose(QuantumState, state_o_mc, rtol=Rtol, atol=Atol))
print("mx:",np.isclose(QuantumState, state_o_mx, rtol=Rtol, atol=Atol))
if dontcare_condition(QuantumState):
  print()
  print("isclose prune:")
  print("mc:",np.isclose(QuantumState, state_p_mc, rtol=Rtol, atol=Atol))
  print("mx:",np.isclose(QuantumState, state_p_mx, rtol=Rtol, atol=Atol))
print()
print("isclose merge:")
print("mc:",np.isclose(QuantumState, state_m_mc, rtol=Rtol, atol=Atol))
print("mx:",np.isclose(QuantumState, state_m_mx, rtol=Rtol, atol=Atol))
print()
print("isclose subtr graphs:")
print("mc:",np.isclose(QuantumState, state_sg_mc, rtol=Rtol, atol=Atol))
print("mx:",np.isclose(QuantumState, state_sg_mx, rtol=Rtol, atol=Atol))
print()
print("isclose subtr paths:")
print("mc:",np.isclose(QuantumState, state_sp, rtol=Rtol, atol=Atol))
#print("mx:",np.isclose(QuantumState, state_m_mx))

plt.show()