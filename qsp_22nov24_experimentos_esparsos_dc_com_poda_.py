import numpy as np
from my_qclib_utils import *
from qclib.util import get_state
import cmath as cm
#from pylatexenc import *
import matplotlib as mpl
from qiskit import QuantumCircuit, transpile
from predefined_states import *

from qsp_22nov24_rotinas import *

"""
   1 - Podar a árvore, eliminando sub-árvores de don't cares, 
       diminuindo assim o espaço de busca para redução por similaridade
   2 - Realizar redução por similaridade
"""

def mycreate_angles_tree(state_tree):
  """
  Modificação da rotina create_angles_tree do qclib. Teve que ser colocada aqui
  pq não quis mexer muito para resolver o problema das variáveis globais dc e dcz

  :param state_tree: state_tree is an output of state_decomposition function
  :return: tree with angles that will be used to perform the state preparation
  """
  global dc
  global dcz

  #mag = 0.0
  if state_tree.mag != 0.0:
    mag = state_tree.right.mag / state_tree.mag
  else:
    mag = dc
    dc = dc - 1

  arg = state_tree.right.arg - state_tree.arg

  # Avoid out-of-domain value due to numerical error.

  if mag <= -9:
    angle_y = mag

  elif mag > -9 and mag < -1.0:
    angle_y = -math.pi
  elif mag > 1.0:
    angle_y = math.pi
  else:
    angle_y = 2 * math.asin(mag)

  if state_tree.mag == 0.0:
    angle_z = dcz
    dcz = dcz - 1
  else:
    angle_z = 2 * arg

  twopi = 2 * np.pi

  #while angle_z > twopi:
  #  angle_z = angle_z - twopi

  while angle_z < 0 and angle_z > -9:
    angle_z = angle_z + 2*twopi

  node = NodeAngleTree(state_tree.index, state_tree.level, angle_y, angle_z, None, None)

  if not is_leaf(state_tree.left):
    node.right = mycreate_angles_tree(state_tree.right)
    node.left = mycreate_angles_tree(state_tree.left)
  return node

exibir_todos_os_grafos =  True
exibir_grafo_inicial = False
exibir_apenas_os_grafos_finais = False
exibir_circuitos = True
subtract_graphs = False
exibir_estados = True
show_stats = True

#Choose a predefined state in the file "predefined_states.py", or elaborate your own.
state, Num_qubits = Norm_State_and_Num_Qubits(Separable_States_Qubits())

state_tree = get_state_tree(state, Num_qubits)

Gy_without_reduction, Gz_without_reduction, global_phase = gen_graphs_Ry_and_RZ(state_tree)

show_all_graphs("Original Graphs", Gy_without_reduction, Gz_without_reduction, exibir_todos_os_grafos)

qc_Ry_original, qc_Rz_original, qc_qsp_original = Build_all_circuits(Num_qubits, Gy_without_reduction, Gz_without_reduction, global_phase)   

show_circuits("Circuit from original trees:", qc_Ry_original, qc_Rz_original, qc_qsp_original, exibir_circuitos)

print_states_from_circuit("Original Circuit", state, qc_qsp_original, exibir_estados)

print("\n ---------Reduções--------\n")

#Gy = Gy_without_reduction.copy()
#Gz = Gz_without_reduction.copy()

print()
print("-#--#--#--#-Reduction with pruning don't cares-#--#--#--#-")
print()

dc = -9
dcz = -9
angle_tree = mycreate_angles_tree(state_tree)

#global_phase = get_global_phase(state_tree)

#Cria grafo a partir da árvore
Gy = tree_to_graph_y_weighted_edges(angle_tree)
Gz = tree_to_graph_z_weighted_edges(angle_tree)

#Gy, Gz, global_phase, dc, dcz = gen_graphs_Ry_and_RZ_with_dont_cares(state_tree)
show_all_graphs("Graphs with don't cares merge", Gy, Gz, exibir_todos_os_grafos)

Gy, Gz = Eliminar_subarvore_dontcares(Gy, Gz, -9)

show_all_graphs("Graphs after pruning", Gy, Gz, exibir_todos_os_grafos)

#Mesclagem de sub-grafos iguais
Gy, Gz = merge_each_graph(Gy, Gz)

show_all_graphs("Graphs after merge", Gy, Gz, exibir_todos_os_grafos)

Gy, Gz = remove_last_node_zero_Gy_and_Gz(Gy, Gz)

show_all_graphs("Graphs after remove zero leafs", Gy, Gz, exibir_todos_os_grafos)

print("\n Eliminação de um nós pais com ângulos zero\n")

Gy, Gz = remove_zero_root_node_Gy_and_Gz(Gy, Gz)

show_all_graphs("Graphs after remove zero root", Gy, Gz, exibir_todos_os_grafos)

qc_Ry_reduzido_dc_noe, qc_Rz_reduzido_dc_noe, qc_qsp_reduzido_dc_noe = Build_all_circuits(Num_qubits, Gy, Gz, global_phase)

show_circuits("Reduced circuit with pruning dont cares:", qc_Ry_reduzido_dc_noe, qc_Rz_reduzido_dc_noe, qc_qsp_reduzido_dc_noe, exibir_circuitos)

print_states_from_circuit("Reduced circuit pruning dont cares", state, qc_qsp_reduzido_dc_noe, exibir_estados)

#Show stats
circuit_transpile_and_counts(state, qc_Ry_original, qc_Ry_reduzido_dc_noe, qc_Rz_original, qc_Rz_reduzido_dc_noe, show_stats )

Gy = subtract_subgraphs_brothers_down_top(Gy)
Gz = subtract_subgraphs_brothers_down_top(Gz)

Gy, Gz = remove_last_node_zero_Gy_and_Gz(Gy, Gz)

qc_Ry_reduzido_subtract, qc_Rz_reduzido_subtract, qc_qsp_reduzido_subtract = Build_all_circuits(Num_qubits, Gy, Gz, global_phase)

show_all_graphs("Reduced circuit after subtract subgraphs:", Gy, Gz, exibir_todos_os_grafos)

show_circuits("Reduced circuit after subtract subgraphs:", qc_Ry_reduzido_subtract, qc_Rz_reduzido_subtract, qc_qsp_reduzido_subtract, exibir_circuitos)

print_states_from_circuit("Reduced circuit after subtract subgraphs", state, qc_qsp_reduzido_subtract, exibir_estados)

circuit_transpile_and_counts(state, qc_Ry_original, qc_Ry_reduzido_subtract, qc_Rz_original, qc_Rz_reduzido_subtract, show_stats )
