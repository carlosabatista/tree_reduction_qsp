import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate, RZGate, CRYGate
from qiskit import transpile #, Aer, execute
from qiskit.visualization import plot_histogram
from qclib.util import get_state
from my_qclib_utils import *
import random

def tree_to_graph_y_weighted_edges(root):
    graph = nx.DiGraph()
    def traverse(node, parent=None):
        if node:
            g_node_index = 2**node.level+node.index
            g_node_weight = np.round(node.angle_y,5)
            graph.add_node(g_node_index, weight=g_node_weight, level = node.level)
            if parent:
                parent = (2**node.level+node.index)//2
                edge_weight = (-1)**(g_node_index+1)
                graph.add_edge(parent, 2**node.level+node.index, weight = edge_weight)
            traverse(node.left, node)
            traverse(node.right, node)
    traverse(root)
    return graph

def tree_to_graph_z_weighted_edges(root):
    graph = nx.DiGraph()
    def traverse(node, parent=None):
        if node:
            g_node_index = 2**node.level+node.index
            g_node_weight = np.round(node.angle_z,5)
            graph.add_node(g_node_index, weight=g_node_weight, level = node.level)
            if parent:
                parent = (2**node.level+node.index)//2
                edge_weight = (-1)**(g_node_index+1)
                graph.add_edge(parent, 2**node.level+node.index, weight = edge_weight)
            traverse(node.left, node)
            traverse(node.right, node)
    traverse(root)
    return graph

def display_graph(G):
    pos = nx.nx_pydot.graphviz_layout(G, prog="dot") 
    labels1 = nx.get_node_attributes(G, 'weight')
    labels2 = nx.get_node_attributes(G,'level')
    
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=8, font_weight='bold')
    #nx.draw_networkx_labels(G, pos, labels={node: f"{node}\n{labels[node]}" for node in labels})
    #nx.draw_networkx_labels(G, pos, labels=labels)
    nx.draw_networkx_labels(G, pos, labels={node: f"q{labels2[node]}     \n\n{np.round(labels1[node],4)}" for node in labels1})
    
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.show()
 
def calcular_profundidade(graph, no_inicial):
    profundidade = 0
    fila = [(no_inicial, 0)]
    visitados = set()

    while fila:
        no_atual, nivel_atual = fila.pop(0)
        if no_atual not in visitados:
            visitados.add(no_atual)
            profundidade = max(profundidade, nivel_atual)
            for vizinho in graph[no_atual]:
                fila.append((vizinho, nivel_atual + 1))

    return profundidade

def check_node_weights_equal(graph, subgraph1, subgraph2):
  """
  Verifica se os pesos dos nós em um subgraph são iguais aos pesos dos nós
  no subgraph irmão, respectivamente, para cada nó.

  As entradas são:
    graph: O graph principal.
    subgraph1: O primeiro subgraph.
    subgraph2: O segundo subgraph (irmão).

  e retorna True se os pesos dos nós forem iguais, False caso contrário.
  """

  if len(subgraph1.nodes()) != len(subgraph2.nodes()):
      return False

  #node_map = {node: i for i, node in enumerate(subgraph1.nodes())} 
  #for node in subgraph1.nodes():
  #    if 'weight' not in graph.nodes[node] or 'weight' not in graph.nodes[list(subgraph2.nodes())[node_map[node]]]:
  #        return False
  #    if graph.nodes[node]['weight'] != graph.nodes[list(subgraph2.nodes())[node_map[node]]]['weight']:
  #        return False


  list_subgraph1 = list(subgraph1.nodes)
  list_subgraph1.sort()
  list_subgraph2 = list(subgraph2.nodes)
  list_subgraph2.sort()


  node_map = {node: i for i, node in enumerate(list_subgraph1)}
  for node in list_subgraph1:
    if 'weight' not in graph.nodes[node] or 'weight' not in graph.nodes[list_subgraph2[node_map[node]]]:
      return False
    if graph.nodes[node]['weight'] != graph.nodes[list_subgraph2[node_map[node]]]['weight']:
      return False
  
  return True    

def eliminar_subgraph(graph, subgraph):
    sorted_list_graph = list(graph.nodes())
    sorted_list_graph.sort()
    if subgraph in sorted_list_graph:
        raio = calcular_profundidade(graph, subgraph)
        sub = list(nx.ego_graph(graph, subgraph, radius=raio))
        sub.sort(reverse=True)
        for node in sub:
            graph.remove_node(node)

def tem_filhos(graph, no):
    return len(list(graph.successors(no)))>0

def remove_last_node_zero(graph):

    Err = 0.0001

    again = True
    
    while again:
        again = False
    
        graph = graph.copy()
        sorted_list_graph = list(graph.nodes())
        sorted_list_graph.sort()
        zeros_to_remove = []
        for node in sorted_list_graph:
            if not(tem_filhos(graph, node)) and abs(graph.nodes[node]["weight"])<=Err:
                zeros_to_remove.append(node)
                again = True
                
        for node in zeros_to_remove:
            graph.remove_node(node)
    return graph

def eliminar_node1_se_zero(graph):
    """_summary_

        Se, no final, o nó 1 tem peso zero, significa que nada ocorre no qubit 1,
        Se for controle de um nó par, significa que é controle aberto, e como a entrada
        é |0>, então a subárvore ocorre de qualquer forma, então elimina apenas o nó 1.
        Se for controle de um nó impar, então é controle fechado e o subgraph a partir dele nunca ocorre,
        então se elimina todo o subgraph a partir de 0 
    """
    
    graph = graph.copy()
    
    sorted_list_graph = list(graph.nodes())
    sorted_list_graph.sort()
    
    list_nodes = list(graph.nodes())
    if 1 in list_nodes:
        if graph.nodes[1]['weight']==0:
            lista_filhos = list(graph.successors(1))
            
            graph.remove_node(1)
                
            for subgraph in lista_filhos:
                if subgraph%2 == 1:
                    sorted_list_graph = list(graph.nodes())
                    if subgraph in sorted_list_graph:
                        raio = calcular_profundidade(graph, subgraph)
                        sub_dead = nx.ego_graph(graph, subgraph, radius=raio)
                        for node in sub_dead:
                            graph.remove_node(node)
    return graph

def merge_subgraphs(graph):
    
    graph = graph.copy()
    
    if len(list(graph.nodes()))<=1:
        return graph, False

    sorted_list_graph = list(graph.nodes())
    sorted_list_graph.sort()
    
    subgraph_to_eliminate = []
    #lembrar de atualizar a lista de childs
    # varrer os nós e retirar os excluidos da lista
    
    for no_pai in sorted_list_graph:
        list_childs = list(graph.successors(no_pai))
        list_childs.sort()  
        for indexnode, no_1 in enumerate(list_childs[:-1]):
            for no_2 in list_childs[indexnode+1:]:
                if graph.nodes[no_1]['level'] == graph.nodes[no_2]['level']:
                    if graph.nodes[no_1]['weight'] == graph.nodes[no_2]['weight']:
                        raio1 = calcular_profundidade(graph, no_1)
                        raio2 = calcular_profundidade(graph, no_2)
                        sub_1 = nx.ego_graph(graph, no_1, radius=raio1)
                        sub_2 = nx.ego_graph(graph, no_2, radius=raio2)
                        if check_node_weights_equal(graph, sub_1, sub_2):
                            subgraph_to_eliminate.append(no_2)
                            
                            graph.remove_edge(no_pai, no_1)
                            grandparent = graph._pred[no_pai]
                            if grandparent:
                                    graph.add_edge(list(grandparent)[0], no_1, weight = graph[list(grandparent)[0]][no_pai]['weight'])
                            eliminar_subgraph(graph, no_2)
                            return graph, True   

            
    if len(subgraph_to_eliminate)>0:
        flag = True
    else:
        flag = False
    
    return graph, flag           

def eliminar_primeiro_zero(graph):
    
    graph = graph.copy()
    
    list_nodes = list(graph)
    list_nodes.sort()
     
    for node in list_nodes:
        if not(graph._pred[node]):
    
        #sorted_list_graph = list(graph.nodes())
        #sorted_list_graph.sort()
        
        #list_nodes = list(graph.nodes())
        #se no nao tem pai e tem valor zero
        #  se aresta filho = 1, elimina toda o subgrafo
        #  se -1, elimina apenas o nó de valor zero
        #if 1 in list_nodes:
            if graph.nodes[node]['weight']==0:
                lista_filhos = list(graph.successors(node))
                
                graph.remove_node(node)
                    
                for subgraph in lista_filhos:
                    if subgraph%2 == 1:
                        sorted_list_graph = list(graph.nodes())
                        if subgraph in sorted_list_graph:
                            raio = calcular_profundidade(graph, subgraph)
                            sub_dead = nx.ego_graph(graph, subgraph, radius=raio)
                            for node in sub_dead:
                                graph.remove_node(node)
    return graph

def substituir_pesos_com_edges(graph, valor):
    # Encontrar o nó com o valor especificado para don't care e substituir
    # a sub-árvore a partir desse nó pela sub-árvore irmã
    no_alvo = None
    substituir = True
    
    while substituir:
       
        sorted_list_graph = list(graph.nodes())
        sorted_list_graph.sort()
        
        for no in sorted_list_graph:
            if graph.nodes[no]['weight'] == valor:
                no_alvo = no
                break
        
        if no_alvo is None:
            #print("Valor não encontrado no graph.")
            return

        # Encontrar o subgraph irmão (nós conectados ao nó alvo)
        raio = calcular_profundidade(graph, no_alvo)
        subgraph_alvo = nx.ego_graph(graph, no_alvo, radius=raio)
        
        subgraph_irmao = None
        b = graph._pred[no_alvo]
        
        for no in sorted_list_graph[no_alvo-2:]:#graph.nodes():
            a = graph._pred[no]
            if no!=no_alvo and list(a)[0]==list(b)[0]:
                raio = calcular_profundidade(graph, no)
                subgraph_irmao = nx.ego_graph(graph, no, radius=raio)
                break

        if subgraph_irmao is None:
            return
        
        
        subgraph_alvo_lista = sorted_list_from_graph(subgraph_alvo)
        subgraph_alvo_lista.sort()
        
        subgraph_irmao_lista = sorted_list_from_graph(subgraph_irmao)
        subgraph_irmao_lista.sort()

        for no, no_alvo in zip(subgraph_alvo_lista, subgraph_irmao_lista):
            graph.nodes[no]['weight'] = subgraph_irmao.nodes[no_alvo]['weight']
            
        sorted_list_graph = list(graph.nodes())
        sorted_list_graph.sort()
        
        for nno in sorted_list_graph:
            if graph.nodes[nno]['weight'] <=-9:
                valor = graph.nodes[nno]['weight']
                substituir = True
                break
            else:
                substituir = False

def Eliminar_subarvore_dontcares(Graphy, Graphz, valor):
    # Encontrar o nó com o valor especificado para don't care e substituir
    # a sub-árvore a partir desse nó pela sub-árvore irmã
    
    graphy = Graphy.copy()
    graphz = Graphz.copy()
    
    sorted_list_graph = list(graphy.nodes())
    sorted_list_graph.sort()
    
    list_dontcares = []
    
    for no in sorted_list_graph:
      if graphy.nodes[no]['weight'] <= valor:
        list_dontcares.append(no)
        
    list_dontcares.sort()
    
    dc_visitados = []
    
    if len(list_dontcares)>0:
      for no_dc in list_dontcares:
        if no_dc not in dc_visitados:
          raio = calcular_profundidade(graphy, no_dc)
          subgrafo_dc_a_eliminar_Y = nx.ego_graph(graphy, no_dc, radius=raio)
          #subgrafo_dc_a_eliminar_Z = nx.ego_graph(graphz, no_dc, radius=raio)
          
          parent_dc = graphy._pred[no_dc]
          if parent_dc:
            children = list(graphy.successors(list(parent_dc)[0]))
            children.remove(no_dc)
            brother = children[0]
            
            pai = list(parent_dc)[0]
            grand_parent = graphy._pred[pai]
            
            
            eliminar_subgraph(graphy, no_dc)
            eliminar_subgraph(graphz, no_dc) 
            
            for n in list(subgrafo_dc_a_eliminar_Y):
              #list_dontcares.remove(n)
              dc_visitados.append(n)
            
            if grand_parent:
              graphy.add_edge(list(grand_parent)[0], brother, weight = graphy[list(grand_parent)[0]][pai]['weight'])
              graphz.add_edge(list(grand_parent)[0], brother, weight = graphz[list(grand_parent)[0]][pai]['weight'])
              
            graphy.remove_edge(pai, brother)
            graphz.remove_edge(pai, brother)
          #sub_list_dc_to_remove = list(subgrafo_dc_a_eliminar_Y)
          
          #if len(sub_list_dc_to_remove)>0:
          #  for noe in sub_list_dc_to_remove:
          #    list_dontcares.remove(noe)
              
          #    #remover edges
        
    return graphy, graphz        

def putting_dont_cares_on_Rz(graphy, graphz, valor):  
  # Encontrar o nó com o valor especificado
  
  sorted_list_graph = list(graphy.nodes())
  sorted_list_graph.sort()
  
  for no in sorted_list_graph:
    if graphy.nodes[no]['weight'] <= valor:
      graphz.nodes[no]['weight'] = graphy.nodes[no]['weight']
    return graphz

def sorted_list_from_graph(graph):
    graph_lista = []
    for no in graph.nodes():
        graph_lista.append(no)
    graph_lista.sort()
    return graph_lista

def build_quantum_circuit_(graph, r_axis, num_qubits):
    
    qc = QuantumCircuit(num_qubits)
    
    paths = find_paths(graph)
    paths = ordenar_paths(paths)
    
    for path in paths:
        
        ctrls = path[:-1]
        ctargets = path[1:]
        qubits = []
        for p in path:
            qubits.append(graph.nodes[p]['level'])
        
        if len(path) == 1:
            qubit_alvo = graph.nodes[path[0]]['level']
            angle = graph.nodes[path[0]]['weight']
            if angle !=0:
                if r_axis == 'y':
                    qc.ry(angle, qubit_alvo)
                if r_axis == 'z':
                    qc.rz(angle, qubit_alvo)
        else:
            
            #target = graph.nodes[path[-1]]['level']
            angle = graph.nodes[path[-1]]['weight']
            if angle !=0:
                num_ctrls = len(ctrls)
                if r_axis == 'y':
                    u = RYGate(angle).control(num_ctrls)
                if r_axis == 'z':
                    u = RZGate(angle).control(num_ctrls)
                
                #adicionar nots para fazer controles abertos
                """
                for c in ctargets:
                    if c%2 == 0:
                        qc.x(graph.nodes[c]['level']-1)
                qc.append(u,qubits)
                for c in ctargets:
                    if c%2 == 0:
                        qc.x(graph.nodes[c]['level']-1)
                """
                """
                for t, c in zip(ctargets, ctrls):
                    if t%2 == 0:
                        qc.x(graph.nodes[c]['level'])
                qc.append(u,qubits)
                for t, c in zip(ctargets, ctrls):
                    if t%2 == 0:
                        qc.x(graph.nodes[c]['level'])
                """
                """         _summary_
                for t, c in zip(ctargets, ctrls):
                    parent_edge = graph.in_edges(t)
                    if parent_edge:
                        if parent_edge[1]['weight'] == -1:
                            qc.x(graph.nodes[c]['level'])
                    qc.append(u,qubits)
                    if parent_edge:
                        if parent_edge[1]['weight'] == -1:
                            qc.x(graph.nodes[c]['level'])    
                
                """
                
                
                for t, c in zip(ctargets, ctrls):
                    #parent_edge = graph.in_edges(t)
                    if graph[c][t]:
                        if graph[c][t]['weight'] == -1:
                            qc.x(graph.nodes[c]['level'])
                qc.append(u,qubits)
                for t, c in zip(ctargets, ctrls):
                    if graph[c][t]:
                        if graph[c][t]['weight'] == -1:
                            qc.x(graph.nodes[c]['level'])
                        
    return qc 

def build_quantum_circuit__(graph, r_axis, num_qubits):
    
    Err = 0.0001
    
    qc = QuantumCircuit(num_qubits)
    
    paths = find_paths(graph)
    paths = ordenar_paths(paths)
    
    for path in paths:
        
        ctrls = path[:-1]
        ctargets = path[1:]
        qubits = []
        for p in path:
            qubits.append(graph.nodes[p]['level'])
        
        if len(path) == 1:
            qubit_alvo = graph.nodes[path[0]]['level']
            angle = graph.nodes[path[0]]['weight']
            if angle !=0:
                if abs(angle - 3.1416)>Err:
                    if r_axis == 'y':
                        qc.ry(angle, qubit_alvo)
                    if r_axis == 'z':
                        qc.rz(angle, qubit_alvo)
                else:   
                    if r_axis == 'y':
                        qc.x(qubit_alvo)
                    if r_axis == 'z':
                        qc.x(qubit_alvo)     
                        
                        
        else:
            
            #target = graph.nodes[path[-1]]['level']
            angle = graph.nodes[path[-1]]['weight']
            if angle !=0:
                num_ctrls = len(ctrls)
                if r_axis == 'y':
                    u = RYGate(angle).control(num_ctrls)
                if r_axis == 'z':
                    u = RZGate(angle).control(num_ctrls)
                for t, c in zip(ctargets, ctrls):
                    #parent_edge = graph.in_edges(t)
                    if graph[c][t]:
                        if graph[c][t]['weight'] == -1:
                            qc.x(graph.nodes[c]['level'])
                if abs(angle - 3.1416)>Err:            
                    qc.append(u,qubits)
                else:
                    qc.mcx(qubits[:-1], qubits[-1])
                for t, c in zip(ctargets, ctrls):
                    if graph[c][t]:
                        if graph[c][t]['weight'] == -1:
                            qc.x(graph.nodes[c]['level'])
                        
    return qc 

def build_quantum_circuit(graph, r_axis, num_qubits):
    
  Err = 0.0001
  
  qc = QuantumCircuit(num_qubits)
  
  paths = find_paths(graph)
  paths = ordenar_paths(paths)
  
  for path in paths:
    ctrls = path[:-1]
    ctargets = path[1:]
    qubits = []
    for p in path:
      qubits.append(graph.nodes[p]['level'])
    
    if len(path) == 1:
      qubit_alvo = graph.nodes[path[0]]['level']
      angle = graph.nodes[path[0]]['weight']
      if angle !=0:
        #if abs(abs(angle) - 3.1416)>Err:
        if r_axis == 'y':
          qc.ry(angle, qubit_alvo)
        if r_axis == 'z':
          qc.rz(angle, qubit_alvo)
        #else:
        #if r_axis == 'y':
        #  qc.x(qubit_alvo)
        #if r_axis == 'z':
        #  qc.rz(angle, qubit_alvo)
    else:
      angle = graph.nodes[path[-1]]['weight']
      #if angle !=0:
      if abs(angle) >= Err:
        num_ctrls = len(ctrls)
        if r_axis == 'y':
          u = RYGate(angle).control(num_ctrls)
        if r_axis == 'z':
          u = RZGate(angle).control(num_ctrls)
        for t, c in zip(ctargets, ctrls):
          if graph[c][t]:
            if graph[c][t]['weight'] == -1:
              qc.x(graph.nodes[c]['level'])
        #if abs(abs(angle) - 3.1416)>Err:
        qc.append(u,qubits)
        #else:
        #  if r_axis == 'y':
        #    qc.mcx(qubits[:-1], qubits[-1])
        #  if r_axis == 'z':
        #    qc.append(u,qubits)
        for t, c in zip(ctargets, ctrls):
          if graph[c][t]:
            if graph[c][t]['weight'] == -1:
              qc.x(graph.nodes[c]['level'])
  return qc 

def build_quantum_circuit_Mozafari_Multi_(graph, num_qubits):
    
    newgraph = graph.copy()
    
    subgrafos = encontrar_subgrafos(newgraph)
    Na = len(subgrafos)
    
    qc = QuantumCircuit(num_qubits+Na)
    
    ancilla = []
    for i in range(Na):
        ancilla.append(i)
        qc.x(i)
    
    for a in ancilla:
        aplicar_ancilla = False
        sgraph = subgrafos[a]
        list_trata_graph = list(sgraph.nodes())
        list_trata_graph.sort()
        
        for node in list_trata_graph:
            if sgraph.nodes[node]['weight'] != 0 and sgraph.nodes[node]['weight'] - 3.1416< 0.0001:
                if tem_filhos(sgraph, node):
                    filhos = list(sgraph.successors(node))
                    if len(filhos) == 1:
                        if sgraph[node][filhos[0]]['weight'] == -1:
                            sgraph.add_node(2*node+1, weight = 0, level = sgraph.nodes[node]['level']+1)
                            sgraph.add_edge(node, 2*node+1, weight = 1)
                        else:
                            sgraph.add_node(2*node, weight = 0, level = sgraph.nodes[node]['level']+1)
                            sgraph.add_edge(node, 2*node, weight = -1)
                    
        display_graph(sgraph)

        sorted_list_graph = list(sgraph.nodes())
        sorted_list_graph.sort()
        
        
        
        #caminhos_por_no = gerar_caminhos_graph_preordem(sgraph, sorted_list_graph[0])
        caminhos_por_no = find_paths_preorder(sgraph, sorted_list_graph[0], weight_preference=1)
        #for no, caminho in caminhos_por_no.items():
        for caminho in caminhos_por_no:
            no = caminho[-1]
            #print(f"Caminho para o nó {no}: {caminho}")
        
            angle = sgraph.nodes[no]['weight']
            if True:#angle !=0:
                if len(caminho) == 1:
                    qubit_alvo = sgraph.nodes[no]['level']+Na
                    
                    if angle !=0:
                      qc.ry(angle, qubit_alvo)
                    
                else:
                    exist_last_parent_one, LP1 = last_parent_one(sgraph, caminho, no)
                    if angle !=0:
                      if exist_last_parent_one and LP1>0:
                          if aplicar_ancilla:
                              my_mcry = MCRy(angle, [a,sgraph.nodes[LP1]['level']+Na],  sgraph.nodes[no]['level']+Na)
                          else:
                              my_mcry = MCRy(angle, [sgraph.nodes[LP1]['level']+Na],  sgraph.nodes[no]['level']+Na)
                          qc.append(my_mcry[0],my_mcry[1])
                      else:
                          if aplicar_ancilla:
                              qc.cry(angle,a,sgraph.nodes[no]['level']+Na)
                          else:
                              qc.ry(angle,sgraph.nodes[no]['level']+Na)
                
                
                    if not(tem_filhos(sgraph,no)) and caminho!=caminhos_por_no[-1]:
                        aplicar_ancilla = True
                        for qctrl, qtrgt in zip(caminho[:-1], caminho[1:]):
                          if sgraph[qctrl][qtrgt]:
                            if sgraph[qctrl][qtrgt]['weight'] == -1:
                              qc.x(sgraph.nodes[qctrl]['level']+Na)
                        list_qubits = []
                        for inode in caminho[:-1]:
                            list_qubits.append(sgraph.nodes[inode]['level']+Na)
                        qc.mcx(list_qubits,a)
                        for qctrl, qtrgt in zip(caminho[:-1], caminho[1:]):
                          if sgraph[qctrl][qtrgt]:
                              if sgraph[qctrl][qtrgt]['weight'] == -1:
                                  qc.x(sgraph.nodes[qctrl]['level']+Na)
                    
    return qc        

def build_quantum_circuit_Mozafari_Multi(graph, num_qubits):
    
    Err = 0.0001
    
    newgraph = graph.copy()
    
    subgrafos = encontrar_subgrafos(newgraph)
    Na = len(subgrafos)
    
    qc = QuantumCircuit(num_qubits+Na)
    
    ancilla = []
    for i in range(Na):
        ancilla.append(i)
        qc.x(i)
    
    for a in ancilla:
        aplicar_ancilla = False
        sgraph = subgrafos[a]
        list_trata_graph = list(sgraph.nodes())
        list_trata_graph.sort()
        
        for node in list_trata_graph:
            if sgraph.nodes[node]['weight'] != 0 and abs(sgraph.nodes[node]['weight'] - 3.1416) > Err:
                if tem_filhos(sgraph, node):
                    filhos = list(sgraph.successors(node))
                    if len(filhos) == 1:
                        if sgraph[node][filhos[0]]['weight'] == -1:
                            sgraph.add_node(2*node+1, weight = 0, level = sgraph.nodes[node]['level']+1)
                            sgraph.add_edge(node, 2*node+1, weight = 1)
                        else:
                            sgraph.add_node(2*node, weight = 0, level = sgraph.nodes[node]['level']+1)
                            sgraph.add_edge(node, 2*node, weight = -1)
                    
        print("\n\n grafo pós-tratamento para MMY")
        display_graph(sgraph)

        sorted_list_graph = list(sgraph.nodes())
        sorted_list_graph.sort()
        
        
        
        #caminhos_por_no = gerar_caminhos_graph_preordem(sgraph, sorted_list_graph[0])
        caminhos_por_no = find_paths_preorder(sgraph, sorted_list_graph[0], weight_preference=1)
        #for no, caminho in caminhos_por_no.items():
        for caminho in caminhos_por_no:
            no = caminho[-1]
            #print(f"Caminho para o nó {no}: {caminho}")
        
            angle = sgraph.nodes[no]['weight']
            if True:#angle !=0:
                if len(caminho) == 1:
                    qubit_alvo = sgraph.nodes[no]['level']+Na
                    
                    if angle !=0:
                      qc.ry(angle, qubit_alvo)
                    
                else:
                    exist_last_parent_one, LP1 = last_parent_one(sgraph, caminho, no)
                    if angle !=0:
                        if abs(angle - 3.1416)>Err:
                            if exist_last_parent_one and LP1>0:
                                if aplicar_ancilla:
                                    my_mcry = MCRy(angle, [a,sgraph.nodes[LP1]['level']+Na],  sgraph.nodes[no]['level']+Na)
                                else:
                                    my_mcry = MCRy(angle, [sgraph.nodes[LP1]['level']+Na],  sgraph.nodes[no]['level']+Na)
                                qc.append(my_mcry[0],my_mcry[1])
                            else:
                                if aplicar_ancilla:
                                    qc.cry(angle,a,sgraph.nodes[no]['level']+Na)
                                else:
                                    qc.ry(angle,sgraph.nodes[no]['level']+Na)
                        else:
                            if exist_last_parent_one and LP1>0:
                                if aplicar_ancilla:
                                    qc.mcx([a,sgraph.nodes[LP1]['level']+Na],sgraph.nodes[no]['level']+Na)
                                else:
                                    qc.cx(sgraph.nodes[LP1]['level']+Na,sgraph.nodes[no]['level']+Na)
                            else:
                                if aplicar_ancilla:
                                    qc.ccx(a,sgraph.nodes[LP1]['level']+Na, sgraph.nodes[no]['level']+Na)
                                else:
                                    qc.cx(sgraph.nodes[LP1]['level']+Na, sgraph.nodes[no]['level']+Na)
                        
                
                    if not(tem_filhos(sgraph,no)) and caminho!=caminhos_por_no[-1]:
                        aplicar_ancilla = True
                        for qctrl, qtrgt in zip(caminho[:-1], caminho[1:]):
                          if sgraph[qctrl][qtrgt]:
                            if sgraph[qctrl][qtrgt]['weight'] == -1:
                              qc.x(sgraph.nodes[qctrl]['level']+Na)
                        list_qubits = []
                        for inode in caminho[:-1]:
                            list_qubits.append(sgraph.nodes[inode]['level']+Na)
                        qc.mcx(list_qubits,a)
                        for qctrl, qtrgt in zip(caminho[:-1], caminho[1:]):
                          if sgraph[qctrl][qtrgt]:
                              if sgraph[qctrl][qtrgt]['weight'] == -1:
                                  qc.x(sgraph.nodes[qctrl]['level']+Na)
                    
    return qc        

def find_paths(graph):
    node_list = list(graph)
    node_list.sort()
    list_paths = []
    if node_list:
        list_paths.append([node_list[0]])
        for i,no_1 in enumerate(node_list):
            parent = list(graph.predecessors(no_1))
            if not parent:
                for no_2 in node_list[1:]:#[i+1:]:
                    if nx.has_path(graph, no_1, no_2):
                        for path in nx.all_simple_paths(graph, no_1, no_2):
                            list_paths.append(list(path))
                            
    return list_paths

def ordenar_paths(paths):
    # Ordena a lista primeiro pelo tamanho dos subelementos e depois pelos valores dos subelementos
    paths_ordenado = sorted(paths, key=lambda x: (len(x), x[-1]))
    #def custom_sort(lst):
    # Ordena primeiro pelo valor do último subelemento e depois pelo tamanho do subelemento
    return sorted(paths, key=lambda x: (x[-1], len(x)))
    
    #return paths_ordenado
            
def encontrar_subgrafos(graph):
    sorted_list_graph = list(graph.nodes())
    sorted_list_graph.sort()
    subgraph_list = []
    #colcoar pra remover todos os nós de um subgrafo que pertencem a um
    #subgrafo que os contém
    for node in sorted_list_graph:
        parent = list(graph.predecessors(node))
        if not(parent):
            raio = calcular_profundidade(graph, node)
            subgraph = nx.ego_graph(graph, node, radius=raio)
            subgraph_list.append(subgraph)
            lista_filtrada = [elem for elem in sorted_list_graph if elem not in list(subgraph)]
            sorted_list_graph = lista_filtrada
    return subgraph_list

def find_paths_preorder(graph, start_node, weight_preference=1):
  """
  Encontra todos os caminhos em pré-ordem, priorizando arestas com o peso especificado.

  Args:
    graph: O grafo em formato de dicionário.
    start_node: O nó inicial para a busca.
    weight_preference: O peso das arestas que devem ser priorizadas.

  Returns:
    Uma lista de caminhos em pré-ordem.
  """

  def _dfs(current_node, visited, current_path):
    visited.add(current_node)
    paths.append(current_path + [current_node])

    #neighbors = graph.get(current_node, {})
    #neighbors = list(graph.successors(current_node))
    # Prioriza vizinhos com peso preferido
    #neighboritems[]
    #for i in neighbors:
    #neighborsitems.append(graph[i])
    neighbors = graph[current_node].items()
    preferred_neighbors = sorted(
        [(neighbor, data) for neighbor, data in neighbors if data.get('weight') == weight_preference],
        key=lambda x: x[0])
    other_neighbors = sorted(
        [(neighbor, data) for neighbor, data in neighbors if data.get('weight') != weight_preference],
        key=lambda x: x[0])

    for neighbor, data in preferred_neighbors + other_neighbors:
      if neighbor not in visited:
        _dfs(neighbor, set(visited), current_path + [current_node])

  paths = []
  _dfs(start_node, set(), [])
  return paths

def last_parent_one(graph, path, node):
    Exist_lp1 = False
    new_path = path[:node+1]
    invpath = new_path[::-1]
    for lpo in range(len(invpath[:-1])):
        if graph[invpath[lpo+1]][invpath[lpo]]['weight']==1:
            Exist_lp1 = True
            return Exist_lp1, invpath[lpo+1]
    return Exist_lp1, 0

from qiskit.circuit.library import RYGate
def MCRy(angle, list_controls, trgt):
    """_summary_

    Args:
        angle (_type_): _description_
        list_controls (_type_): _description_
        trgt (_type_): _description_

    Returns:
        u, list_controls_
        
    Uso: my_mcry = MCRy(angle, list_controls,  target)
         qc.append( my_mcry[0],my_mcry[1])
    """
    num_ctrls = len(list_controls)
    u = RYGate(angle).control(num_ctrls)
    list_controls.append(trgt)
    u.name = "MCRy"
    u.label = "MCRy"
    u.num_ctrl_qubits = num_ctrls
    u.num_qubits = num_ctrls + 1
    return  u,list_controls

def subtract_subgraphs_com_pesos_nas_arestas_(graph):
    """_summary_
    Se nó  tem irmão, e ambos valores não-zero então:
    ----o------*-----     ----o------*----*------     --------------*------
        |      |       =      |      |    |        =                |
    ----A------B-----     ----A------A---B-A-----     ----A--------B-A-----
    
    Para o Nó A:
    - elimina edge (pai_de_A, A)
    - se avô, adiciona (avo_de_A, A)
    
    Para o nó B, simplesmente faz weight_B = weight_B - weight_A 
    
    
    Primeiro realizar todas as subtrações, depois remover as arestas
    Args:
        graph (_type_): _description_
    """
    
    graph = graph.copy()
    
    sorted_list_graph = list(graph.nodes())
    #sorted_list_graph.sort(reverse = True)
    sorted_list_graph.sort()
    
         
        
        
    for  no_1 in sorted_list_graph[1:]:
        if no_1%2==0:
            no_2 = no_1+1
            if no_2 in sorted_list_graph:
                if graph.nodes[no_1]['level'] == graph.nodes[no_2]['level']:
                    parent_1 = graph._pred[no_1]
                    parent_2 = graph._pred[no_2]
                    pai_1 = []
                    pai_2 = []
                    
                    raio = calcular_profundidade(graph, no_1)
                    sub_1 = nx.ego_graph(graph, no_1, radius=raio)
                    sub_2 = nx.ego_graph(graph, no_2, radius=raio)
                    
                    list_sub_1 = list(sub_1)
                    list_sub_1.sort()
                    
                    list_sub_2 = list(sub_2)
                    list_sub_2.sort()
                    
                    for n1, n2 in zip(list_sub_1, list_sub_2):
                        graph.nodes[n2]['weight'] = graph.nodes[n2]['weight'] - graph.nodes[n1]['weight']

                    
                    if parent_1:
                        pai_1 = list(parent_1)[0]
                        if True:#parent_2:
                            pai_2 = list(parent_2)[0]
                            if True:#pai_1 == pai_2:
                                #if graph.has_edge(pai_1, no_1):
                                            
                                if True:#graph[pai_1][no_1]['weight']==-1 :
                                    #if no_1 % 2==0:
                                    graph.remove_edge(pai_1, no_1)
                                    grandparent = graph._pred[pai_1]
                                    if grandparent:
                                        grandparent_edge = graph[list(grandparent)[0]][pai_1]['weight']#grandparent[1]['weight']
                                        graph.add_edge(list(grandparent)[0], no_1, weight = grandparent_edge)
                                                
                            
                                #for n1, n2 in zip(list_sub_1, list_sub_2):
                                #    graph.nodes[n2]['weight'] = graph.nodes[n2]['weight'] - graph.nodes[n1]['weight']
                                #graph.nodes[no_2]['weight'] = graph.nodes[no_2]['weight'] - graph.nodes[no_1]['weight']
                                #while graph.nodes[no_2]['weight'] < 0:
                                #    graph.nodes[no_2]['weight']  = graph.nodes[no_2]['weight'] + 2*np.pi
    return graph                                    

def subtract_subgraphs_com_pesos_nas_arestas(graph):
    """_summary_
    Se nó  tem irmão, e ambos valores não-zero então:
    ----o------*-----     ----o------*----*------     --------------*------
        |      |       =      |      |    |        =                |
    ----A------B-----     ----A------A---B-A-----     ----A--------B-A-----
    
    Para o Nó A:
    - elimina edge (pai_de_A, A)
    - se avô, adiciona (avo_de_A, A)
    
    Para o nó B, simplesmente faz weight_B = weight_B - weight_A 
    
    
    Primeiro realizar todas as subtrações, depois remover as arestas
    Args:
        graph (_type_): _description_
    """
    
    graph = graph.copy()
    
    sorted_list_graph = list(graph.nodes())
    sorted_list_graph.sort()
    
    display_graph(graph)    
    for  no_pai in sorted_list_graph:
        list_childs = list(graph.successors(no_pai))
        list_childs.sort()
        for indexnode, no_1 in enumerate(list_childs[:-1]):
            for no_2 in list_childs[indexnode+1:]:
                if graph.nodes[no_1]['level'] == graph.nodes[no_2]['level']:
                    if graph[no_pai][no_1]['weight']==-graph[no_pai][no_2]['weight']:
                    
                        raio = calcular_profundidade(graph, no_1)
                        sub_1 = nx.ego_graph(graph, no_1, radius=raio)
                        sub_2 = nx.ego_graph(graph, no_2, radius=raio)
                        
                        list_sub_1 = list(sub_1)
                        list_sub_1.sort()
                        
                        list_sub_2 = list(sub_2)
                        list_sub_2.sort()
                        
                        for n1, n2 in zip(list_sub_1, list_sub_2):
                            graph.nodes[n2]['weight'] = graph.nodes[n2]['weight'] - graph.nodes[n1]['weight']
                            
    display_graph(graph) 
    reverse_list = sorted_list_graph
    reverse_list.sort(reverse = True)
    for filho in reverse_list:
        parent = graph._pred[filho]
        if parent:
            pai = list(parent)[0]
            if graph[pai][filho]['weight']==-1:       
                grandparent = graph._pred[pai]
                
                if grandparent:
                    avo = list(grandparent)[0]
                    edge_grandparent = graph[avo][pai]['weight']
                    graph.add_edge(list(grandparent)[0], filho, weight = edge_grandparent)
                graph.remove_edge(pai, filho)
    display_graph(graph)
                     
    
    
                            
    """            
    grandparent = graph._pred[no_pai]
    if grandparent:
        grandparent_edge = graph[list(grandparent)[0]][no_pai]['weight']
        graph.add_edge(list(grandparent)[0], no_1, weight = grandparent_edge)
        
    graph.remove_edge(no_pai, no_1)            
    """
#display_graph(graph)
    #for n1, n2 in zip(list_sub_1, list_sub_2):
    #    graph.nodes[n2]['weight'] = graph.nodes[n2]['weight'] - graph.nodes[n1]['weight']
    #graph.nodes[no_2]['weight'] = graph.nodes[no_2]['weight'] - graph.nodes[no_1]['weight']
    #while graph.nodes[no_2]['weight'] < 0:
    #    graph.nodes[no_2]['weight']  = graph.nodes[no_2]['weight'] + 2*np.pi
    return graph                                    

def subtract_brothers_nodes(graph):
    """_summary_
    Se nó  tem irmão, e ambos valores não-zero então:
    ----o------*-----     ----o------*----*------     --------------*------
        |      |       =      |      |    |        =                |
    ----A------B-----     ----A------A---B-A-----     ----A--------B-A-----
    
    Para o Nó A:
    - elimina edge (pai_de_A, A)
    - se avô, adiciona (avo_de_A, A)
    
    Para o nó B, simplesmente faz weight_B = weight_B - weight_A 
    
    
    Primeiro realizar todas as subtrações, depois remover as arestas
    Args:
        graph (_type_): _description_
    """
    
    graph = graph.copy()
    
    sorted_list_graph = list(graph.nodes())
    sorted_list_graph.sort()
    
    display_graph(graph)    
    for  no_pai in sorted_list_graph:
        list_childs = list(graph.successors(no_pai))
        list_childs.sort()
        for indexnode, no_1 in enumerate(list_childs[:-1]):
            for no_2 in list_childs[indexnode+1:]:
                if graph.nodes[no_1]['level'] == graph.nodes[no_2]['level']:
                    if graph[no_pai][no_1]['weight'] ==- graph[no_pai][no_2]['weight']:
                    
                        raio1 = calcular_profundidade(graph, no_1)
                        raio2 = calcular_profundidade(graph, no_2)
                        sub_1 = nx.ego_graph(graph, no_1, radius=raio1)
                        sub_2 = nx.ego_graph(graph, no_2, radius=raio2)
                        
                        if raio1==raio2:
                          
                          list_sub_1 = list(sub_1)
                          list_sub_1.sort()
                          
                          list_sub_2 = list(sub_2)
                          list_sub_2.sort()
                          
                          for n1, n2 in zip(list_sub_1, list_sub_2):
                              graph.nodes[n2]['weight'] = graph.nodes[n2]['weight'] - graph.nodes[n1]['weight']
                              grandparent = graph._pred[no_pai]
                              if grandparent:
                                  grandparent_edge = graph[list(grandparent)[0]][no_pai]['weight']
                                  graph.add_edge(list(grandparent)[0], no_1, weight = grandparent_edge)
                                  
                              graph.remove_edge(no_pai, no_1) 
                    
                    
                    
                    #if not (tem_filhos(graph, no_1) or tem_filhos(graph, no_2)):
                    #  if graph[no_pai][no_1]['weight'] == graph[no_pai][no_2]['weight']:
                    #    graph.nodes[no_1]['weight'] = graph.nodes[no_1]['weight']+graph.nodes[no_2]['weight']
                    #    graph.remove(no_2)
                              
    display_graph(graph) 
    reverse_list = sorted_list_graph
    reverse_list.sort(reverse = True)
    """
    for filho in reverse_list:
        parent = graph._pred[filho]
        if parent:
            pai = list(parent)[0]
            if graph[pai][filho]['weight']==-1:       
                grandparent = graph._pred[pai]
                
                if grandparent:
                    avo = list(grandparent)[0]
                    edge_grandparent = graph[avo][pai]['weight']
                    graph.add_edge(list(grandparent)[0], filho, weight = edge_grandparent)
                graph.remove_edge(pai, filho)
    display_graph(graph)
    """                 
    
    
                            
    """            
    grandparent = graph._pred[no_pai]
    if grandparent:
        grandparent_edge = graph[list(grandparent)[0]][no_pai]['weight']
        graph.add_edge(list(grandparent)[0], no_1, weight = grandparent_edge)
        
    graph.remove_edge(no_pai, no_1)            
    """
    #display_graph(graph)
    #for n1, n2 in zip(list_sub_1, list_sub_2):
    #    graph.nodes[n2]['weight'] = graph.nodes[n2]['weight'] - graph.nodes[n1]['weight']
    #graph.nodes[no_2]['weight'] = graph.nodes[no_2]['weight'] - graph.nodes[no_1]['weight']
    #while graph.nodes[no_2]['weight'] < 0:
    #    graph.nodes[no_2]['weight']  = graph.nodes[no_2]['weight'] + 2*np.pi
    return graph                                    

def get_state_tree(state, Num_qubits):
    global state_tree
    data = [Amplitude(i, a) for i, a in enumerate(state)]
    state_tree = state_decomposition(Num_qubits, data)
    return state_tree
def gen_graphs_Ry_and_RZ(state_tree):
  """_summary_

  Args:
      state (complex): amplitude vector
      Num_qubits (int): Number qubits 

  Returns:
      _type_: Angles graphs for Ry and Ry, and the first state tree element
      as the global phase
              
  """


  if state_tree.arg != 0:
    #print(state_tree.arg)
    global_phase = state_tree.arg
  else:
    global_phase = 0

  original_angle_tree = create_angles_tree(state_tree)
  Ry_angles_graph =  tree_to_graph_y_weighted_edges(original_angle_tree)
  Rz_angles_graph =  tree_to_graph_z_weighted_edges(original_angle_tree)
  
  return Ry_angles_graph, Rz_angles_graph, global_phase

"""
def gen_graphs_Ry_and_RZ_with_dont_cares(state_tree):
    _summary_

    Args:
        state (complex): amplitude vector
        Num_qubits (int): Number qubits

    Returns:
        _type_: Angles graphs for Ry and Ry, and the first state tree element
        as the global phase

    

    if state_tree.arg != 0:
        print(state_tree.arg)
        global_phase = state_tree.arg
    else:
        global_phase = 0

    angle_tree, dc, dcz = mycreate_angles_tree(state_tree)
    Ry_angles_graph = tree_to_graph_y_weighted_edges(angle_tree)
    Rz_angles_graph = tree_to_graph_z_weighted_edges(angle_tree)

    return Ry_angles_graph, Rz_angles_graph, global_phase, dc, dcz
"""
def Build_all_circuits(Num_qubits, Gy_without_reduction, Gz_without_reduction, global_phase, show_circuits = True):
  """_summary_

  Args:
      Num_qubits (int): number of qubits of the system
      Gy_without_reduction (graph): Ry angles graph without any reduction
      Gz_without_reduction (_type_): _description_
      global_phase (_type_): _description_
      show_circuits (bool, optional): _description_. Defaults to True.

  Returns:
      _type_: _description_
  """
  qc_Ry = build_quantum_circuit(Gy_without_reduction,'y', Num_qubits)
  qc_Rz = build_quantum_circuit(Gz_without_reduction,'z', Num_qubits)
  qc_qsp_complete = QuantumCircuit.compose(qc_Ry, qc_Rz)
  qc_qsp_complete.global_phase = global_phase
  """
  if show_circuits:
    print("\n Ry Circuit:\n")
    print(qc_Ry)
    print("\n Rz Circuit:\n")
    print(qc_Rz)
    print("\n Complete Circuit:\n")
    print(qc_qsp_complete)
  """
  return qc_Ry,qc_Rz,qc_qsp_complete

def Norm_State_and_Num_Qubits(state):
  Num_qubits = int(np.log2(len(state)))
  state = state/np.linalg.norm(state)
  return state, Num_qubits

def show_all_graphs(name, G1,G2,show):
    if show == True:
        print(name)
        display_graph(G1)
        display_graph(G2)

def merge_each_graph(G1, G2):
  #global houve_mesclagem, G1, G2
  print("Mesclagem:")
  houve_mesclagem = True
  while houve_mesclagem: G1, houve_mesclagem = merge_subgraphs(G1)
  houve_mesclagem = True
  while houve_mesclagem: G2, houve_mesclagem = merge_subgraphs(G2)
  return G1, G2

def print_states_from_circuit(name, state_vector, qc, show):
    if show == True:
      params = get_state(qc.reverse_bits())

      print("\n state: \n")
      print(np.round(state_vector, 4), "\n")
      print("\n abs state")
      print(np.round(abs(state_vector), 4), "\n")
      print("\n state from ", name, ":")
      print(np.round(params, 4), "\n")
      print("\n abs state from ", name, ":")
      print(np.round(abs(params), 4))
      print("\n")

def get_global_phase(state_tree):
  if state_tree.arg != 0:
    print(state_tree.arg)
    global_phase = state_tree.arg
  else:
    global_phase = 0
  return global_phase

def remove_last_node_zero_Gy_and_Gz(G1, G2):
  G1 = remove_last_node_zero(G1)
  G2 = remove_last_node_zero(G2)
  return G1, G2

def remove_zero_root_node_Gy_and_Gz(G1, G2):
  G1 = eliminar_primeiro_zero(G1)
  G2 = eliminar_primeiro_zero(G2)
  return G1, G2

def show_circuits(name, qc_ry, qc_rz, qc_qsp, show):
  if show == True:
      print(name)
      print()
      print("Ry quantum circuit:")
      print(qc_ry)
      print()
      print("Rz quantum circuit:")
      print(qc_rz)
      print()
      print("Complete quantum circuit\n")
      print(qc_qsp)

def circuit_transpile_and_counts(state, qc_Ry_original, qc_Ry_reduzido_dc_noe, qc_Rz_original, qc_Rz_reduzido_dc_noe, show ):
  if show:
      RY_original_transpiled = transpile(qc_Ry_original, basis_gates=['u', 'cx'], optimization_level=0)
      RY_depth_original = RY_original_transpiled.depth()
      RY_cx_original = RY_original_transpiled.count_ops().get('cx', 0)
      #RY_reduced_transpiled = transpile(qc_Ry_reduzido, basis_gates=['u', 'cx'], optimization_level=0)
      #RY_depth_reduced = RY_reduced_transpiled.depth()
      #RY_cx_reduced = RY_reduced_transpiled.count_ops().get('cx', 0)
      #RY_reduced_dc_transpiled = transpile(qc_Ry_reduzido_dc, basis_gates=['u', 'cx'], optimization_level=0)
      #RY_depth_reduced_dc = RY_reduced_dc_transpiled.depth()
      #RY_cx_reduced_dc = RY_reduced_dc_transpiled.count_ops().get('cx', 0)
      RY_reduced_dc_transpiled_noe = transpile(qc_Ry_reduzido_dc_noe, basis_gates=['u', 'cx'], optimization_level=0)
      RY_depth_reduced_dc_noe = RY_reduced_dc_transpiled_noe.depth()
      RY_cx_reduced_dc_noe = RY_reduced_dc_transpiled_noe.count_ops().get('cx', 0)
      RZ_original_transpiled = transpile(qc_Rz_original, basis_gates=['u', 'cx'], optimization_level=0)
      RZ_depth_original = RZ_original_transpiled.depth()
      RZ_cx_original = RZ_original_transpiled.count_ops().get('cx', 0)
      #RZ_reduced_transpiled = transpile(qc_Rz_reduzido, basis_gates=['u', 'cx'], optimization_level=0)
      #RZ_depth_reduced = RZ_reduced_transpiled.depth()
      #RZ_cx_reduced = RZ_reduced_transpiled.count_ops().get('cx', 0)
      #RZ_reduced_dc_transpiled = transpile(qc_Rz_reduzido_dc, basis_gates=['u', 'cx'], optimization_level=0)
      #RZ_depth_reduced_dc = RZ_reduced_dc_transpiled.depth()
      #RZ_cx_reduced_dc = RZ_reduced_dc_transpiled.count_ops().get('cx', 0)
      RZ_reduced_dc_transpiled_noe = transpile(qc_Rz_reduzido_dc_noe, basis_gates=['u', 'cx'], optimization_level=0)
      RZ_depth_reduced_dc_noe = RZ_reduced_dc_transpiled_noe.depth()
      RZ_cx_reduced_dc_noe = RZ_reduced_dc_transpiled_noe.count_ops().get('cx', 0)

      print()
      print("----Ry----")
      print()
      print("CNOT gates counts:")
      print("Original Circuit", RY_cx_original)
      #print("Reduced Circuit", RY_cx_reduced)
      #print("Reduced Circuit DC", RY_cx_reduced_dc)
      print("Reduced Circuit", RY_cx_reduced_dc_noe)
      print()
      print("Depth:")
      print("Original Circuit", RY_depth_original)
      #print("Reduced Circuit", RY_depth_reduced)
      #print("Reduced Circuit DC", RY_depth_reduced_dc)
      print("Reduced Circuit", RY_depth_reduced_dc_noe)
      print()
      print("----Rz----")
      print()
      print("CNOT gates counts:")
      print("Original Circuit", RZ_cx_original)
      #print("Reduced Circuit", RZ_cx_reduced)
      #print("Reduced Circuit DC", RZ_cx_reduced_dc)
      print("Reduced Circuit", RZ_cx_reduced_dc_noe)
      print()
      print("Depth:")
      print("Original Circuit", RZ_depth_original)
      #print("Reduced Circuit", RZ_depth_reduced)
      #print("Reduced Circuit DC", RZ_depth_reduced_dc)
      print("Reduced Circuit", RZ_depth_reduced_dc_noe)
      print()
      print("----Complete Circuit----")
      print()
      print("CNOT gates counts:")
      print("Original Circuit", RY_cx_original+RZ_cx_original)
      #print("Reduced Circuit", RY_cx_reduced+RZ_cx_reduced)
      #print("Reduced Circuit DC", RY_cx_reduced_dc+RZ_cx_reduced_dc)
      print("Reduced Circuit", RY_cx_reduced_dc_noe+RZ_cx_reduced_dc_noe)
      print()
      print("Depth:")
      print("Original Circuit", RY_depth_original+RZ_depth_original)
      #print("Reduced Circuit", RY_depth_reduced+RZ_depth_reduced)
      #print("Reduced Circuit DC", RY_depth_reduced_dc+RZ_depth_reduced_dc)
      print("Reduced Circuit", RY_depth_reduced_dc_noe+RZ_depth_reduced_dc_noe)
      print()

      total_cnots_original = RY_cx_original+RZ_cx_original
      #total_cnots_reduced = RY_cx_reduced+RZ_cx_reduced
      #total_cnots_dc = RY_cx_reduced_dc+RZ_cx_reduced_dc
      total_cnots_dc_noe = RY_cx_reduced_dc_noe + RZ_cx_reduced_dc_noe

      total_depth_original = RY_depth_original+RZ_depth_original
      #total_depth_reduced = RY_depth_reduced+RZ_depth_reduced
      #total_depth_dc = RY_depth_reduced_dc+RZ_depth_reduced_dc
      total_depth_dc_noe = RY_depth_reduced_dc_noe+RZ_depth_reduced_dc_noe

      print("Sparsity:", 100 * (1 - np.count_nonzero(state) / len(state)), "%")
      print("---Total Reduction of Original Circuit:---")
      print("Cnots gate:")
      #print("Reduced:", np.round(100*(total_cnots_original-total_cnots_reduced)/total_cnots_original,2),"%")
      #print("Reduced dc:",np.round(100*(total_cnots_original-total_cnots_dc)/total_cnots_original,2),"%" )
      print("Reduced Circuit:", np.round(100*(total_cnots_original-total_cnots_dc_noe)/total_cnots_original,2),"%")
      print("Circuit depth:")
      #print("Reduced:", np.round(100*(total_depth_original-total_depth_reduced)/total_depth_original,2),"%")
      #print("Reduced dc:", np.round(100*(total_depth_original-total_depth_dc)/total_depth_original,2),"%" )
      print("Reduced Circuit:", np.round(100*(total_depth_original-total_depth_dc_noe)/total_depth_original, 2),"%")


"""
if subtract_graphs:
  print("subtract graphs")
  #Gy = subtract_subgraphs_com_pesos_nas_arestas(Gy)
  #Gz = subtract_subgraphs_com_pesos_nas_arestas(Gz)
  Gy = subtract_brothers_nodes(Gy)
  Gz = subtract_brothers_nodes(Gz)
  
  #if exibir_todos_os_grafos: 
  display_graph(Gy)
  display_graph(Gz)
"""

"""
subtrair subgrafos
1 - lista todos os nós que não são folhas, ou seja, são pais
2 - inverte a lista, ordem decrescente, pois vamos visitar a partir de baixo
3 - para cada subgrafo sob aresta zero do pai, eleva a aresta pai para o avô
4 - para cada subgrafo sob aresta 1 do pai, soubtrai os nós do subgrafo irmão (esquerdo)

"""

def subtract_subgraphs_brothers_down_top(graph):
    graph = graph.copy()

    sorted_list_graph = list(graph.nodes())
    sorted_list_graph.sort(reverse=True)

    #remove leafs
    for node in sorted_list_graph:
        if not(tem_filhos(graph, node)):
            sorted_list_graph.remove(node)

    #display_graph(graph)

    for no_pai in sorted_list_graph:
        list_childs = list(graph.successors(no_pai))
        list_childs.sort()
        for indexnode, no_1 in enumerate(list_childs[:-1]):
            for no_2 in list_childs[indexnode + 1:]:
                if graph.nodes[no_1]['level'] == graph.nodes[no_2]['level']: #same level?
                    if graph[no_pai][no_1]['weight'] == - graph[no_pai][no_2]['weight']: #different values edges?

                        raio1 = calcular_profundidade(graph, no_1)
                        raio2 = calcular_profundidade(graph, no_2)
                        sub_1 = nx.ego_graph(graph, no_1, radius=raio1)
                        sub_2 = nx.ego_graph(graph, no_2, radius=raio2)

                        if raio1 == raio2:

                            list_sub_1 = list(sub_1)
                            list_sub_1.sort()

                            list_sub_2 = list(sub_2)
                            list_sub_2.sort()

                            for n1, n2 in zip(list_sub_1, list_sub_2):
                                graph.nodes[n2]['weight'] = graph.nodes[n2]['weight'] - graph.nodes[n1]['weight']
                                grandparent = graph._pred[no_pai]
                                if grandparent:
                                    grandparent_edge = graph[list(grandparent)[0]][no_pai]['weight']
                                    graph.add_edge(list(grandparent)[0], no_1, weight=grandparent_edge)

                                if graph.has_edge(no_pai,no_1):

                                    graph.remove_edge(no_pai, no_1)

                                # if not (tem_filhos(graph, no_1) or tem_filhos(graph, no_2)):
                    #  if graph[no_pai][no_1]['weight'] == graph[no_pai][no_2]['weight']:
                    #    graph.nodes[no_1]['weight'] = graph.nodes[no_1]['weight']+graph.nodes[no_2]['weight']
                    #    graph.remove(no_2)

    #display_graph(graph)
    reverse_list = sorted_list_graph
    reverse_list.sort(reverse=True)
    """
    for filho in reverse_list:
        parent = graph._pred[filho]
        if parent:
            pai = list(parent)[0]
            if graph[pai][filho]['weight']==-1:       
                grandparent = graph._pred[pai]

                if grandparent:
                    avo = list(grandparent)[0]
                    edge_grandparent = graph[avo][pai]['weight']
                    graph.add_edge(list(grandparent)[0], filho, weight = edge_grandparent)
                graph.remove_edge(pai, filho)
    display_graph(graph)
    """

    """            
    grandparent = graph._pred[no_pai]
    if grandparent:
        grandparent_edge = graph[list(grandparent)[0]][no_pai]['weight']
        graph.add_edge(list(grandparent)[0], no_1, weight = grandparent_edge)

    graph.remove_edge(no_pai, no_1)            
    """
    # display_graph(graph)
    # for n1, n2 in zip(list_sub_1, list_sub_2):
    #    graph.nodes[n2]['weight'] = graph.nodes[n2]['weight'] - graph.nodes[n1]['weight']
    # graph.nodes[no_2]['weight'] = graph.nodes[no_2]['weight'] - graph.nodes[no_1]['weight']
    # while graph.nodes[no_2]['weight'] < 0:
    #    graph.nodes[no_2]['weight']  = graph.nodes[no_2]['weight'] + 2*np.pi
    return graph
