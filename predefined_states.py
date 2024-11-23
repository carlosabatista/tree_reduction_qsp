import numpy as np

def MMY_real_state():
  """_summary_
    Real state example in Efficient Deterministic Preparation of Quantum States Using Decision Diagrams
    Fereshte Mozafari, Giovanni De Micheli, Yuxiang Yang - 2022
    https://arxiv.org/abs/2206.08588
  
  """
  state = np.zeros(16)
  state[0] = np.sqrt(0.5)
  state[2] = np.sqrt(0.5)
  state[9] = np.sqrt(2.0)
  state[14] = np.sqrt(1.0)
    
  return state
    
def MMY_complex_state():
  """_summary_
    State example in fficient Deterministic Preparation of Quantum States Using Decision Diagrams
    Fereshte Mozafari, Giovanni De Micheli, Yuxiang Yang - 2022 - replacing real amplitudes with complex amplitudes
    https://arxiv.org/abs/2206.08588 
  """
  state = np.zeros(16, dtype=complex)
  state[0] = np.sqrt(0.5)*np.exp(1j)
  state[2] = np.sqrt(0.5)*np.exp(1j)
  state[9] = np.sqrt(2.0)*np.exp(2j)
  state[14] = np.sqrt(1.0)*np.exp(1j)
   
  return state

def Real_Uniform_State_0x239C():
  state = np.ones(16)

  state[2] = 0
  state[3] = 0
  state[9] = 0
  state[13] = 0
  
  return state
  
def GHZ(Num_Qubits):
  state = np.zeros(2**Num_Qubits)
  state[0]=1
  state[-1]=1
  
def Norm_state_and_Num_Qubits(state):
  Num_qubits = int(np.log2(len(state)))
  state = state/np.linalg.norm(state)
  return state, Num_qubits

def Sparse_4qubits_0xC890():
  state = np.array([np.sqrt(0.2)*np.exp(2.1j), np.sqrt(0.3)*np.exp(0.5j),  0., 0., np.sqrt(0.1)*np.exp(1.1j), 0., 0., 0., np.sqrt(0.3)*np.exp(1.4j), 0., 0., np.sqrt(0.1)*np.exp(1.7j), 0., 0., 0., 0.])
  return state

def Separable_States_Qubits():
  s1 =  np.array([np.sqrt(0.3)*np.exp(1j*np.pi/1.5), np.sqrt(0.7)*np.exp(1j*np.pi/8.5)]) 
  s2 =  np.array([np.sqrt(0.1)*np.exp(1j*np.pi/4.0), np.sqrt(0.9)*np.exp(1j*np.pi/4.5)])
  s3 =  np.array([np.sqrt(0.4)*np.exp(1j*np.pi/2.7), np.sqrt(0.2)*np.exp(1j*np.pi/3.5), np.sqrt(0.3)*np.exp(1j*np.pi/4), np.sqrt(0.1)*np.exp(1j*np.pi/1.5)]) 
  s4 =  np.array([np.sqrt(0.2)*np.exp(1j*np.pi/1.3), np.sqrt(0.7)*np.exp(1j*np.pi/4.5), np.sqrt(0.35)*np.exp(1j*np.pi/3),  np.sqrt(0.75)*np.exp(1j*np.pi/5.4)]) 

  state = np.kron(s1,s2)
  state = np.kron(state,s3)
  #state = np.kron(state,s4)
  
  return state

#https://arxiv.org/pdf/2409.01418
def State_Hanyu_Wang():
  state = np.zeros(8)
  norm_factor = 1./np.sqrt(8)

  state[0] = np.sqrt(2)
  state[2] = 1
  state[3] = 1
  state[4] = -1
  state[5] = 1
  state[7] = np.sqrt(2)
  state = state/norm_factor

  return state

