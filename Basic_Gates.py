
# general imports
import numpy as np  # type: ignore
from scipy import linalg # type: ignore # for exponential of a matrix
import matplotlib.pyplot as plt # type: ignore






def qubit_rot(theta):
    return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                     [-1j*np.sin(theta/2), np.cos(theta/2)]])

def GZ(theta):
    return np.array([[np.exp(-1j*theta/2), 0],
                     [0,  np.exp(1j*theta/2)]])

def interaction_gate(theta):
    return np.array([[np.cos(theta/4), 0, 0, -1j*np.sin(theta/4)],
                     [0, np.cos(theta/4), -1j*np.sin(theta/4), 0],
                     [0, -1j*np.sin(theta/4), np.cos(theta/4), 0],
                     [-1j*np.sin(theta/4), 0, 0, np.cos(theta/4)]])

def qubit_rotQSP(x):
    return np.array([[np.cos(x), 1j*np.sin(x)],
                     [1j*np.sin(x), np.cos(x)]])

def Z_rot(theta):
    return np.array([[np.exp(-1j*theta/2), 0],
                     [0,  np.exp(1j*theta/2)]])

def SingValT(Angles_SVT,x):
    #Z=np.array([[1,0],[0,-1]])
    U=Z_rot(Angles_SVT[0])
    #U=linalg.expm((1j)*Angles_SVT[0]*Z)
    O=qubit_rot(x)
    M=U.dot(O)
    for l in range(len(Angles_SVT)-1): 
        U=Z_rot(Angles_SVT[l+1])
        M=M.dot(U.dot(O))
    return M





#qt total qubits
#circuit call and measurement
def simulate(circuit,qt):
    #state prepap |0>
    the_state=np.zeros((2**qt,1))
    the_state[0] = 1 # start in |000>
    the_state=circuit.dot(the_state)
    return the_state


#qm qubit measured
def measurement(state,qm,qt):
    den_state=state.dot(state.transpose().conjugate())
    ket1 = np.array([[0], [1]])
    bra1 = np.array([0, 1])
    P1=np.kron(np.identity(2**(qm-1)),ket1*bra1)
    P2=np.kron(P1,np.identity(2**(qt-qm))).dot(den_state)
    return P2.trace()

#qubit measured 6 qubits 010010
def measurement6(state):
    den_state=state.dot(state.transpose().conjugate())
    ket1 = np.array([[0], [1]])
    bra1 = np.array([0, 1])
    P01=np.kron(np.identity(2),ket1*bra1)
    P001=np.kron(np.identity(2**2),ket1*bra1)
    P01001=np.kron(P01,P001)
    P010010=np.kron(P01001,np.identity(2)).dot(den_state)
    return P010010.trace()






