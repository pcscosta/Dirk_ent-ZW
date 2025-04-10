import numpy as np # type: ignore
from Basic_Gates import SingValT, qubit_rot
from Basic_Gates import GZ
from Basic_Gates import interaction_gate





#Input parameters of the circuit
        #1) input_data is the elements of the domain used for the training
        #2) parameter_vector is the parameters of the circuit, elements of the population
       

def the_circuit(input_data,Angles,ord=6):
    
    parameter_vector=Angles[2*ord+4:2*ord+11]
    theta0 = parameter_vector[0]
    theta1 = parameter_vector[1]
    theta2 = parameter_vector[2]
    theta3 = parameter_vector[3] 
    phi0 = parameter_vector[4]
    phi1 = parameter_vector[5]
    xi=parameter_vector[6]
  
    M = np.kron(SingValT(Angles[0:ord],10**(-2)*int(str(input_data)[0:2])),np.kron(SingValT(Angles[ord:2*ord],10**(-2)*int(str(input_data)[2:4])),  # SVT on qubit 1,2
                                SingValT(Angles[2*ord:2*ord+4],10**(-3)*int(str(input_data)[4:7])))) # SVT on qubit 3
    M = np.kron(interaction_gate(xi),    # XX(xi) on qubits 1, 2
                        np.identity(2)).dot(M) # type: ignore
    M = np.kron(np.identity(2),
                        interaction_gate(xi)).dot(M)  # XX(xi) on qubits 2, 3
    M = np.kron(GZ(theta0),                   # GZ(theta0) on qubit 1
                        np.kron(GZ(theta1),           # GZ(theta1) on qubit 2
                                GZ(theta2))).dot(M)         # GZ(theta2) on qubit 3
    M = np.kron(np.identity(2),np.kron(qubit_rot(theta3),           # Rx(theta3) on qubit 2
                                np.identity(2))).dot(M) 
    M = np.kron(interaction_gate(xi),    # XX(xi) on qubits 1, 2
                        np.identity(2)).dot(M)
    M = np.kron(np.identity(2),
                        interaction_gate(xi)).dot(M)  # XX(xi) on qubits 2, 3
    M = np.kron(np.identity(2),
                        np.kron(GZ(phi0),          # GZ(phi0) on qubit 2
                                np.identity(2))).dot(M)
    circuit = np.kron(np.identity(2),
                        np.kron(qubit_rot(phi1),    # Rx(phi1) on qubit 2
                                np.identity(2))).dot(M)
    return circuit

