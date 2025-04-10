import numpy as np
from Basic_Gates import qubit_rot
from Basic_Gates import GZ
from Basic_Gates import interaction_gate





#Input parameters of the circuit
        #1) input_data is the elements of the domain used for the training
        #2) parameter_vector is the parameters of the circuit, elements of the population
       

def the_circuit(input_data,parameter_vector):
    input = input_data
    theta0 = parameter_vector[0]
    theta1 = parameter_vector[1]
    theta2 = parameter_vector[2]
    theta3 = parameter_vector[6]
    theta4 = parameter_vector[8] 
    phi0 = parameter_vector[3]
    phi1 = parameter_vector[4]
    phi2 = parameter_vector[5]
    phi3 = parameter_vector[7]
    xi=parameter_vector[9]
  
    M = np.kron(qubit_rot(theta0),np.kron(qubit_rot(theta1),  # Rx(theta_0) on qubit 1, Rx(theta_1) on qubit 2
                                qubit_rot(theta2))) # Rx(theta_2) on qubit 3
    M = np.kron(GZ(input),                            # GZ(x) on qubit 1
                        np.kron(GZ(input),                    # GZ(x) on qubit 2
                                GZ(input))).dot(M)                  # GZ(x) on qubit 3
    M = np.kron(interaction_gate(xi),    # XX(xi) on qubits 1, 2
                        np.identity(2)).dot(M) # type: ignore
    M = np.kron(np.identity(2),
                        interaction_gate(xi)).dot(M)  # XX(xi) on qubits 2, 3
    M = np.kron(GZ(phi0),                   # GZ(phi_0) on qubit 1
                        np.kron(GZ(phi1),           # GZ(phi_1) on qubit 2
                                GZ(phi2))).dot(M)         # GZ(phi_2) on qubit 3
    M = np.kron(np.identity(2),np.kron(qubit_rot(theta3),           # Rx(theta_3) on qubit 2
                                np.identity(2))).dot(M) 
    M = np.kron(interaction_gate(xi),    # XX(xi) on qubits 1, 2
                        np.identity(2)).dot(M)
    M = np.kron(np.identity(2),
                        interaction_gate(xi)).dot(M)  # XX(xi) on qubits 2, 3
    M = np.kron(np.identity(2),
                        np.kron(GZ(phi3),          # GZ(phi_3) on qubit 2
                                np.identity(2))).dot(M)
    circuit = np.kron(np.identity(2),
                        np.kron(qubit_rot(theta4),    # Rx(theta_4) on qubit 2
                                np.identity(2))).dot(M)
    return circuit

