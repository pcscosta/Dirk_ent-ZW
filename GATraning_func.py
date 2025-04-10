
# general imports
import numpy as np # type: ignore
import random
import math

from Basic_Gates import measurement
from Basic_Gates import simulate

#Choose the circuit
#from Dirk_Circuit import the_circuit #This was used for PW
from DirkPower_Circuit import the_circuit 


def init_pop(no_parameters,population_size):
    return np.random.uniform(0, 2*np.pi, (population_size,no_parameters))


pi = math.pi

def mut_pop(population,sigma,pm):
    population_size = population.shape
    population_new =np.zeros((population_size[0],population_size[1]))
    for i in range(population_size[0]):
        for j in range(population_size[1]):
            if np.random.randint(low=0, high=100, size=1)/100<=pm:
                a=abs(math.fmod(np.random.normal(population[i][j], sigma, 1),2*pi))
                population_new[i][j]=a
            else:
                population_new[i][j]=population[i][j]
    return population_new



def cross_pop(population,pc):
    population_size = population.shape
    pop_cross=np.zeros((population_size[0],population_size[1]))
    j=0
    for i in range(int(population_size[0]/2)-1):
        if float(np.random.randint(low=0, high=10, size=1)/10) <= pc:
            elem_cross=int(np.random.randint(low=1, high=population_size[1]/2, size=1))
            for k in range(elem_cross,int(population_size[1])):
                pop_cross[j][k]=population[j+1][k]
                pop_cross[j+1][k]=population[j][k]
        else:
            pop_cross[j]=population[j]
            pop_cross[j+1]=population[j+1]
        j=j+2
    return pop_cross



def update_pop(population,fitness,sigma,pm,pc):
    normalised_fitness = (10 * np.exp(-(fitness)**0.25))**6 
    normalised_fitness = normalised_fitness / np.sum(normalised_fitness)
    unmutated_pop = np.array(random.choices(population, weights=normalised_fitness, k=len(population)))
    cros_Pop=cross_pop(unmutated_pop,pc)
    mutated_pop=mut_pop(cros_Pop,sigma,pm)
    return mutated_pop





#Resc for fake data 0.4
#Resc for PW 0.5
#Samp gives the number of samples used - samp=0 goes to ideal prob.
def fitness(population,input_data,output_data,q,qt,samp,resc_Prob=0.5):
    population_size = len(population)
    output = np.zeros(population_size)
    for idx_pop in range(population_size):
        for idx_input in range(len(input_data)):
            Pr = measurement(simulate(the_circuit(input_data[idx_input],population[idx_pop]),qt),q,qt)
            if samp != 0:
                p = measurement(simulate(the_circuit(input_data[idx_input],population[idx_pop]),qt),q,qt)
                Pr=sum(np.random.binomial(1, p, samp))/samp
            target_prob = resc_Prob + output_data[idx_input]
            output[idx_pop] += (Pr - target_prob)**2
    #np.exp(-(output/len(input_data))**0.25)
    return np.abs(output/len(input_data))

