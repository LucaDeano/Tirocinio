import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

def extract_model_data(model):
    layers = [layer for layer in model.layers if 'dense' in layer.name]
    k = len(layers)
    S = [{i for i in range(784)}]
    W = {}
    b = {}
    A = {}

    neuron_count = 784  # Inizia il conteggio dei neuroni da 784 per MNIST
    neuron_count1 = 0  # Inizia il conteggio dei neuroni da 0 per il primo layer
    for i, layer in enumerate(layers):
        W[i+1] = {}
        weights, biases = layer.get_weights()
        # Assegna bias e funzioni di attivazione usando il contatore globale dei neuroni
        b[i+1] = {neuron_count + j: biases[j] for j in range(len(biases))}
        A[i+1] = {neuron_count + j: layer.activation.__name__ for j in range(len(biases))}
        
        # Creazione di una lista ordinata degli identificativi dei neuroni per il nuovo layer
        new_neurons = [neuron_count + j for j in range(weights.shape[1])]
        S.append(new_neurons)
        
        # Aggiorna i pesi con gli identificatori univoci dei neuroni
        for input_index in range(weights.shape[0]):
            for output_index in range(weights.shape[1]):
                # Usa il contatore globale per identificare univocamente i neuroni di output
                W[i+1][(input_index + neuron_count1, neuron_count + output_index)] = weights[input_index][output_index]
        
        # Aggiorna il contatore globale dei neuroni
        neuron_count += weights.shape[1]
        neuron_count1 += weights.shape[0]

    return {'k': k, 'S': S, 'W': W, 'b': b, 'A': A}

def lump_neural_network(N):
    k, S, W, b, A = N['k'], N['S'], N['W'], N['b'], N['A']

    # Initialize the equivalence classes for the first layer
    S_prime = {0: {i: [s_i] for i, s_i in enumerate(S[0])}}

    # Initialize the dictionaries for O and W_tilde
    O = {}
    W_tilde = {}

    # Fill O and W_tilde for the first layer
    for Si in S_prime[0]:
        for sj in S[1]:
            O[(Si, sj)] = W[1][(S_prime[0][Si][0], sj)]
            W_tilde[(Si, (sj,))] = W[1][(S_prime[0][Si][0], sj)]

    # Initialize dictionaries for the output
    W_prime = {}
    b_prime = {}
    A_prime = {}

    # Process each layer
    for l in range(1, k):
        S_prime[l] = {}
        Sl = list(S[l])
        b_prime[l] = {}
        A_prime[l] = {}
        W_prime[l] = {}

        while Sl:
            s = Sl.pop(0)
            C = [s]
            b_prime[l][tuple(C)] = b[l][s]
            A_prime[l][tuple(C)] = A[l][s]

            for S_prime_l_minus_1 in S_prime[l - 1]:
                W_prime[l][(S_prime_l_minus_1, tuple(C))] = W_tilde[(S_prime_l_minus_1, (s,))]

            for s_prime in Sl[:]:
                if b[l][s_prime] != 0:
                    rho_s_prime = b[l][s] / b[l][s_prime]
                    #print(rho_s_prime)
                    consistent = True

                    for S_prime_l_minus_1 in S_prime[l - 1]:
                        #print(rho_s_prime * O[(S_prime_l_minus_1, s_prime)], O[(S_prime_l_minus_1, s)])
                        if not np.isclose(rho_s_prime * O[(S_prime_l_minus_1, s_prime)], O[(S_prime_l_minus_1, s)], atol=0.00):
                            consistent = False
                            break

                    if consistent:
                        C.append(s_prime)
                        Sl.remove(s_prime)
                        S_prime[s_prime] = rho_s_prime

            #print(C)
            S_prime[l][tuple(C)] = C

        if l + 1 < k: # if not the last layer
            for C in S_prime[l]:
                for s_prime in S[l + 1]:
                    O[(C, s_prime)] = sum(W[l + 1][(r, s_prime)] for r in C)
                    W_tilde[(C, (s_prime,))] = sum(S_prime.get(r, 1) * W[l + 1][(r, s_prime)] for r in C)

    return k, S_prime, W_prime, b_prime, A_prime

model_path = 'converted_model.h5'
model = load_model(model_path)

N = extract_model_data(model)

#print(N['S'])

#metti N in un txt
with open('dictionary.txt', 'w') as f:
   f.write(str(N))

k, S_prime, W_prime, b_prime, A_prime = lump_neural_network(N)

with open('lumped_neurons.txt', 'w') as f:
   f.write(str(S_prime))

#print(k)
#print(S_prime)
#print(W_prime)
#print(b_prime)
#print(A_prime)