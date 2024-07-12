import torch
import torch.nn as nn
from tensorflow import keras
import numpy as np
from models import LeNet300_100

#Caricamento modello di riferimento
existing_model = keras.models.load_model('quantized_mnist_scaler1_2024-06-25_093159298149.h5')

#Inizializzazione modello Pytorch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
pytorch_model = LeNet300_100(10, device)

#Caricamento modello preaddestrato in Pytorch
pytorch_model.load_state_dict(torch.load('pruned_model.pth', map_location=device))

#Strasporto pesi da Pytorch a Keras
def transpose_weights(pytorch_layer, keras_layer):
    
    pt_weights = pytorch_layer.weight.detach().numpy()
    pt_biases = pytorch_layer.bias.detach().numpy()
    
    #Faccio il trasporto dei pesi
    pt_weights = pt_weights.T

    #Assegno i pesi e i bias al layer di Keras
    keras_layer.set_weights([pt_weights, pt_biases])

#Prendiamo i layer che servono sia dal modello Pytorch che da quello di Keras
pytorch_layers = [pytorch_model.classifier[0], pytorch_model.classifier[2], pytorch_model.classifier[4]]
keras_layers = [existing_model.layers[1], existing_model.layers[3], existing_model.layers[5]]


#for layer in keras_layers:
    #print(layer.name)

#for layer in pytorch_layers:
    #print(layer)

# Trasporto i pesi da Pytorch a Keras
for pt_layer, keras_layer in zip(pytorch_layers, keras_layers):
    transpose_weights(pt_layer, keras_layer)

existing_model.save('converted_model.h5')