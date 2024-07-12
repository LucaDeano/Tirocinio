# Thesis-implementations
This repository contains all the implementations done for the thesis "Quantization Algorithms for Neural Networks: A comparative analysis with other reduction techniques".  

## Quantization
The original quantization method can be found in this repository: 
* [quantized_neural_networks](https://github.com/elybrand/quantized_neural_networks).

To run the training and quantization scripts, follow the steps in that repository using the modified version of the files found in this repository. 

Also, if you want to quantize a PyTorch model, you'll first need to 

* Convert it to a .txt file using the `pth_to_txt.py` script since the quantization implmentation in [quantized_neural_networks](https://github.com/elybrand/quantized_neural_networks) is done in TensorFlow.
* Input the .txt file into the `modified_train_mnist_mlp.py` script to get a TensorFlow model with the correct architecture.
* Run the `modified_quantized_network` script to get a quantized .h5 Keras model that can now be lumped using the `LumpingAlgorithm.py` script.

## Lumping
The lumping algorithm was implemented from this paper:
*  [Compressing Neural Networks via Formal Methods](https://www.sciencedirect.com/science/article/pii/S0893608024003356).

A "toy" example of how the algorithm works can be found below:

Input:
```
N = {
    'k': 4,
    'S': [{0, 1}, {2, 3}, {4, 5}, {6}],
    'W': {
        1: {(0, 2): 0.1, (1, 3): 0.2, (0, 3): 0.3, (1, 2): 0.4},
        2: {(2, 4): 0.5, (3, 4): 0.6, (2, 5): 1, (3, 5): 1.2},
        3: {(4, 6): 0.7, (5, 6): 0.8}
    },
    'b': {
        1: {2: 0.1, 3: 0.2},
        2: {4: 0.3, 5: 0.6},
        3: {6: 0.4}
    },
    'A': {
        1: {2: 'ReLU', 3: 'ReLU'},
        2: {4: 'ReLU', 5: 'ReLU'},
        3: {6: 'ReLU'}
    }
}
```

Output: 
```
4
{0: {0: [0], 1: [1]}, 1: {(2,): [2], (3,): [3]}, 2: {(4, 5): [4, 5]}, 5: 0.5, 3: {(6,): [6]}}
{1: {(0, (2,)): 0.1, (1, (2,)): 0.4, (0, (3,)): 0.3, (1, (3,)): 0.2}, 2: {((2,), (4,)): 0.5, ((3,), (4,)): 0.6}, 3: {((4, 5), (6,)): 1.1}}
{1: {(2,): 0.1, (3,): 0.2}, 2: {(4,): 0.3}, 3: {(6,): 0.4}}
{1: {(2,): 'ReLU', (3,): 'ReLU'}, 2: {(4,): 'ReLU'}, 3: {(6,): 'ReLU'}}
```
