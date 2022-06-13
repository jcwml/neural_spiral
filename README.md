# neural_spiral
A Feed-forward Neural Network trained to interpolate a spiral.

This is an experiment to see how well a simple Feed-forward Neural Network can map a spiral function and interpolate between the mapped points of resolution.

## Demo
In this example I have trained a very small network to learn 64 points on the spiral and then the other points between them the neural network has to guess. The input to the neural network is just one normalised float between 0-1 which defines at what position the neural network should return an x,y position on the spiral for, thus making the network have only two outputs.

You can run this demo by executing `M1.sh` although you may need to recompile the binary first by executing `release.sh`.

**Network Topology**
```
Layers: 3
Unit per Layer: 8
Activation Function: tanh
Optimiser: adam
Training data batches: 3
Training samples: 64
Training epoches: 66,666
```
**With such a small network the initial results were quite impressive, the original CPU spiral in red and the Neural spiral in green;**

64 samples *(original training set size)*  | 256 samples (4x)
------------- | -------------
![64 Samples](https://raw.githubusercontent.com/jcwml/neural_spiral/main/models/M1/64.png) | ![256 Samples](https://raw.githubusercontent.com/jcwml/neural_spiral/main/models/M1/256.png)

512 samples (8x)  | 8192 samples (128x)
------------- | -------------
![512 Samples](https://raw.githubusercontent.com/jcwml/neural_spiral/main/models/M1/512.png) | ![8192 Samples](https://raw.githubusercontent.com/jcwml/neural_spiral/main/models/M1/8192.png)

**Hunting for the best model;**<br>
*activation, optimiser, layers, units per layer, batches, sample resolution, epoches, [accuracy]*

tanh_adam_3_8_6_64_100000 [0.95]  | tanh_adam_6_32_6_64_30000 [0.98]
------------- | -------------
![tanh_adam_3_8_6_64_100000 [0.95]](https://raw.githubusercontent.com/jcwml/neural_spiral/main/models/1.png) | ![tanh_adam_6_32_6_64_30000](https://raw.githubusercontent.com/jcwml/neural_spiral/main/models/2.png)

selu_adam_6_32_6_64_30000 [0.97]  | selu_adam_6_32_6_512_30000 [0.99]
------------- | -------------
![selu_adam_6_32_6_64_30000 [0.97]](https://raw.githubusercontent.com/jcwml/neural_spiral/main/models/3.png) | ![selu_adam_6_32_6_512_30000 [0.99]](https://raw.githubusercontent.com/jcwml/neural_spiral/main/models/4.png)

## Argv

`./main <csv> <samples>`
- `<csv>`: Path to the neural csv file generated by `fit.py`.
- `<samples>`: Sampling rate of the two spirals.
  
The neural csv file contains a total of 8192 samples extrapolated/interpolated from the sample rate you trained the network at.

`python3 fit.py <layers> <unit per layer> <training batches> <activation function> <optimiser> <cpu train only 0/1> <training sample resolution> <training epoches>`
- `<layers>`: The amount of layers in the Feed-forward Neural Network.
- `<unit per layer>`: How many perceptron units per layer of the network.
- `<training batches>`: How many forward passes to average together before doing a backpropergation pass.
- `<activation function>`: The activation function used by the hidden layers of the neural network, the output layer is always linear.
- `<optimiser>`: The optimiser used by the neural network.
- `<cpu train only 0/1>`: To train on the GPU only pass 0 or to train on the CPU only pass 1.
- `<training sample resolution>`: The amount of sample points from the spiral to train the neural network with.
- `<training epoches>`: The amont of times the neural network is trained with the same set of sample points.

## Dependencies
Linux, gcc, python3, tensorflow, numpy, etc.
