# neural_spiral
A Feed-forward Neural Network trained to interpolate a spiral.

## Intro
This is an experiment to see how well a simple Feed-forward Neural Network can map a spiral function and interpolate between the mapped points of resolution.

In this example I have trained the network to learn 64 points on the spiral and the other points between them the neural network has to guess. The input to the neural network is just one normalised float 0-1 which defines at what position the neural network should reurn an x,y position for, thus making the network have only two outputs.

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
With such a small network the initial results were quite impressive, the original CPU spiral in red and the Neural spiral in green;


64 samples  | 256 samples (4x)
| ------------- | ------------- |
| ![64 Samples](https://raw.githubusercontent.com/jcwml/neural_spiral/main/models/M1/64.png)  | ![256 Samples](https://raw.githubusercontent.com/jcwml/neural_spiral/main/models/M1/256.png)  |
512 samples (8x)  | 8192 samples (128x)
| ------------- | ------------- |
| ![512 Samples](https://raw.githubusercontent.com/jcwml/neural_spiral/main/models/M1/512.png)  | ![8192 Samples](https://raw.githubusercontent.com/jcwml/neural_spiral/main/models/M1/8192.png)  |


64 samples  | 256 samples (4x) | 512 samples (8x) | 8192 samples (128x)
------------- | -------------
![64 Samples](https://raw.githubusercontent.com/jcwml/neural_spiral/main/models/M1/64.png) | ![256 Samples](https://raw.githubusercontent.com/jcwml/neural_spiral/main/models/M1/256.png) | ![512 Samples](https://raw.githubusercontent.com/jcwml/neural_spiral/main/models/M1/512.png) | ![8192 Samples](https://raw.githubusercontent.com/jcwml/neural_spiral/main/models/M1/8192.png)



## With such a small network the initial results were quite impressive, the original CPU spiral in red and the Neural spiral in green;

### At 64 samples accurate to what the network was trained at.
![64 Samples](https://raw.githubusercontent.com/jcwml/neural_spiral/main/models/M1/64.png)

### At 256 samples interpolated to 4x the original sample resolution.
![256 Samples](https://raw.githubusercontent.com/jcwml/neural_spiral/main/models/M1/256.png)

### At 512 samples interpolated to 8x the original sample resolution.
![512 Samples](https://raw.githubusercontent.com/jcwml/neural_spiral/main/models/M1/512.png)

### At 8192 samples interpolated to 128x the original sample resolution.
![8192 Samples](https://raw.githubusercontent.com/jcwml/neural_spiral/main/models/M1/8192.png)

## Argv

`main <samples> <csv> <csv samples>`
- <samples>: The amount of samples to produce the CPU spiral (in red).
- <csv>: Path to the neural csv file generated by `fit.py`.
- <csv samples>: The amount of samples to take from the neual csv file evenly spaced out to produce the Neural spiral (in green).
  
The neural csv file contains a total of 8192 samples extrapolated/interpolated from the sample rate you trained the network at.

`python3 fit.py <layers> <unit per layer> <training batches> <activation function> <optimiser> <cpu train only 0/1> <training sample resolution> <training epoches>`
- <layers>: The amount of layers in the Feed-forward Neural Network.
- <unit per layer>: How many perceptron units per layer of the network.
- <training batches>: How many forward passes to average together before doing a backpropergation pass.
- <activation function>: The activation function used by the hidden layers of the neural network, the output layer is always linear.
- <optimiser>: The optimiser used by the neural network.
- <cpu train only 0/1>: To train on the GPU only pass 0 or to train on the CPU only pass 1.
- <training sample resolution>: The amount of sample points from the spiral to train the neural network with.
- <training epoches>: The amont of times the neural network is trained with the same set of sample points.


