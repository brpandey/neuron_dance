TODO

- [ ] add more tests
- [X] keep analyzing performance bottleneck in optimizer calcuate function 
    - [!] already tried map_inplace, zip's for_each and par_for_each for map
    - [!] map instances here refer to bias_correct, and momentum expression

<p float="center">
  <img src='images/perfreport.jpg' width='1000' height='500'/>
</p>


- [ ] protocol buffer serialization for network fields, activation and cost functions
      further: different files for different types of data, weight goes in weight file, hyper param into hyper file -> same directory
- [ ] cost functions, e.g. rms prop, sparse categorical entropy
- [ ] integrate [generics](https://github.com/brpandey/neuron_dance/tree/generics) branch?
- [ ] activation functions, e.g. [selu](https://www.tensorflow.org/api_docs/python/tf/keras/activations/selu) [swish](https://en.wikipedia.org/wiki/Swish_function)?
- [ ] add convolutional layer
- [ ] batch normalization layer
- [ ] custom layers e.g. residual blocks
- [ ] dropout, monte carlo dropout
- [ ] transfer learning, loading a pretrained model
- [ ] computation graph
