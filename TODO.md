TODO

- [ ] add more tests
- [X] keep analyzing performance bottleneck in optimizer calcuate function 
      (already tried map_inplace, zip's for_each and par_for_each)
- [ ] protocol buffer serialization for network fields, activation and cost functions
      further: different files for different types of data, weight goes in weight file, hyper param into hyper file -> same directory
- [ ] cost functions, e.g. rms prop, sparse categorical entropy
- [ ] integrate generics branch?
- [ ] activation functions, e.g. selu? swish?
- [ ] add convolutional layer
- [ ] batch normalization layer
- [ ] custom layers e.g. residual blocks
- [ ] dropout, monte carlo dropout
- [ ] transfer learning, loading a pretrained model
- [ ] computation graph
