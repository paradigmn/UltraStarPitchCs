# UltrastarPitchCs
Reimplementation of [ultrastar-pitch](https://github.com/paradigmn/ultrastar_pitch)'s core functionality in C#.  
  
## Disclaimer
This project is not going to replace ultrastar-pitch. It provides neither all the features nor the performance of the original project. If you want a software which automatically predicts the pitches of your ultrastar project, visit the link above!
  
The software can be considered an early stage of a software plugin. If you desire vocal pitch detection in your software, you can include the provided source code.
  
## Implementation
The original idea was to use Math.NET for the number crunching and the ONNX Runtime for deep learning inference. In the end, every math library using managed code, had intolerable performance issues. The ONNX Runtime on the other hand suffered from huge libraries (>100MB), additionally platform independence wasn't guaranteed.  
  
This resulted in the decision to implement everything in native C# without any outside dependencies. To get as close as possible to the performance, the original (Numpy) implementation provides, several performance optimizations had to be made:  
  
* replacing 2d arrays (slow) by 1d and jagged arrays
* using SIMD vector arithmetic
* enable code parallelization
* prune fft algo for use case
  
Despite all this steps, the managed code (C#) is still inferior to the performance of a C implementation (Numpy).

## Known issues and future plans
This implementation requires the input audio signal to be a mono WAV array with a sample rate of 16kHz. This is the configuration the preprocessing and detection algos were developed for. To deal with common audio signals (44.1khz / 48khz) a bit rate converter needs to be implemented.  
Furthermore, the original software has a postprocessing module to optimize the pitch prediction. It uses a key matrix to predict a pseudo code, the song was written in. The module still needs to be ported.  
Last but not least, the implementation is not sufficiently tested and does not provide exception handling as of now.

## Assets
The project consists of multiple modules, which can be chained together in a pipeline to provide the required functionality.
### LinearAlgebraUtils
A helper module with highly optimized elementary vector operations (add, sub, dot). It is used by all modules that rely on mathematic array calculations. Additionally, a minimalistic parser for Numpy Ndarrays was implemented to load numeric constants.  
  
### ProjectParser
A parser for note.txt files with the option to load audio. Mainly for testing purposes.

### Audio Preprocessor
Process audio segments into an array of features for classification.

### Magnitude Spectrum
Helper class for the audio processor. Consist of an optimized radix-2 fft algorithm.

### Pitch Classifier
Predict the pitch from a feature array.

## Binaries
Numpy Ndarray files with mathematical constants.

### PCA Components and Mean Values
The constants of the PCA mean vector and component matrix for feature dimensionality reduction.

### Model Biases and Weights
The constants of the neuronal network layers for prediction.

