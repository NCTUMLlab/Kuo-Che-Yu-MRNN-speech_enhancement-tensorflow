# Markov Recurrent Neural Network for speech enhancement
In this project, we implement Markov Recurrent Neural Network (MRNN) for speech enhancement.

Markov recurrent neural network (MRNN) explore the stochastic transitions in recurrent neural networks by incorporating the Markov property with discrete random variables. This model was proposed to deal with highly structured sequential data with complicated latent information. The discrete samples are drawn from the parameterized categorical distribution at each time step, and latent information is encoded by different state encoders depends on which state is selected.

<img src="Others/Model.png" width="100%">



## Setting
- Hardware:
	- CPU: Intel Core i7-4930K @3.40 GHz
	- RAM: 64 GB DDR3-1600
	- GPU: NVIDIA Tesla K20c 6 GB RAM
- Tensorflow 0.12
- Dataset
	- Wall Street Journal Corpus
	- Noises are collected from [freeSFX](http://www.freesfx.co.uk/soundeffects/) and [AudioMicro](http://www.audiomicro.com/free-sound-effects)

## Result
- An example of original data signal

|<img src="Others/spectrum_mix.png" width="80%">|
|:--------------------------------------------:|
|Mixed signal|


|<img src="Others/spectrum_clean.png" width="80%">|
|:--------------------------------------------:|
|Clean signal|


- An example of testing outputs

|<img src="Others/exp_result/spec_mix.png" width="80%">|
|:--------------------------------------------:|
|Mixed signal|


|<img src="Others/exp_result/spec_clean.png" width="80%">|
|:--------------------------------------------:|
|Clean signal|


|<img src="Others/exp_result/spec_noise.png" width="80%">|
|:--------------------------------------------:|
|noise signal|

|<img src="Others/exp_result/spec_demix_mrnn.png" width="80%">|
|:--------------------------------------------:|
|output signal demixed by MRNN|

|<embed src="Others/Original_clean.wav">|
