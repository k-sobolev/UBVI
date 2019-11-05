# Universal Boosting Variational Inference

This is unofficial Pytorch implementation of Universal Boosting Variational Inference. Based on unfinished author’s implementation on Numpy: https://github.com/trevorcampbell/ubvi

The project is based on the paper: https://arxiv.org/pdf/1906.01235.pdf

The proposal for the project may be found here: https://www.overleaf.com/7769475349bxtstwpmmzsq

The presentation can be found here: https://docs.google.com/presentation/d/1U1OPyvEx97d3yEUld0rSrA6vNYsNDSrqel1fOZjFXjw/edit#slide=id.g641ef2860b_3_192

The project report can be found here:
https://www.overleaf.com/read/tzntkfphzrds


## Requirements

```
PyTorch 1.3.0
numpy 1.14
matplotlib
```

## Project goals

During the project, we are going to:

1. Implement algorithm in PyTorch;
2. Replicate experiments states in the paper with UBVI (synthesised data from Cauchy and banana distributions);
3. Implement BBVI;
4. Compare results.

## How to use

```
python test_ubvi.py
```
```
python test_bbvi.py
```
The resulting graphs will be in the results folder.

```experiment_distributions.ipynb``` - notebook with experiments for report

## Cauchy distribution approximation examples

![alt text](https://raw.githubusercontent.com/k-sobolev/UBVI/master/pics/cauchi_plots.png)

## Banana distribution approximation examples

![alt text](https://raw.githubusercontent.com/k-sobolev/UBVI/master/pics/Banana.png)

## 


## Team members: 
 * Yuriy Biktairov
 * Konstantin Sobolev
 * Nurislam Tursynbek
