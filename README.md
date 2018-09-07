# Graph2Gauss 

<img src="https://www.kdd.in.tum.de/fileadmin/_processed_/csm_g2g_f6c3f13530.png" width="600">


Tensorflow implementation of the method proposed in the paper:
"[Deep Gaussian Embedding of Graphs: Unsupervised Inductive Learning via Ranking](https://openreview.net/forum?id=r1ZdKJ-0W)", Aleksandar Bojchevski and Stephan Günnemann, ICLR 2018.

## Installation
```bash
python setup.py install
```

## Requirements
* tensorflow (>=1.4)
* sklearn (only for evaluation)

## Demo
See the notebook example.ipynb for a simple demo.

## Graphs without attributes
If you graph has no attribute information you can run the one-hot version of Graph2Gauss (G2G_oh) by setting X=I, where I is the identity matrix. Additionally, setting X=A+I, where A is the adjacency matrix often yields even better perfomance.

## Misc
For an animation of Graph2Gauss learning to embed nodes as 2D Gaussians see: https://twitter.com/abojchevski/status/958278834025091072?s=19


## Cite
Please cite our paper if you use this code in your own work:

```
@inproceedings{
bojchevski2018deep,
title={Deep Gaussian Embedding of Graphs:  Unsupervised Inductive Learning via Ranking},
author={Aleksandar Bojchevski and Stephan Günnemann},
booktitle={International Conference on Learning Representations},
year={2018},
url={https://openreview.net/forum?id=r1ZdKJ-0W},
}
```
