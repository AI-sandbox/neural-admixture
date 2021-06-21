# Neural ADMIXTURE: Population Clustering with Autoencoders

## Replication instructions for chromosome 22

We provide a simple step-by-step guide to train an unsupervised, pretrained and supervised version of Neural ADMIXTURE for the dataset CHM-22, with the same settings as in the paper. Other datasets have been avoided to avoid data size issues for hosting.

Note that results may differ by a very small amount with those presented due to the hardware used. Moreover, if a GPU is to be used (which we recommend), it needs to have at least 12GB of memory.

1. Create a fresh Python 3.9 environment using `virtualenv`, then install the requirements:
```{sh}
> mkdir ~/venv # If first time using virtualenv
> virtualenv --python=python3.9 ~/venv/nadmenv
> source ~/venv/nadmenv/bin/activate
(nadmenv) > pip install -r requirements.txt 
```

2. Download the data from [INVALID LINK](). Extract the content of the file in the root directory.

3. Launch the following command to start training:
```{sh}
cd src && python3 replicate_experiments.py
```

Three models will be trained in total, one per experiment. After every model is trained, several metrics reported in the paper will be computed and written to the standard output. Moreover, the weights of the networks will be stored in `outputs/weights`, while visualizations of the Q estimates will be saved to `outputs/figures`.
