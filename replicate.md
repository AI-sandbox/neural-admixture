# Experiments replication

We provide a simple step-by-step guide to train an unsupervised, pretrained and supervised version of Neural ADMIXTURE for the dataset CHM-22, with the same settings as in the paper. Other datasets have been avoided to avoid data size issues for hosting.

If a GPU is to be used for training (which we recommend), it needs to have at least 12GB of memory.

The data can be downloaded from [this link](https://www.dropbox.com/s/6z5ln82qz6jels6/chm-22.tar.gz?dl=0). It is a compressed file containing the data of CHM-22, as well as results from classical ADMIXTURE experiments. Extract its content in the root directory. Check that the folder `data/CHM-22` contains these 10 files:

```sh
- CHM-22_classic_train.P  # ADMIXTURE output
- CHM-22_classic_train.Q  # ADMIXTURE output
- CHM-22_classic_valid.P  # ADMIXTURE output
- CHM-22_classic_valid.Q  # ADMIXTURE output
- CHM-22-SUPERVISED_classic_train.P  # ADMIXTURE output
- CHM-22-SUPERVISED_classic_train.Q  # ADMIXTURE output
- CHM-22-SUPERVISED_classic_valid.P  # ADMIXTURE output
- CHM-22-SUPERVISED_classic_valid.Q  # ADMIXTURE output
- train.h5  # Training data
- validation.h5  # Validation data
```

You will need to clone this repository to run the replication scripts. You can either use the same environment used to run Neural ADMIXTURE or install the requirements from `requirements.txt `in a fresh environment.

To launch training, simply run:

```console
> cd neural-admixture/src && python3 launch_trainings.py
```

Note that this will overwrite previously trained models, included the downloaded ones.

Three models will be trained in total, one per experiment. Moreover, the weights of the networks will be stored in `outputs/weights`, while visualizations of the Q estimates will be saved to `outputs/figures`.

To compute and export metrics from trained models, launch the following command:

```console
> cd neural-admixture/src && python3 evaluate.py
```

Several metrics reported will be computed and written to the standard output for every experiment.

## Pre-trained models

Pre-trained models can be downloaded from [this link](https://www.dropbox.com/s/6ybdy8siclul1o7/chm-22-weights.tar.gz?dl=0). Simply download the file and extract it in the root folder. This will place the weights into the folder `outputs/weights`, and they will be ready to run evaluation on.

## Results

Note that results may differ by a very small amount with those presented due to the hardware used, specially if the models used are not the pre-trained. Expected results of the neural version are:

|      Dataset      |   Loss  | Δ<sub>T</sub> | Δ<sub>V</sub> | AMI<sub>T</sub> | AMI<sub>V</sub> |
|:-----------------:|:-------:|:-------:|:-------:|:-----:|:-----:|
|       CHM-22      | 6.802e8 |   2.4   |   .67   |  .88  |  .87  |
| CHM-22-PRETRAINED | 6.621e8 |   6.0   |   1.5   |  .79  |  .78  |
| CHM-22-SUPERVISED | 6.695e8 |  1.1e-5 |   .26   |  1.0  |   .9  |