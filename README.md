# Neural ADMIXTURE

![multi_head_arch](https://user-images.githubusercontent.com/31998088/123008111-99a46e00-d3ba-11eb-8ced-d394ef903487.png)


## Requirements

We recommend creating a fresh Python 3.9 environment using `virtualenv` (or `conda`), and then install the requirements there. As an example, for `virtualenv`, one should launch the following commands:

```console
> virtualenv --python=python3.9 ~/venv/nadmenv
> source ~/venv/nadmenv/bin/activate
(nadmenv) > pip3 install -r requirements.txt 
```

## Training from scratch

The HDF5 files containing the train and validation datasets must be placed in the data path passed as an argument, and they must be named `train.h5` and `validation.h5` respectively.
To train a model from scratch, simply invoke the following commands inside the `src` directory. For more info about all the arguments, please run `python3 train.py --help`. If training a single-head version of the network suffices, please set `min_k` and `max_k` to the same value (K).

For unsupervised Neural ADMIXTURE:

```console
> python3 train.py --min_k MIN_K --max_k MAX_K --epochs NUM_EPOCHS --decoder_init pckmeans --name RUN_NAME --data_path DATA_PATH --save_dir SAVE_PATH --init_path INIT_PATH
```

For supervised Neural ADMIXTURE:

```console
> python3 train.py --min_k MIN_K --max_k MAX_K --supervised --epochs NUM_EPOCHS --decoder_init supervised --name RUN_NAME --data_path DATA_PATH --save_dir SAVE_PATH --init_path INIT_PATH  # only single-head support at the moment
```

For pre-trained Neural ADMIXTURE:

```console
> python3 train.py --min_k MIN_K --max_k MAX_K --freeze_decoder --epochs NUM_EPOCHS --decoder_init admixture --name RUN_NAME --data_path DATA_PATH --save_dir SAVE_PATH --init_path INIT_PATH  # only single-head support at the moment
```

## Experiments replication

We provide a simple step-by-step guide to train an unsupervised, pretrained and supervised version of Neural ADMIXTURE for the dataset CHM-22, with the same settings as in the paper. Other datasets have been avoided to avoid data size issues for hosting.

If a GPU is to be used for training (which we recommend), it needs to have at least 12GB of memory.

The data can be downloaded from [this link](https://www.dropbox.com/s/6z5ln682qz6jels6/chm-22.tar.gz?dl=0). It is a compressed file containing the data of CHM-22, as well as results from classical ADMIXTURE experiments. Extract its content in the root directory. Check that the folder `data/CHM-22` contains these 10 files:

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

To launch training, simply run:

```console
> cd src && python3 launch_trainings.py
```

Note that this will overwrite previously trained models, included the downloaded ones.

Three models will be trained in total, one per experiment. Moreover, the weights of the networks will be stored in `outputs/weights`, while visualizations of the Q estimates will be saved to `outputs/figures`.

To compute and export metrics from trained models, launch the following command:

```console
> cd src && python3 evaluate.py
```

Several metrics reported will be computed and written to the standard output for every experiment.

### Pre-trained models

Pre-trained models can be downloaded from [this link](https://www.dropbox.com/s/6ybdy8siclul1o7/chm-22-weights.tar.gz?dl=0). Simply download the file and extract it in the root folder. This will place the weights into the folder `outputs/weights`, and they will be ready to run evaluation on.

### Results

Note that results may differ by a very small amount with those presented due to the hardware used, specially if the models used are not the pre-trained. Expected results of the neural version are:

|      Dataset      |   Loss  | Δ<sub>T</sub> | Δ<sub>V</sub> | AMI<sub>T</sub> | AMI<sub>V</sub> |
|:-----------------:|:-------:|:-------:|:-------:|:-----:|:-----:|
|       CHM-22      | 6.802e8 |   2.4   |   .67   |  .88  |  .87  |
| CHM-22-PRETRAINED | 6.621e8 |   6.0   |   1.5   |  .79  |  .78  |
| CHM-22-SUPERVISED | 6.695e8 |  1.1e-5 |   .26   |  1.0  |   .9  |


## License

**NOTICE**: This software is available for use free of charge for academic research use only. Academic users may fork this repository and modify and improve to suit their research needs, but also inherit these terms and must include a licensing notice to that effect. Commercial users, for profit companies or consultants, and non-profit institutions not qualifying as "academic research" should contact the authors for a separate license. This applies to this repository directly and any other repository that includes source, executables, or git commands that pull/clone this repository as part of its function. Such repositories, whether ours or others, must include this notice.
