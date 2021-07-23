# Neural ADMIXTURE

![multi_head_arch](https://user-images.githubusercontent.com/31998088/123008111-99a46e00-d3ba-11eb-8ced-d394ef903487.png)

Neural ADMIXTURE is an unsupervised global ancestry inference technique based on ADMIXTURE. By using neural networks, Neural ADMIXTURE offers high quality ancestry assignments with a running time which is much faster than ADMIXTURE's. For more information, we recommend reading [the corresponding article](https://www.biorxiv.org/content/early/2021/06/28/2021.06.27.450081).

The software can be invoked via CLI and has a similar interface to ADMIXTURE (_e.g._ the output format is completely interchangeable). While the software runs in both CPU and GPU, we recommend using GPUs if available to take advantage of the neural network-based implementation.

## Requirements

We recommend creating a fresh Python 3.9 environment using `virtualenv` (or `conda`), and then install the requirements there. As an example, for `virtualenv`, one should launch the following commands:

```console
> virtualenv --python=python3.9 ~/venv/nadmenv
> source ~/venv/nadmenv/bin/activate
(nadmenv) > pip3 install -r requirements.txt 
```

## Training from scratch

To train a model from scratch, simply invoke the following commands from the root directory of the project. For more info about all the arguments, please run `python3 train.py --help`. If training a single-head version of the network suffices, please set `min_k` and `max_k` to the same value (K). Note that only HDF5 and VCF files are supported as of now. The

For unsupervised Neural ADMIXTURE:

```console
> python3 train.py --min_k MIN_K --max_k MAX_K --epochs NUM_EPOCHS --decoder_init pckmeans --name RUN_NAME --data_path DATA_PATH --save_dir SAVE_PATH --init_path INIT_PATH --batch_size BSIZE
```

For supervised Neural ADMIXTURE:

```console
> python3 train.py --min_k MIN_K --max_k MAX_K --supervised --epochs NUM_EPOCHS --decoder_init supervised --name RUN_NAME --data_path DATA_PATH --save_dir SAVE_PATH --init_path INIT_PATH --batch_size BSIZE # only single-head support at the moment
```

For pre-trained Neural ADMIXTURE:

```console
> python3 train.py --min_k MIN_K --max_k MAX_K --freeze_decoder --epochs NUM_EPOCHS --decoder_init admixture --name RUN_NAME --data_path DATA_PATH --save_dir SAVE_PATH --init_path INIT_PATH  --batch_size BSIZE # only single-head support at the moment
```

As an example, the following ADMIXTURE call

```console
> ./admixture snps_data.bed 8 -s 42
```

would be mimicked in Neural ADMIXTURE by running

```console
> python3 train.py --min_k 8 --max_k 8 --epochs 10 --decoder_init pckmeans --name snps_data --data_path snps_data.vcf --save_dir SAVE_PATH --init_path INIT_PATH --seed 42 --batch_size 200
```

with some parameters such as the number of epochs, the decoder initialization or the save directories not having a direct equivalent. Note that if you have BED files, you should convert them to VCF using, for example, [plink](https://www.cog-genomics.org/plink/2.0/).

Several files will be output to the `SAVE_PATH` directory (the `name` parameter will be used to create the whole filenames):
- A `.P` file, similar to ADMIXTURE.
- A `.Q` file, similar to ADMIXTURE.
- A `.pt` file, containing the weights of the trained network.
- A `.json` file, with the configuration of the network.

The last two files are required to run posterior inference using the network, so be aware of not deleting them accidentally! Logs are printed to the `stdout` channel by default. If you want to store them in a file, you can use a pipe (`python3 train.py ... > test.log`) so that the output is redirected to that file.

## Inference mode (projective analysis)

ADMIXTURE allows reusing computations in the _projective analysis_ mode, in which the `P` (`F`, frequencies) matrix is fixed to an already known result and only the assignments are computed. Due to the nature of our algorithm, assignments can be computed for unseen data by simply feeding the data through the encoder. The entry-point for this mode is `inference.py`.

For example, assuming we have a trained Neural ADMIXTURE (named `nadm_test`) in the path `./outputs`, one could run inference on unseen data (`./data/unseen_data.vcf`) via the following command:

```console
> python3 inference.py --name nadm_test --save_dir ./outputs --out_name unseen_nadm_test --data_path ./data/unseen_data.vcf
```

For this command to work, files `./outputs/nadm_test.pt` and `./outputs/nadm_test_config.json`, which are training outputs, must exist. In this case, only a `.Q` will be created, which will contain the assignments for this data (the parameter of the flag `out_name` will be used to generate the output file name). This file will be written in the `--save_dir` directory (in this case, `./outputs`).


## Experiments replication

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

## Cite

When using this software, please cite the following paper (currently pre-print):

```{tex}
@article {Mantes2021.06.27.450081,
	author = {Mantes, Albert Dominguez and Montserrat, Daniel Mas and Bustamante, Carlos and Giró-i-Nieto, Xavier and Ioannidis, Alexander G},
	title = {Neural ADMIXTURE: rapid population clustering with autoencoders},
	elocation-id = {2021.06.27.450081},
	year = {2021},
	doi = {10.1101/2021.06.27.450081},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Characterizing the genetic substructure of large cohorts has become increasingly important as genetic association and prediction studies are extended to massive, increasingly diverse, biobanks. ADMIXTURE and STRUCTURE are widely used unsupervised clustering algorithms for characterizing such ancestral genetic structure. These methods decompose individual genomes into fractional cluster assignments with each cluster representing a vector of DNA marker frequencies. The assignments, and clusters, provide an interpretable representation for geneticists to describe population substructure at the sample level. However, with the rapidly increasing size of population biobanks and the growing numbers of variants genotyped (or sequenced) per sample, such traditional methods become computationally intractable. Furthermore, multiple runs with different hyperparameters are required to properly depict the population clustering using these traditional methods, increasing the computational burden. This can lead to days of compute. In this work we present Neural ADMIXTURE, a neural network autoencoder that follows the same modeling assumptions as ADMIXTURE, providing similar (or better) clustering, while reducing the compute time by orders of magnitude. In addition, this network can include multiple outputs, providing the equivalent results as running the original ADMIXTURE algorithm many times with different numbers of clusters. These models can also be stored, allowing later cluster assignment to be performed with a linear computational time.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2021/06/28/2021.06.27.450081},
	eprint = {https://www.biorxiv.org/content/early/2021/06/28/2021.06.27.450081.full.pdf},
	journal = {bioRxiv}
}
