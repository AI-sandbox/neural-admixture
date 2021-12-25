![PyPI - Python Version](https://img.shields.io/pypi/pyversions/neural-admixture)
![PyPI - Version](https://img.shields.io/pypi/v/neural-admixture)
![PyPI - License](https://img.shields.io/pypi/l/neural-admixture)
![PyPI - Status](https://img.shields.io/pypi/status/neural-admixture)
![PyPI - Downloads](https://img.shields.io/pypi/dm/neural-admixture)

# Neural ADMIXTURE

![nadm_mna](https://user-images.githubusercontent.com/31998088/147361139-a964ead1-993a-4654-84ee-d3efb850dedc.png)


Neural ADMIXTURE is an unsupervised global ancestry inference technique based on ADMIXTURE. By using neural networks, Neural ADMIXTURE offers high quality ancestry assignments with a running time which is much faster than ADMIXTURE's. For more information, we recommend reading [the corresponding article](https://www.biorxiv.org/content/10.1101/2021.06.27.450081v3).

The software can be invoked via CLI and has a similar interface to ADMIXTURE (_e.g._ the output format is completely interchangeable). While the software runs in both CPU and GPU, we recommend using GPUs if available to take advantage of the neural network-based implementation.

## Installation

The package can be easily installed using `pip` (make sure to add the `--upgrade` flag if updating the version):

```console
> pip3 install neural-admixture
```

We recommend creating a fresh Python 3.9 environment using `virtualenv` (or `conda`), and then install the package `neural-admixture` there. As an example, for `virtualenv`, one should launch the following commands:

```console
> virtualenv --python=python3.9 ~/venv/nadmenv
> source ~/venv/nadmenv/bin/activate
(nadmenv) > pip3 install neural-admixture
```

## Usage 
### Running Neural ADMIXTURE

To train a model from scratch, simply invoke the following commands from the root directory of the project. For more info about all the arguments, please run `neural-admixture train --help`. If training a single-head version of the network suffices, please use the flag `--k` instead of `--min_k` and `--max_k`. Note that VCF, BED, PGEN and HDF5 files are supported as of now. 

For unsupervised Neural ADMIXTURE (single-head):

```console
> neural-admixture train --k K --name RUN_NAME --data_path DATA_PATH --save_dir SAVE_PATH --init_file INIT_FILE
````

For unsupervised Neural ADMIXTURE (multi-head):

```console
> neural-admixture train --min_k MIN_K --max_k MAX_K --name RUN_NAME --data_path DATA_PATH --save_dir SAVE_PATH --init_file INIT_FILE
```

For supervised Neural ADMIXTURE:

```console
> neural-admixture train --k K --supervised --name RUN_NAME --data_path DATA_PATH --save_dir SAVE_PATH # only single-head support at the moment
```

As an example, the following ADMIXTURE call

```console
> ./admixture snps_data.bed 8 -s 42
```

would be mimicked in Neural ADMIXTURE by running

```console
> neural-admixture train --k 8 --data_path snps_data.bed --save_dir SAVE_PATH --init_file INIT_FILE --name snps_data --seed 42
```

with some parameters such as the decoder initialization or the save directories not having a direct equivalent.

Several files will be output to the `SAVE_PATH` directory (the `name` parameter will be used to create the whole filenames):
- If the unsupervised version is run, a `Pickle` binary file containing the PCA object (using the `init_name` parameter), as well as an image file containing a PCA plot.
- A `.P` file, similar to ADMIXTURE.
- A `.Q` file, similar to ADMIXTURE.
- A `.pt` file, containing the weights of the trained network.
- A `.json` file, with the configuration of the network.

The last two files are required to run posterior inference using the network, so be aware of not deleting them accidentally! Logs are printed to the `stdout` channel by default. If you want to save them to a file, you can use the command `tee` along with a pipe:

```console
> neural-admixture train --k 8 ... | tee run.log
```

### Initialization method

As described in the article, Neural ADMIXTURE's decoder(s) can be initialized using several methods, which will be indicated by the required `initialization` argument. The best-performing initialization method depends, mainly, on the structure of the data. The main options are:

- `pckmeans`: initialize using PCK-Means (Algorithm 1 in the paper). Performs best if data contains a large number of single-ancestry individuals.
- `pcarchetypal`: initialize using PCArchetypal (Algorithm 2 in the paper). Performs best if data contains a large number of admixed individuals.

### Inference mode (projective analysis)

ADMIXTURE allows reusing computations in the _projective analysis_ mode, in which the `P` (`F`, frequencies) matrix is fixed to an already known result and only the assignments are computed. Due to the nature of our algorithm, assignments can be computed for unseen data by simply feeding the data through the encoder. This mode can be run by typing `infer` instead of `train` right after the `neural-admixture` call.

For example, assuming we have a trained Neural ADMIXTURE (named `nadm_test`) in the path `./outputs`, one could run inference on unseen data (`./data/unseen_data.vcf`) via the following command:

```console
> neural-admixture infer --name nadm_test --save_dir ./outputs --out_name unseen_nadm_test --data_path ./data/unseen_data.vcf
```

For this command to work, files `./outputs/nadm_test.pt` and `./outputs/nadm_test_config.json`, which are training outputs, must exist. In this case, only a `.Q` will be created, which will contain the assignments for this data (the parameter of the flag `out_name` will be used to generate the output file name). This file will be written in the `--save_dir` directory (in this case, `./outputs`).

### Pretrained Neural ADMIXTURE

As also mentioned in the paper, Neural ADMIXTURE can be used to learn a function (through the encoder) given the results of the frequency matrix `P`/`F` of another algorithm so out-of-training data inference can be performed using the structure learnt by the other algorithm. The following arguments should be used to run Neural ADMIXTURE in this mode:

- `--initialization pretrained`: indicates than an ADMIXTURE-like `.P` file must be loaded to initialize the decoder weights. The path of the file is specified using the `--init_file` argument.
- `--freeze_decoder`: indicates that the decoder weights will be frozen during training.

If the second argument is skipped, then the decoder weights will be updated and the solution won't yield exactly the same `P`/`F` matrix that was used as input.

## Other options

- `l2_penalty`: indicates penalty to apply to encoder weights. Defaults to 0.0005. The greater the penalty, the "softer" the cluster assignments will be, limiting to a uniform distribution.
- `batch_size`: number of samples used at every update. If you have memory issues, try setting a lower batch size. Defaults to 200.
- `pca_components`: dimension of the PCA projection for the PC-KMeans initialization. Defaults to 8.
- `max_epochs`: maximum number of times the whole training dataset is used to update the weights. Defaults to 50. 
- `tol`: will stop optimization when difference in objective function between two iterations is smaller than this value. Defaults to 1e-6.
- `learning_rate`: dictates how big an update to the weights will be. If you find the loss function oscillating, try setting a lower value. If convergence is slow, try setting a higher value. Defaults to 0.0001.
- `seed`: seed for replication purposes, similar to ADMIXTURE's. Defaults to 42.

## Using Plink2 binary files (.pgen)

If the data format you will be working on is _Plink2 Binary Files (.pgen, .psam, .pvar)_ then you also need to install the package `pgenlib`. This package is not available in PyPi, but is included in the [plink repository](https://github.com/chrchang/plink-ng/tree/master/2.0/Python). Installation instructions can be found in the [corresponding `README.md` file](https://github.com/chrchang/plink-ng/blob/master/2.0/Python/ReadMe.md). While you will need to clone the whole repository, you can remove it after installing the package, unless you plan to work with it.

## Experiments replication

Some of the datasets used in the article will be available soon, along with the exact options to be run to replicate the results described in the paper.

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
