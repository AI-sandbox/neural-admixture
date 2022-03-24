# Demo

## Demo data

This demo uses a very small subset of the dataset _AllChms_ used in the article (105 individuals, 8451 variants). Therefore, results are not intended to be meaningful and should not be intepreted as such.

## Running the demo

Make sure the package is installed (`> pip3 install neural-admixture`) in the current environment and run the following command:

```console
> sh run_demo.sh
```

This will launch a 5-epoch training of Neural ADMIXTURE. When the training's finished, the `Q` and `P` outputs are then compared against the expected outputs. Note that the output may be a bit different to the expected one depending on the hardware, even to the point where the message _Output and expected output are not similar_ is displayed. Therefore, seeing this message at the end of the demo does not necessarily mean that the installation is faulty.

The expected output was generated using the version `1.1.2` of the software on a 2019 MacBook Air (Intel) running MacOS BigSur 11.2.3 (Python version: `3.9.7`). The demo took ~7 seconds to execute under the same conditions. Be aware that the first usage may be a bit slower due to first-time package loading, especially on a recently created environment.
