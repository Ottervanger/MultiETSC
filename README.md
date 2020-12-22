# MultiETSC

MultiETSC implements Automated Machine Learing (AutoML) for 
Early Time Series Classification (ETSC).
This is done by simultaneously optimizing for both earliness and accuracy using the 
multi-objective algorithm configurator MO-ParamILS.
The search space of this optimization consists of the set of ETSC algorithms and 
their hyper-parameters.


The following algorithms are included:

| Algorithm | code | Publication |
| --------- | ---- | ----------- |
| ECTS | C++ | [ Z. z. Xing, J. Pei, and P. Yu (2011). “Early classification on time series”](https://doi.org/10.1007/s10115-011-0400-x) |
| EDSC | C++ | [ Z. z. Xing, J. Pei, P. Yu, and K. Wang (2011). “Extracting interpretable featuresfor early classification on time series” ](https://doi.org/10.1137/1.9781611972818.22) |
| RelClass | MATLAB | [N. Parrish, H. S. Anderson, M. R. Gupta, and D. Y. Hsiao (2013). “Classifying with confidence from incomplete information”](http://jmlr.org/papers/v14/parrish13a.html) |
| ECDIRE | R | [ U. Mori, A. Mendiburu, E. Keogh, and J. Lozano (2016). “Reliable early classifi-cation of time series based on discriminating the classes over time”](https://doi.org/10.1007/s10618-016-0462-1) |
| SR-CF | R | [U. Mori, A. Mendiburu, S. Dasgupta, and J. A. Lozano (2018). “Early classifi-cation of time series by simultaneously optimizing the accuracy and ear-liness”](https://doi.org/10.1109/TNNLS.2017.2764939) |
| SR-CF MO | R | [ U. Mori, A. Mendiburu, I. Miranda, and J. Lozano (2019). “Early classification of time series using multi-objective optimization techniques”](http://www.sciencedirect.com/science/article/pii/S0020025519303317) |
| ECEC | Java| [ J. Lv, X. Hu, L. Li, and P. Li (2019). “An effective confidence-based early classification of time series”](https://doi.org/10.1109/ACCESS.2019.2929644) |
| TEASER | Java | [P. Sch ̈afer and U. Leser (2019). “Teaser: Early and accurate time series classification”](https://arxiv.org/abs/1908.03405) |
| EARLIEST | Python | [T. Hartvigsen, C. Sen, X. Kong, and E. Rundensteiner (2019). “Adaptive-haltingpolicy network for early classification”](https://web.cs.wpi.edu/~xkong/publications/papers/kdd19.pdf) |

## Installation
MultiETSC is mainly built as a combination of python and bash scripts and in itself requires no installation.
However, the set of ETSC algorithms as well as the algorithm configurator(s) do each have their own dependencies.
In order to set up all dependencies of MultiETSC run the following command:
```bash
$ make build # TODO: not implemeted
```

## Usage
Included is the main script that can be used to find optimal algorithm configurations for a specific dataset.
MultiETSC uses the training set for algorithm configuration and can provide test performance on a specified test set.
MultiETSC is designed to use a 5 fold crossvalidation protocol for the algorithm configuration phase,
which requires the training set to include at least 5 examples of each class.
MultiETSC, having been developed with the [UCR Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)
in mind, will be able to run on by far the most UCR datasets out of the box.

We have included a few UCR datasets for testing which can be used to run a simple instance of MultiETSC. 
The following command will run on the `Coffee` dataset using 60 seconds for algorithm configuration with 
a max running time per algorithm of 1 second. 
Note that this is meant as a very short example, in our experiments we used 7200s configurator time
with a cutoff of 180s which might still be considered as little time.
```bash
$ MultiETSC/main --dataset test/data/Coffee_TRAIN.tsv  --test test/data/Coffee_TEST.tsv --timeout 60 --cutoff 1
```
This command, after multiple lines of progress output, returns the following result:
```
Running test evaluation:
Result: status, time, [earliness, error rate], 0, configuration
Result: SUCCESS, 0.0142195, [0, 0.464286], 0, -algorithm 'fixed/run.py' -percLen '0.0'
Result: SUCCESS, 0.010000, [0.719905, 0.000000], 0, -algorithm 'ECTS/bin/ects' -min_support '0.0' -version 'loose'
Result: SUCCESS, 0.119657, [0.101399, 0.107143], 0, -algorithm 'fixed/run.py' -percLen '0.1'
Result: SUCCESS, 0.119354, [0.178322, 0.178571], 0, -algorithm 'fixed/run.py' -percLen '0.18'
```
What can be seen here is the test evaluation of the four selected algorithm combinations.
Note that, while all four are non-dominated on the validation data,
some might be dominated when evaluated on the test data.
The fourth configuration is an example of this.