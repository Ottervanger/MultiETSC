# EarlyTSC

Implementation of early time series classification algorithms. The aim is to produce a python package that provides interface to these methods conforming to scikit-learn standards. This is done by keeping as much of the existing implementations intact as possible.

The following algorithms are (planned to be) included:

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

