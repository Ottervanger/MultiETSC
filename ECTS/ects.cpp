
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <math.h>
#include <set>
#include <map>
#include <algorithm>
#include <vector>
#include <time.h>
#include <limits>
#include <numeric>
#include <cassert>
#include <iomanip>
#include <sys/stat.h>

#include "util.h"

// ALgorithm parameters
const double MIN_SUPPORT = 0;

enum Version { STRICT, LOOSE };

namespace GLOBAL {
    int DIM = 0;
    std::string DATA_PATH;
    std::string DATA_NAME;
    std::string OUT_FILE = "";
    Version version = LOOSE;
}

inline std::string trainFile() {
    return GLOBAL::DATA_PATH + GLOBAL::DATA_NAME + "/" + GLOBAL::DATA_NAME+"_TRAIN.tsv";
}

inline std::string testFile() {
    return GLOBAL::DATA_PATH+GLOBAL::DATA_NAME + "/" + GLOBAL::DATA_NAME + "_TEST.tsv";
}

inline std::string distMatFile() {
    return "/tmp/" + GLOBAL::DATA_NAME + "_distMat";
}

inline std::string distIdxFile() {
    return "/tmp/" + GLOBAL::DATA_NAME + "_distIdx";
}

// global variable
std::vector<int> labelTrain; // training data class labels
std::vector<std::vector<int> > distIdx;//  store the 1NN for each space, no ranking tie
std::vector<std::vector<double> > distMat; //  the pairwise distance array of full length
std::vector<double> classSupport;
std::vector<int> fullLenCorrect;// 0: incorrectly, 1: correct
std::vector<int> predictionPrefix;
std::vector<std::set<int>> nnSetList;// store the MNCS;
std::vector<int> nnSetListLength;
std::vector<int> nnSetListClassMap;// store the class label of the MNCS
double classificationTime; // classification time of one instance
double trainingTime; // classification time of one instance

// changes classSupport to contain label frequency times minimalSupport for each label
void computeClassSupport(const std::vector<int> &labels,
                         std::vector<double> &classSupport,
                         const double minimalSupport) {
    std::map<int, int> labelCount;
    for (int label : labels)
        labelCount[label]++;
    for (auto c : labelCount)
        classSupport.push_back(c.second * minimalSupport);
}

void argparse(int argc, char* argv[]) {
    for (int i = 1; i < argc-1; i++) {
        if (argv[i][0] == '-') {
            switch (argv[i][(argv[i][1] == '-' ? 2 : 1)]) {
              case 'v': // version
                if (argv[i+1][0] == 's') {
                    GLOBAL::version = STRICT;
                } else if (argv[i+1][0] == 'l') {
                    GLOBAL::version = LOOSE;
                }
                i++;
                break;
              case 'p': // path to UCR datasets
                GLOBAL::DATA_PATH = argv[i+1];
                i++;
                break;
              case 'd': // name of the datset
                GLOBAL::DATA_NAME = argv[i+1];
                i++;
                break;
              case 'o': // output file
                GLOBAL::OUT_FILE = argv[i+1];
                i++;
                break;
            }
        }
    }
}

std::set<int> setRNN(int l, std::set<int>& s) { // find a set's RNN on prefix l
    std::set<int> index;

    std::set<int>::iterator i;
    for (i = s.begin(); i != s.end(); i++) {
        int element = *i;
        for (int j = 0; j < distIdx.size(); j++) {
            if (distIdx[j][l - 1] == element && s.count(j) == 0) {
                index.insert(j);
            }
        }
    }
    return index;
}

int nnConsistent(int l, std::set<int>& s) { // return 1, if it is NN consistent, return 0, if not
    std::set<int>::iterator i;
    int consistent = 1;
    for (i = s.begin(); i != s.end(); i++) {
        int element = *i;

        int NNofelement = distIdx[element][l - 1]; // get the NN

        if (s.count(NNofelement) == 0) {
            consistent = 0;
            break;
        }
    }
    return consistent;
}

int getMPL(std::set<int>& s) {
    int MPL = GLOBAL::DIM;

    std::set<int> ::iterator i;
    if (s.size() == 1) { // simple RNN Method
        std::set<int> LastRNN = setRNN(GLOBAL::DIM, s); // get full length RNN
        int FirstElement = *(s.begin());
        double label = labelTrain[FirstElement];
        int labelIndex = 0;
        if (label == -1) {labelIndex = 1;}
        else {labelIndex = label - 1;}
        // get the intersect(RNN(S,L), T')=RNN(S,L) in the same class as S.
        std::set<int> LastUsefulRNN;
        for (i = LastRNN.begin(); i != LastRNN.end(); i++) {
            int element = *i;
            if ( fullLenCorrect[element] == 1) {
                LastUsefulRNN.insert(element);
            }
        }
        // compute the support of the sequence
        int Support = s.size() + LastUsefulRNN.size();
        if (Support >= classSupport[labelIndex]) {
            if (LastUsefulRNN.size() > 0) {
                std::set<int> PreviousRNN;
                for (int le = GLOBAL::DIM - 1; le >= 1; le--) {
                    PreviousRNN = setRNN(le, s);
                    std::set<int> PreviousUsefulRNN;
                    for (i = PreviousRNN.begin(); i != PreviousRNN.end(); i++) {
                        int element = *i;
                        if (fullLenCorrect[element] == 1) {
                            PreviousUsefulRNN.insert(element);
                        }
                    }
                    if (  LastUsefulRNN == PreviousUsefulRNN ) {
                        PreviousRNN.clear();
                        PreviousUsefulRNN.clear();
                    } else {
                        MPL = le + 1;
                        break;
                    }
                }// end of for
            } else {MPL = GLOBAL::DIM;}
        } else {
            MPL = GLOBAL::DIM;
        }
    } else { // super-sequence Method
        std::set<int> LastRNN = setRNN(GLOBAL::DIM, s); // get full length RNN
        int FirstElement = *(s.begin());
        double label = labelTrain[FirstElement];
        int labelIndex = 0;
        if (label == -1) {labelIndex = 1;}
        else {labelIndex = label - 1;}
        // get the intersect(RNN(S,L), T')=RNN(S,L) in the same class as S.
        std::set<int> LastUsefulRNN;
        for (i = LastRNN.begin(); i != LastRNN.end(); i++) {
            int element = *i;
            if (fullLenCorrect[element] == 1) {
                LastUsefulRNN.insert(element);
            }
        }
        // compute the support of the sequence
        int Support = s.size() + LastUsefulRNN.size();
        if (Support >= classSupport[labelIndex]) {
            std::set<int> PreviousRNN;
            for (int le = GLOBAL::DIM - 1; le >= 1; le--) {
                if (nnConsistent(le, s) == 0) {
                    MPL = le + 1;
                    break;
                } else {
                    PreviousRNN = setRNN(le, s);
                    std::set<int> PreviousUsefulRNN;
                    for (i = PreviousRNN.begin(); i != PreviousRNN.end(); i++) {
                        int element = *i;
                        if ( fullLenCorrect[element] == 1) {
                            PreviousUsefulRNN.insert(element);
                        }
                    }
                    if ( LastUsefulRNN == PreviousUsefulRNN) {
                        PreviousRNN.clear();
                        PreviousUsefulRNN.clear();
                    } else {
                        MPL = le + 1;
                        break;
                    }
                }// end of else
            }// end of for
        } else {
            MPL = GLOBAL::DIM;
        }
    }
    return MPL;
}

int getMPLStrict(std::set<int>& s) {
    int MPL = GLOBAL::DIM;
    std::set<int> ::iterator i;
    if (s.size() == 1) { // simple RNN Method
        std::set<int> LastRNN = setRNN(GLOBAL::DIM, s); // get full length RNN
        int FirstElement = *(s.begin());
        double label = labelTrain[FirstElement];
        int labelIndex = 0;
        if (label == -1) {labelIndex = 1;}
        else {labelIndex = label - 1;}
        // get the intersect(RNN(S,L), T')=RNN(S,L) in the same class as S.
        std::set<int> LastUsefulRNN;
        for (i = LastRNN.begin(); i != LastRNN.end(); i++) {
            int element = *i;
            if ( fullLenCorrect[element] == 1) {
                LastUsefulRNN.insert(element);
            }
        }
        // compute the support of the sequence
        int Support = s.size() + LastUsefulRNN.size();
        if (Support >= classSupport[labelIndex]) {
            if (LastUsefulRNN.size() > 0) {
                std::set<int> PreviousRNN;
                for (int le = GLOBAL::DIM - 1; le >= 1; le--) {
                    PreviousRNN = setRNN(le, s);
                    std::set<int> PreviousUsefulRNN;
                    for (i = PreviousRNN.begin(); i != PreviousRNN.end(); i++) {
                        int element = *i;
                        if (fullLenCorrect[element] == 1) {
                            PreviousUsefulRNN.insert(element);
                        }
                    }
                    if (  LastUsefulRNN == PreviousUsefulRNN ) {
                        // CurrentUsefulRNN=PreviousUsefulRNN;
                        PreviousRNN.clear();
                        PreviousUsefulRNN.clear();
                    } else {
                        MPL = le + 1;
                        break;
                    }
                }// end of for
            } else {MPL = GLOBAL::DIM;}
        } else {
            MPL = GLOBAL::DIM;
        }
    } else { // super-sequence Method
        std::set<int> LastRNN = setRNN(GLOBAL::DIM, s); // get full length RNN
        int FirstElement = *(s.begin());
        double label = labelTrain[FirstElement];
        int labelIndex = 0;
        if (label == -1) {labelIndex = 1;}
        else {labelIndex = label - 1;}
        // compute the support of the sequence
        int Support = s.size() + LastRNN.size();
        if (Support >= classSupport[labelIndex]) {
            std::set<int> PreviousRNN;
            for (int le = GLOBAL::DIM - 1; le >= 1; le--) {
                if (nnConsistent(le, s) == 0) {
                    MPL = le + 1;
                    break;
                } else {
                    PreviousRNN = setRNN(le, s);
                    if ( LastRNN == PreviousRNN) {
                        PreviousRNN.clear();
                    } else {
                        MPL = le + 1;
                        break;
                    }
                }// end of else
            }// end of for
        } else {
            MPL = GLOBAL::DIM;
        }
    }
    return MPL;
}

int updateMPL(std::set<int> s) {

    int length = getMPL(s);

    std::set<int> ::iterator jj;
    for (jj = s.begin(); jj != s.end(); jj++) {
        if (predictionPrefix[*jj] > length) {
            predictionPrefix[*jj] = length;
        }
    }
    return length;
}

int updateMPLStrict(std::set<int> s) {

    int length = getMPLStrict(s);

    std::set<int> ::iterator jj;
    for (jj = s.begin(); jj != s.end(); jj++) {
        if (predictionPrefix[*jj] > length) {
            predictionPrefix[*jj] = length;
        }
    }
    return length;
}

double Euclidean(std::vector<double> a, std::vector<double> b, int length) {
    double ret = 0;
    for (int i = 0; i < length; i++) {
        double dist = a[i] - b[i];
        ret += dist * dist;
    }
    // since we only use the distance to order 
    // we can leave out taking the root
    return ret;
}

int findNN(std::vector<double> ts, int len,
           std::vector<std::vector<double> > dataTrain) {

    int indexOfNN = -1;
    double Mindis = 100000;
    for (int i = 0; i < dataTrain.size(); i++) {
        double tempdis = Euclidean(ts, dataTrain[i], len);
        if (tempdis < Mindis) {
            Mindis = tempdis;
            indexOfNN = i;
        }
    }
    return indexOfNN;
}


// used globals:
// 
// predictionPrefix

void classification(std::vector<std::vector<double> > dataTest,
                    std::vector<std::vector<double> > dataTrain,
                    std::vector<int> &labelPred,
                    std::vector<int> &predLen,
                    std::vector<int> labelTrain) {

    int startfrom = *std::min_element(predictionPrefix.begin(), predictionPrefix.end());

    // for each instance
    for (int i = 0; i < dataTest.size(); i++) {
        // for each observation
        // start from the smallest mimimum prefix in the train data
        for (int j = startfrom; j <= dataTest[i].size(); j++) {
            // find the nearest neigbour of i in j-prefix space
            int tempNN = findNN(dataTest[i], j, dataTrain);

            // trigger mechanism
            if (predictionPrefix[tempNN] <= j) {
                // label instance i
                labelPred[i] = labelTrain[tempNN];
                // record earliness
                predLen[i] = j;
                break;
            }
        }
    }
}

inline double mean(std::vector<int> data) {
    return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
}

void report(std::vector<int> labelTest, std::vector<int> labelPred, std::vector<int> predLen) {
    int correct = 0;
    // compute the false positive  and true positve
    int FP = 0, TP = 0, TC = 0, FC = 0, n = labelPred.size();
    for (int i = 0; i < n; i++) {
        if (labelPred[i] == 1 && labelTest[i] == -1) FP++;
        if (labelPred[i] == 1 && labelTest[i] ==  1) TP++;
        if (labelPred[i] == labelTest[i]) correct++;

        if (labelTest[i] == 1) {
            TC++;
        } else {
            FC++;
        }
    }

    double plTest  = mean(predLen);
    double plTrain = mean(predictionPrefix);
    double acc = double(correct) / n;

    double FPRate = (double)FP / FC;
    double TPRate = (double)TP / TC;

    std::ostringstream ss;
    ss << "Algorithm Report: using " << ((GLOBAL::version == LOOSE) ? "loose" : "strict") << " version\n"
       << std::setprecision(3) << std::fixed
       << "av. pred. len. test:  " << std::setw(7) << plTest  << "\n"
       << "av. pred. len. train: " << std::setw(7) << plTrain << "\n"
       << "accuracy:             " << std::setw(7) << acc << "\n"
       << std::setprecision(6) << std::fixed
       << "av. classif. time     " << std::setw(10) << classificationTime / n << "s\n"
       << "    training time     " << std::setw(10) << trainingTime << "s\n"
       << "FP rate:              " << std::setw(10) << FPRate << "\n"
       << "TP rate:              " << std::setw(10) << TPRate << "\n";

    // write output to console
    std::cout << ss.str();

    // write output to file
    if (GLOBAL::OUT_FILE != "") {
        std::ofstream outputFile(GLOBAL::OUT_FILE, std::ofstream::out | std::ofstream::app);
        outputFile << ss.str();
        outputFile.close();
    }
}


// Computes the set distance: min distance between two items each
// of a different set, based on the full len
double SetDis( std::set<int> A, std::set<int> B ) {

    double minimal = 10000000;

    std::set<int> :: iterator iA;
    std::set<int> :: iterator iB;

    for ( iA = A.begin(); iA != A.end(); iA++) {
        for (iB = B.begin(); iB != B.end(); iB++) {
            double temp = distMat[*iA][*iB];
            if (minimal > temp) {
                minimal = temp;
            }
        }
    }
    return minimal;
}

// get the NN of cluster, if there is no MN, return -1, others , return the value
int GetMN(std::vector<std::set<int>> & List, int ClusterIndex) {
    // find the NN of ClusterIndex
    double Mindis = 100000;
    int NN = -1;
    for (int i = 0; i < List.size(); i++) {

        if (i != ClusterIndex) {

            double tempdis = SetDis( List[ClusterIndex], List[i]);

            if (tempdis < Mindis) {
                Mindis = tempdis;
                NN = i;
            }
        }
    }

    // find the NN of MinIndex
    Mindis = 100000;
    int MinIndex = -1;
    for (int i = 0; i < List.size(); i++) {
        if (i != NN) {
            double tempdis = SetDis( List[NN], List[i]);
            if (tempdis < Mindis) {
                Mindis = tempdis;
                MinIndex = i;
            }
        }
    }

    if (MinIndex == ClusterIndex) {return NN;}
    else {return -1;}

}
int getMPL(std::set<int>& s, int la, int lb) {
    int startFrom = 0;

    if (la <= lb)
        startFrom = lb;
    else
        startFrom = la;

    int MPL = GLOBAL::DIM;

    std::set<int> LastRNN = setRNN(GLOBAL::DIM, s); // get full length RNN

    int FirstElement = *(s.begin());
    double label = labelTrain[FirstElement];
    int labelIndex = 0;
    if (label == -1) {labelIndex = 1;}
    else {labelIndex = label - 1;}

    // get the intersect(RNN(S,L), T')=RNN(S,L) in the same class as S.
    std::set<int> LastUsefulRNN;
    for (auto i = LastRNN.begin(); i != LastRNN.end(); i++) {
        int element = *i;
        if (fullLenCorrect[element] == 1) {
            LastUsefulRNN.insert(element);
        }
    }
    // compute the support of the sequence
    int Support = s.size() + LastUsefulRNN.size();
    std::set<int> PreviousRNN;
    std::set<int> PreviousUsefulRNN;
    if (Support >= classSupport[labelIndex]) {
        for (int le = startFrom - 1; le >= 1; le--) {
            if (nnConsistent(le, s) == 0) {
                MPL = le + 1;
                break;
            } else {
                PreviousRNN = setRNN(le, s);
                for (auto i = PreviousRNN.begin(); i != PreviousRNN.end(); i++) {
                    int element = *i;
                    if ( fullLenCorrect[element] == 1) {
                        PreviousUsefulRNN.insert(element);
                    }
                }
                if ( LastUsefulRNN == PreviousUsefulRNN) {
                    PreviousRNN.clear();
                    PreviousUsefulRNN.clear();
                } else {
                    MPL = le + 1;
                    break;
                }
            }
        }
    } else {
        MPL = GLOBAL::DIM;
    }
    return MPL;
}

int getMPLStrict(std::set<int>& s, int la, int lb) {
    int startFrom = 0;

    if (la <= lb)
        startFrom = lb;
    else
        startFrom = la;

    int MPL = GLOBAL::DIM;
    std::set<int> LastRNN = setRNN(GLOBAL::DIM, s); // get full length RNN

    int FirstElement = *(s.begin());
    double label = labelTrain[FirstElement];
    int labelIndex = 0;
    if (label == -1) {labelIndex = 1;}
    else {labelIndex = label - 1;}

    // compute the support of the sequence
    int Support = s.size() + LastRNN.size();
    std::set<int> PreviousRNN;

    if (Support >= classSupport[labelIndex]) {
        for (int le = startFrom - 1; le >= 1; le--) {
            if (nnConsistent(le, s) == 0) {
                MPL = le + 1;
                break;
            } else {
                PreviousRNN = setRNN(le, s);
                if ( LastRNN == PreviousRNN) {
                    PreviousRNN.clear();
                } else {
                    MPL = le + 1;
                    break;
                }
            }
        }
    } else {
        MPL = GLOBAL::DIM;
    }
    return MPL;
}

void computeDist(const std::vector<std::vector<double> > &data,
                 std::vector<std::vector<int> > &distIdx,
                 std::vector<std::vector<double> > &dist) {
    int n = data.size();
    // currentMinDis[i][l] stores the distance to i's nn at prefix l
    std::vector<std::vector<double> > currentMinDis(n);
    // distIdx[i][l] stores the index of that nn
    distIdx.resize(n);
    // dist[i][j] stores the full length distance between i and j
    dist.resize(n, std::vector<double>(n, -1));

    for (int i = 0; i < n; i++) {
        currentMinDis[i].resize(data[i].size(), std::numeric_limits<double>::infinity());
        distIdx[i].resize(data[i].size());
    }

    // compute the pairwise distances
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double prefixSqDist = 0;
            int L = std::min(data[i].size(), data[j].size());
            for (int l = 0; l < L; l++) {
                double prefixDiff = data[i][l] - data[j][l];
                prefixSqDist += prefixDiff * prefixDiff;
                if (prefixSqDist < currentMinDis[i][l]) {
                    currentMinDis[i][l] = prefixSqDist;
                    distIdx[i][l] = j;
                }
                if (prefixSqDist < currentMinDis[j][l]) {
                    currentMinDis[j][l] = prefixSqDist ;
                    distIdx[j][l] = i;
                }
            }
            // full length distance
            dist[i][j] = dist[j][i] = sqrt(prefixSqDist);
        }
    }
}

inline bool exists (const std::string &file) {
      struct stat buffer;
      return (stat (file.c_str(), &buffer) == 0);
}

int main (int argc, char* argv[]) {
    argparse(argc, argv);
    // load training data
    std::vector<std::vector<double> > dataTrain;
    bool ok;
    ok = util::readUCRData(trainFile(), dataTrain, labelTrain);
    if (!ok) exit(1);
    computeClassSupport(labelTrain, classSupport, MIN_SUPPORT);

    if (exists(distMatFile()) && exists(distIdxFile())) {
        ok = util::readDMatrix(distMatFile(), distMat);
        ok &= util::readDMatrix(distIdxFile(), distIdx);
        if (!ok) exit(1);
    } else {
        computeDist(dataTrain, distIdx, distMat);
        util::saveMatrix(distIdxFile(), distIdx);
        util::saveMatrix(distMatFile(), distMat);
    }
    
    // compute the full length classification status, 0 incorrect, 1 correct
    clock_t t3, t4;

    GLOBAL::DIM = dataTrain[0].size();

    int n = dataTrain.size();
    fullLenCorrect.resize(n, 0);
    for (int i = 0; i < n; i++) {
        int NNofi = distIdx[i].back();
        if (labelTrain[i] == labelTrain[NNofi]) {
            fullLenCorrect[i] = 1;
        }
    }

    predictionPrefix.resize(n);
    // intialize the prediction prefix
    for (int i = 0; i < n; i++) {
        predictionPrefix[i] = dataTrain[i].size();
    }

    // simple RNN method
    t3 = clock();
    // simple RNN method
    for (int i = 0; i < n; i++) {
        std::set<int>  s;
        s.insert(i);

        if (GLOBAL::version == LOOSE) {
            predictionPrefix[i] = getMPL(s);
        } else {
            predictionPrefix[i] = getMPLStrict(s);
        }
    }

    int latticeLevel = 1;

    std::vector<std::set<int>> biRoots; // is a vector of sets
    std::vector<int> biRootsLength;// the length of the node
    std::vector<double> setLabel;// -,1,2,3,4,5,6,   0 means no label
    // initial the length to L
    std::vector<int> currentNew;
    std::vector<int> nextNew;

    std::vector<bool> UsedInBiRoots(n, 0); // 0 not used, 1 used

    for (int i = 0; i < n; i++) {
        if (UsedInBiRoots[i] == 0) {
            int NNofi = distIdx[i].back();
            // get bi-root
            if (distIdx[NNofi].back() == i ) {
                std::set<int> temp;
                temp.insert(i);
                temp.insert(NNofi);

                if (labelTrain[i] == labelTrain[NNofi]) {
                    int tempL;
                    if (GLOBAL::version == LOOSE) {
                        tempL = updateMPL(temp);
                    } else {
                        tempL = updateMPLStrict(temp);
                    }

                    biRootsLength.push_back(tempL);
                    setLabel.push_back(labelTrain[NNofi]);
                    biRoots.push_back(temp);
                    currentNew.push_back(biRoots.size() - 1);

                } else { // unpure group
                    biRootsLength.push_back(0);

                    setLabel.push_back(0);
                    biRoots.push_back(temp);
                    currentNew.push_back(biRoots.size() - 1);
                }

                UsedInBiRoots[i] = 1;
                UsedInBiRoots[NNofi] = 1;
            } else { // single element std::set
                std::set<int> temp;
                temp.insert(i);
                biRootsLength.push_back(predictionPrefix[i]);
                setLabel.push_back(labelTrain[i]);
                biRoots.push_back(temp);
            }
        }
    } // end for

    // Heuristic algorithm
    latticeLevel = 2;
    int mergedPair = 0;

    std::vector<std::set<int>> nnSetListCurrent = biRoots;
    std::vector<int> nnSetListCurrentLength = biRootsLength;
    std::vector<double> nnSetListCurrentLabel = setLabel;

    std::vector<std::set<int>> nnSetListNext;
    std::vector<int> nnSetListNextLength;
    std::vector<double> nnSetListNextLabel;

    while (nnSetListCurrent.size() > 1) {
        mergedPair = 0;
        nnSetListNext.clear();
        nnSetListNextLength.clear();
        nnSetListNextLabel.clear();
        nextNew.clear();

        std::vector<int> status;
        for (int i = 0; i < nnSetListCurrent.size(); i++) {
            status.push_back(0);
        }

        for (int i = 0; i < currentNew.size(); i++) {
            int tempi = currentNew[i];
            if (status[tempi] == 0) {
                std::vector<int> pair;
                int MNofi = GetMN(nnSetListCurrent, tempi);
                if (MNofi != -1) {

                    pair.push_back(tempi);
                    pair.push_back(MNofi);

                    // test if they are of the same label
                    double label1 = nnSetListCurrentLabel[pair[0]];
                    double label2 = nnSetListCurrentLabel[pair[1]];

                    std::set<int> tempSet;
                    std::set<int> setA = nnSetListCurrent[pair[0]];
                    std::set<int> setB = nnSetListCurrent[pair[1]];

                    std::merge(setA.begin(), setA.end(),
                               setB.begin(), setB.end(),
                               std::inserter(tempSet, tempSet.begin()));

                    if (label1 != 0 && label2 != 0 && label1 == label2 ) {
                        int tempLength;

                        if (setA.size() > 1 && setB.size() > 1) {
                            if (GLOBAL::version == LOOSE) {
                                tempLength = getMPL(tempSet, nnSetListCurrentLength[pair[0]], nnSetListCurrentLength[pair[1]]);
                            } else {
                                tempLength = getMPLStrict(tempSet, nnSetListCurrentLength[pair[0]], nnSetListCurrentLength[pair[1]]);
                            }
                        } else {
                            if (GLOBAL::version == LOOSE) {
                                tempLength = getMPL(tempSet);
                            } else {
                                tempLength = getMPLStrict(tempSet);
                            }
                        }

                        // update the length

                        std::set<int> ::iterator jj;
                        for (jj = tempSet.begin(); jj != tempSet.end(); jj++) {
                            if (predictionPrefix[*jj] > tempLength) {
                                predictionPrefix[*jj] = tempLength;
                            }
                        }

                        nnSetListNextLength.push_back(tempLength);
                        nnSetListNext.push_back(tempSet);
                        nextNew.push_back(nnSetListNext.size() - 1);

                        nnSetListNextLabel.push_back(label1);
                        mergedPair++;
                        status[pair[0]] = 1;
                        status[pair[1]] = 1;

                    } else { // the pair is not pure
                        nnSetListNextLength.push_back(0);
                        nnSetListNext.push_back(tempSet);
                        nextNew.push_back(nnSetListNext.size() - 1);

                        nnSetListNextLabel.push_back(0);
                        status[pair[0]] = 1;
                        status[pair[1]] = 1;
                    }
                }
            }// end if
        }// end for

        if (mergedPair > 0) {
            for (int k = 0; k < nnSetListCurrent.size(); k++) {
                if (status[k] == 0) {
                    nnSetListNext.push_back(nnSetListCurrent[k]);
                    nnSetListNextLength.push_back(nnSetListCurrentLength[k]);
                    nnSetListNextLabel.push_back(nnSetListCurrentLabel[k]);
                }
            }// end for

            nnSetListCurrent.clear();
            for (int kk = 0; kk < nnSetListNext.size(); kk++) {
                nnSetListCurrent.push_back(nnSetListNext[kk]);
            }
            nnSetListNext.clear();

            nnSetListCurrentLength.clear();
            for (int kk = 0; kk < nnSetListNextLength.size(); kk++) {
                nnSetListCurrentLength.push_back(nnSetListNextLength[kk]);
            }
            nnSetListNextLength.clear();

            nnSetListCurrentLabel.clear();
            for (int kk = 0; kk < nnSetListNextLabel.size(); kk++) {
                nnSetListCurrentLabel.push_back(nnSetListNextLabel[kk]);
            }
            nnSetListNextLabel.clear();

            // update currentNew and nextNew
            currentNew.clear();
            for (int kk = 0; kk < nextNew.size(); kk++) {
                currentNew.push_back(nextNew[kk]);
            }
            nextNew.clear();

            latticeLevel++;

        } else {
            break;
        }

    }// end while

    // end of heuristic algorithm
    t4 = clock();
    trainingTime = (double)(t4 - t3) / CLOCKS_PER_SEC  ;

    clock_t t;
    t = clock();

    std::vector<std::vector<double> > dataTest;
    std::vector<int> labelTest;
    util::readUCRData(testFile(), dataTest, labelTest);

    std::vector<int> labelPred(labelTest.size());
    std::vector<int> predLen(labelTest.size()); // predicted length by the classifier
    classification(dataTest, dataTrain, labelPred, predLen, labelTrain);

    classificationTime = ((double)(clock() - t))/CLOCKS_PER_SEC;
    report(labelTest, labelPred, predLen);
}// end main