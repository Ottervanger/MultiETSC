// this version is for the non-redundant top-K feature selection
// in this version, ranked by earliness sctore, and classifying all the classes together,
// the default classifier has  been added as the majority class
// output into the text file

#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cstring>
#include "math.h"
#include <algorithm>
#include <numeric>
#include <vector>
#include <unordered_map>
#include <time.h>
#include <climits>
#include <cfloat>
#include "util.h"



struct BitArray {
    std::vector<bool> a;
    BitArray() { }
    BitArray(std::size_t size) : a(size, false) { }
    bool operator[] (std::size_t pos) { return a[pos]; }
    BitArray operator& (BitArray ba) {
        BitArray r(std::max(ba.a.size(), a.size()));
        for (int i = 0; i < a.size() && i < ba.a.size(); ++i) {
            r.a[i] = a[i] & ba.a[i];
        }
        return r;
    }
    void set(std::size_t pos) { a[pos] = true; }
    void reset(std::size_t pos) { a[pos] = false; }
    int count() { int c = 0; for (bool v : a) { c += v; } return c;}
    size_t size() { return a.size(); }
    void resize(size_t n) { return a.resize(n); }
};

// structure of a feature
struct Feature {
    //double * f;
// int segmentIndex;
    int instanceIndex;
    int startPosition;
    int length;
    double label;
    double threshold;
    double recall;
    double precision;
    double fscore;
    double earlyFScore;
    BitArray bmap;  // std::bitset does not change
};

struct Result {
    double earliness, errorRate, time;
};

struct Config {                 // data structure storing all hyperparameters
    const char* trainingFileName;
    const char* testingFileName;
    int thresholdMethod = 1;            // thresholdMethod =1 using thresholdAll, thresholdMethod=2, using the KDE cut
    double boundThreshold = 3;           // define the parameter of the Chebyshev's inequality; only used if thresholdMethod == 1
    int alpha = 3;                      // only used if thresholdMethod == 1
    double probablityThreshold = 0.95;  // only used if thresholdMethod == 2
    int DisArrayOption = 2;             // =1 , naive version =2 fast version
    int minK = 5;                       // minimal feature length
    int maxK = 50;                      // maximal length
    double recallThreshold = 0;
};

// global variable
std::vector<std::vector<double> > training;         // training data set
std::vector<int> labelTraining;                     // training data class labels
std::vector<std::vector<double> > testing;          // testing data set
std::vector<int> labelTesting;                      // testing data class labels
std::vector<Feature *> finalFset;
std::vector<Feature *> AllClassesFset;


// functions in the same file
void DisArray(int instanceIndex, double ** DisArray, int ** EndP, int n, size_t tsLen, int minK, int maxK);
void DisArray2(int instanceIndex, double ** DisArray, int ** EndP, int n, size_t tsLen, int minK, int maxK);
double * getFeatureSegment(Feature *f);
void reduceFset(std::vector<Feature *> &finalFset, std::vector<Feature *> &fSet, BitArray &bmap);

double ComputeFScore(double recall, double precision);
Feature * ThresholdLearningAll(int instanceIndex, int startPosition,  double m, int k, int targetClass, double * DisAi, double RecallThreshold, int * EndPi, int alpha, const std::vector<int> &labelTraining);
Feature * ThresholdLearningKDE(int Instanceindex, int startPosition, int k, int targetClass, double * DisAi, double RecallThreshold, double ProbalityThreshold, int * EndPi, int alpha, const std::vector<int> &labelTraining);
Result classificationAllClasses(std::vector<Feature *> &Fs, const std::unordered_map<int,int> &classes, size_t tsLen) ;

void computeWholeSubstringArray(int iLengthOfArray1,
                                int iLengthOfArray2,
                                double * pMatrixOriginal,
                                double * pArraySubstring, double * pArraySubstringEndingPosition, int MaximalLength);

Feature * computeBmap(Feature * currentf, int targetClass, int nTarget, double * DisAi, int * EndPi, double RecallThreshold, int alpha);

// compute the mean
inline double getMean(std::vector<double> v) { // segmentIndex starting from 1
    return std::accumulate( v.begin(), v.end(), 0.0) / v.size();
}

// compute the standard deviation given the mean
inline double getSTD(std::vector<double> v, double mean) {
    std::vector<double> d(v.size());
    std::transform(v.begin(), v.end(), d.begin(), [mean](double x) { return x - mean; });
    double ss = std::inner_product(d.begin(), d.end(), d.begin(), 0.0);
    return std::sqrt(ss / v.size());
}

double OneKDE(std::vector<double> v, double q, double h, double constant) {
    return std::accumulate(v.begin(), v.end(), 0.0,
        [q,h](const double &a, const double &b) {
            return a + std::exp((b - q) * (b - q) * (-1 / (2 * h * h)));
        }) * constant;
}

Result trainAndClassify(const Config &conf) {
    clock_t tStart = clock();
    // load data
    util::readUCRData(conf.trainingFileName, training, labelTraining);
    util::readUCRData(conf.testingFileName, testing, labelTesting);

    std::unordered_map<int,int> classes;
    for (int c : labelTraining)
        classes[c]++;
    size_t tsLen = training[0].size();

    for (auto it : classes) {
        int cl = it.first;
        std::vector<Feature *> reducedFset; // restore the set of the first round set cover

        // initialize the DisA and EndP
        for (int tIndex = 0; tIndex < labelTraining.size(); tIndex++) {
            if (labelTraining[tIndex] != cl)
                continue;
            // loop over instances of class cl

            int NumberOfSegments = 0;
            for (int i = tsLen - conf.minK + 1; i >= tsLen - conf.maxK + 1; i--) {
                NumberOfSegments = NumberOfSegments + i;
            }

            double ** DisA = new double *[NumberOfSegments];
            int ** EndP = new int *[NumberOfSegments];
            for (int i = 0; i < NumberOfSegments; i++) {
                DisA[i] = new double[training.size()];
                EndP[i] = new int[training.size()]; // ending position array
            }

            if (conf.DisArrayOption == 2)
                DisArray2(tIndex,  DisA, EndP, training.size(), tsLen, conf.minK, conf.maxK);
            else
                DisArray(tIndex,  DisA, EndP, training.size(), tsLen, conf.minK, conf.maxK);

            // compute
            std::vector<Feature *> fSet;
            int DisArrayIndex = 0;
            for (int currentL = conf.minK; currentL <= conf.maxK; currentL++ ) { // for each length

                for (int startPosition = 0; startPosition <= tsLen - currentL; startPosition++) {

                    Feature * temp;
                    if (conf.thresholdMethod == 1) {
                        temp = ThresholdLearningAll(tIndex, startPosition, conf.boundThreshold,
                            currentL, cl, DisA[DisArrayIndex], conf.recallThreshold,
                            EndP[DisArrayIndex], conf.alpha, labelTraining);
                    } else {
                        temp = ThresholdLearningKDE(tIndex, startPosition, currentL, cl,
                            DisA[DisArrayIndex], conf.recallThreshold, conf.probablityThreshold,
                            EndP[DisArrayIndex], conf.alpha, labelTraining);
                    }

                    if (temp != NULL) {
                        fSet.push_back(temp);
                    }
                    DisArrayIndex++;
                }// end of position for
            } // end of length for

            BitArray totalBmap(labelTraining.size());                      // the union bit map of a certain length of a certain class
            for (Feature * p : fSet)
                for (int i = 0; i < (p->bmap).size(); i++)
                    if ((p->bmap)[i])
                        totalBmap.set(i);  //  set the total Bmap for each length's set cover


            reduceFset(reducedFset, fSet, totalBmap);
            // relase the memory
            for (int i = 0; i < NumberOfSegments; i++) {
                delete [] DisA[i];
                delete [] EndP[i];
            }
            delete [] DisA;
            delete [] EndP;
        } // end of outer for, end of feature for each isntance

        // second round of set cover

        BitArray allLengthBmap(labelTraining.size());                  // the union bit map of all length of a certain class
        for (Feature * p : reducedFset)
            for (int i = 0; i < (p->bmap).size(); i++)
                if ((p->bmap)[i])
                    allLengthBmap.set(i); // set the all length Bmap for the second round of set cover
        reduceFset(finalFset, reducedFset, allLengthBmap);

        // UPDATE THE ALL CLASS DATA SET
        AllClassesFset.insert(AllClassesFset.end(), finalFset.begin(), finalFset.end());
        finalFset.clear();
    }// end of for for each classes

    Result res = classificationAllClasses(AllClassesFset, classes, tsLen);
    res.time = (double)(clock() - tStart) / CLOCKS_PER_SEC;
    return res;
}

// feature selection by top-K non-redundant
void reduceFset(std::vector<Feature *> &finalFset, std::vector<Feature *> &fSet, BitArray &bmap) {
    // until all instances that can be classified are correctly classified by at least one feature
    while (bmap.count() > 0 && fSet.size()) {
        // find the most promising feature
        double max = -1;
        double maxLength = -1;
        int index = -1;
        for (unsigned int i = 0; i < fSet.size(); i++) {
            double temp = fSet[i]->earlyFScore;
            if (temp > max ) {
                max = temp;
                maxLength = fSet[i]->length;
                index = i;
            } else if  (temp == max && fSet[i]->length > maxLength) {
                max = temp;
                maxLength = fSet[i]->length;
                index = i;
            }
        }
        // move this feature to reducedFset and reset others
        // check if the intersection of the two sets is not empty
        if ((fSet[index]->bmap & bmap).count() > 0) {
            finalFset.push_back(fSet[index]);
            for (unsigned int j = 0; j < (fSet[index]->bmap).size(); j++) { // for the current set, update the other set
                if (fSet[index]->bmap[j]) {
                    bmap.reset(j);  // update the total covered set
                }
            }
        }
        fSet.erase(fSet.begin() + index);
    }// end while
    for (Feature * p : fSet)
        delete p;
}

Result classificationAllClasses(std::vector<Feature *> &Fs, const std::unordered_map<int,int> &classes, size_t tsLen) {
    int CorrectlyClassified = 0;
    int sumLength = 0;

    int defaultlabel = 0;
    int maxFreq = 0;
    for (auto& it : classes) {
        if (it.second > maxFreq) {
            maxFreq = it.second;
            defaultlabel = it.first;
        }
    }

    int n = labelTesting.size();
    std::vector<int> predictedLabel(n);
    for (int i = 0; i < n; i++) {
        bool matched = 0;
        for (int j = 0; j < tsLen; j++) { // j is the current ending position of the stream
            for (unsigned int f = 0; f < Fs.size(); f++) {
                int tempLength = Fs[f]->length;
                int startingPosition = j - tempLength + 1;
                if (startingPosition >= 0) {
                    double * currentseg = new double[tempLength];
                    for (int ss = 0; ss < tempLength; ss++)
                    {currentseg[ss] = testing[i][ss + startingPosition];}
                    double * tempFeatureSeg = getFeatureSegment(Fs[f]);
                    double tempDis = util::euclidean(tempFeatureSeg, currentseg, tempLength);
                    delete [] tempFeatureSeg;
                    delete [] currentseg;
                    if (tempDis <= (Fs[f]->threshold)) {
                        predictedLabel[i] = (int)Fs[f]->label;
                        matched = 1;
                        sumLength = sumLength + j + 1;
                        if (Fs[f]->label == labelTesting[i]) {
                            CorrectlyClassified++;
                        }
                        break;
                        // break to stop checking more features
                    }
                }
            }
            if (matched == 1) // break the segment loop, finish the current testing example
                break;
        }
        if (matched == 0) { // classified by the default classifier
            if (labelTesting[i] == defaultlabel)
                CorrectlyClassified++;
            sumLength += tsLen;
        }
    }

    Result res;
    res.errorRate = 1. - ((double)CorrectlyClassified / n);
    res.earliness =  ((double)sumLength / tsLen) / n;
    return res;
}

double ComputeFScore(double recall, double precision) {
    return 2 * recall * precision / (recall + precision);
}


// learning threshold based on the one tail Chebyshev's inequality
// this index starting from 1, m is the parameter in the bound, k is the length of feature
Feature * ThresholdLearningAll(int instanceIndex, int startPosition,  double m, int k, int targetClass, double * DisAi, double RecallThreshold, int * EndPi, int alpha, const std::vector<int> &labelTraining) {
    Feature * currentf = new Feature();
    currentf->instanceIndex = instanceIndex;
    currentf->startPosition = startPosition;
    currentf->length = k;
    currentf->label = targetClass;
    currentf->bmap.resize(labelTraining.size());

    std::vector<double> nonTargetDis;
    
    for (int i = 0; i < labelTraining.size(); ++i)
        if (labelTraining[i] != targetClass)
            nonTargetDis.push_back(DisAi[i]);

    // compute the mean, standard deviation and the threshold
    double mu = getMean(nonTargetDis);
    double sd = getSTD(nonTargetDis, mu);

    currentf->threshold = mu - m * sd;

    // compute recall, precision and bitmap
    return computeBmap(currentf, targetClass, labelTraining.size()-nonTargetDis.size(), DisAi, EndPi, RecallThreshold, alpha);
}

//learning the threshold by KDE classification.
Feature * ThresholdLearningKDE(int Instanceindex, int startPosition, int k, int targetClass, double * DisAi, double RecallThreshold, double ProbalityThreshold, int * EndPi, int alpha, const std::vector<int> &labelTraining) {
    Feature * currentf = new Feature();
    currentf->instanceIndex = Instanceindex;
    currentf->startPosition = startPosition;
    currentf->length = k;
    currentf->bmap.resize(labelTraining.size());

    currentf->label = targetClass;

    std::vector<double> nonTargetDis;
    std::vector<double> TargetDis;
    std::vector<double> CurrentDis(DisAi, DisAi + labelTraining.size());
    
    for (int i = 0; i < labelTraining.size(); ++i) {
        if (labelTraining[i] == targetClass) {
            TargetDis.push_back(DisAi[i]);
        } else {
            nonTargetDis.push_back(DisAi[i]);
        }
    }

    // compute the mean, standard deviation and the threshold, and optimal h
    //  for the nonTarget classes
    double muNonTarget = getMean(nonTargetDis);

    double sdNonTarget = getSTD(nonTargetDis, muNonTarget);

    double hNonTarget = 1.06 * sdNonTarget / pow (nonTargetDis.size(), 0.2);

    double constantNT = 1 / (sqrt(2 * 3.14159265) * nonTargetDis.size() * hNonTarget);
    //  for the TargetClasses
    double muTarget = getMean(TargetDis);

    double sdTarget = getSTD(TargetDis, muTarget);

    double hTarget = 1.06 * sdTarget / pow (TargetDis.size(), 0.2);
    double constantT = 1 / (sqrt(2 * 3.14159265) * TargetDis.size() * hTarget);

    // sort the totalDis
    std::sort(CurrentDis.begin(), CurrentDis.end());
    //  compute the Probablity<0;
    int n = labelTraining.size();
    double NegativeTestPoint = -CurrentDis[n - 1] / (n - 1);
    double densityNonTarget = OneKDE(nonTargetDis, NegativeTestPoint, hNonTarget,  constantNT);
    double densityTarget = OneKDE(TargetDis, NegativeTestPoint, hTarget,  constantT);
    double tempTarget = ((double)TargetDis.size() / n) * densityTarget;
    double tempNonTarget = (1 - ((double)TargetDis.size() / n)) * densityNonTarget;
    double ProTarget = tempTarget / ( tempTarget + tempNonTarget);

    if (ProTarget > ProbalityThreshold) {
        // compute the breaking Index
        int breakIndex = 0;
        int i = 0;
        for (i = 0; i < n; i++) {
            densityNonTarget = OneKDE(nonTargetDis, CurrentDis[i], hNonTarget,  constantNT);
            densityTarget = OneKDE(TargetDis, CurrentDis[i], hTarget,  constantT);
            tempTarget = ((double)TargetDis.size() / n) * densityTarget;
            tempNonTarget = (1 - ((double)TargetDis.size() / n)) * densityNonTarget;
            ProTarget = tempTarget / ( tempTarget + tempNonTarget);
            if (ProTarget < ProbalityThreshold) { // belong to the non-target class
                breakIndex = i;
                break;
            }
        }
        //   compute the breaking point between breakingIndex and the previous point
        if (breakIndex >= 1) {
            int NonofBreakingPoint = 20;
            double value = 0;
            for (value = CurrentDis[breakIndex - 1]; value < CurrentDis[breakIndex]; value = value + (CurrentDis[breakIndex] - CurrentDis[breakIndex - 1]) / NonofBreakingPoint) {
                densityNonTarget = OneKDE(nonTargetDis, value, hNonTarget,  constantNT);
                densityTarget = OneKDE(TargetDis, value, hTarget,  constantT);
                tempTarget = ((double)TargetDis.size() / n) * densityTarget;
                tempNonTarget = (1 - ((double)TargetDis.size() / n)) * densityNonTarget;
                ProTarget = tempTarget / ( tempTarget + tempNonTarget);
                if (ProTarget < ProbalityThreshold) { // belong to the non-target class
                    currentf->threshold = value;
                    break;
                }

            }
            if (value >= CurrentDis[breakIndex]) {
                currentf->threshold = CurrentDis[breakIndex];
            }
        } else {
            currentf->threshold = -1;
        }

    } else {
        currentf->threshold = -1;
    }

    return computeBmap(currentf, targetClass, TargetDis.size(), DisAi, EndPi, RecallThreshold, alpha);
}

Feature * computeBmap(Feature * currentf, int targetClass, int nTarget, double * DisAi, int * EndPi, double RecallThreshold, int alpha) {
    if (currentf->threshold <= 0) {
        delete currentf;
        return NULL;
    }
    int targetCount = 0;
    double weightedRecall = 0;
    int totalCount = 0;
    for (int i = 0; i < labelTraining.size(); i++) {
        double temp = DisAi[i];
        if (temp <= currentf->threshold) {
            totalCount++;
            if (labelTraining[i] == targetClass) {
                targetCount++;
                (currentf->bmap).set(i);  // set the bmap
                weightedRecall = weightedRecall + pow(((double)1 / EndPi[i]), (double)1 / alpha);
            }
        }
    }
    currentf->recall = (double)targetCount / nTarget; // it is the absolute recall
    currentf->precision = (double)targetCount / totalCount;
    currentf->fscore = ComputeFScore(currentf->recall, currentf->precision);
    currentf->earlyFScore = ComputeFScore(weightedRecall, currentf->precision);

    if (currentf->recall < RecallThreshold) {
        delete currentf;
        return NULL;
    }
    return currentf;
}

// compute the best match distance array of the selected class and the ending position
//  the instanceIndex is the absolute instanceIndex
void DisArray(int instanceIndex, double ** DisArray, int ** EndP, int n, size_t tsLen, int minK, int maxK) {
    // compute the DisArray
    for (int tl = 0; tl < n; tl++) { // for each training example
        // listing substring, for each lenght
        int DisArrayIndex = 0;

        for (int currentL = minK; currentL <= maxK; currentL++ ) { // for each length

            for (int startPosition = 0; startPosition <= tsLen - currentL; startPosition++) {
                // compute the best match using naive early stopping
                // compute the best match
                int bestMatchStartP = -1;
                double minDis = FLT_MAX;
                for (int l = 0; l < tsLen - currentL + 1; l++) { // a possible match's starting position
                    double ret = 0;
                    for (int ii = 0; ii < currentL; ii++) {
                        double dist = training[instanceIndex][startPosition + ii] - training[tl][l + ii];
                        ret += dist * dist;
                        if (ret >= minDis) {
                            break; // early stopping
                        }
                    }
                    if (ret < minDis) {
                        minDis = ret;
                        bestMatchStartP = l;
                    }
                } // end of best match

                DisArray[DisArrayIndex][tl] = sqrt(minDis);
                EndP[DisArrayIndex][tl] = bestMatchStartP + currentL; // starting from position 1, since 0 will cause infinity
                DisArrayIndex++;
            } // end of startPosition for
        }// end of length for
    }// end of each training example for
} // end of function

double * getFeatureSegment(Feature *f) {
    double * temp = new double[f->length];
    for (int i = 0; i < f->length; i++) {
        temp[i] = training[f->instanceIndex][f->startPosition + i];
    }
    return temp;
}

//the following code compute the DisArray
// compute original matrix
double * createOriginalMatrix(const std::vector<double> &array1,
                              const std::vector<double> &array2) {
    double * pMatrix = new double[array1.size()*array2.size()];
    for (int j = 0; j < array2.size(); j++) {
        for (int i = 0; i < array1.size(); i++) {
            pMatrix[j * array1.size() + i] = (array1[i] - array2[j]) * (array1[i] - array2[j]);
        }
    }
    return pMatrix;
}

void computeWholeSubstringArray(int iLengthOfArray1,
                                int iLengthOfArray2,
                                double * pMatrixOriginal,
                                double * pArraySubstring, int * pArraySubstringEndingPosition, int MaximalLength) {
    if (pArraySubstring == NULL)
        return;

    double* tempMatrix = new double[iLengthOfArray1 * iLengthOfArray2]{0};
    int iIndexofSubstring = 0;
    int i, j, k;

    //from the length of 1 ot length of Array1
    for (i = 0; i < MaximalLength; i++) { // this is the different length of segment
        //update the tempMatrix
        for (j = 0; j < iLengthOfArray1 - i; j++) { // the start position of array1
            double fMinimum = 0;
            int iMinimumEndingPosition = 0;
            for (k = 0; k < iLengthOfArray2 - i; k++) {

                double fTempValue = tempMatrix[k * iLengthOfArray1 + j] + pMatrixOriginal[(k + i) * iLengthOfArray1 + (j + i)];
                if ( k == 0 ) {
                    fMinimum = fTempValue;
                    iMinimumEndingPosition = k + i + 1;
                } else if (fTempValue < fMinimum) {
                    fMinimum = fTempValue;
                    iMinimumEndingPosition = k + i + 1;
                }
                tempMatrix[k * iLengthOfArray1 + j] = fTempValue;
            }
            pArraySubstring[iIndexofSubstring] = sqrt(fMinimum);
            pArraySubstringEndingPosition[iIndexofSubstring] = iMinimumEndingPosition;
            iIndexofSubstring++;
        }
    }
    delete [] tempMatrix;
}
// compute the best match distance array of the selected class and the ending position
//  the instanceIndex is the absolute instanceIndex
void DisArray2(int instanceIndex, double ** DisArray, int ** EndP, int n, size_t tsLen, int minK, int maxK) {
    // compute the DisArray
    int pLengthOfSubstringArray = 0;
    for (int c = tsLen; c >= tsLen - maxK + 1; c--)
        pLengthOfSubstringArray += c;
    double* pArraySubstring = new double[pLengthOfSubstringArray];
    int* pArraySubstringEndingPosition = new int[pLengthOfSubstringArray];
    // how many from 1 to minK
    int offset = 0;
    for (int c = tsLen; c >= tsLen - (minK - 1) + 1; c--)
        offset += c;

    for (int tl = 0; tl < n; tl++) { // for each training example
        double * pMatrixOriginal = createOriginalMatrix(training[instanceIndex], training[tl]);
        computeWholeSubstringArray(training[instanceIndex].size(), training[tl].size(), pMatrixOriginal, pArraySubstring, pArraySubstringEndingPosition, maxK);

        // copy into the disarray and the pend array
        for (int i = offset; i < pLengthOfSubstringArray; i++) {
            DisArray[i - offset][tl] = pArraySubstring[i];
            EndP[i - offset][tl] = pArraySubstringEndingPosition[i];
        }

        // release the memory
        delete[] pMatrixOriginal;
    }// end of each training example for
    delete[] pArraySubstring;
    delete[] pArraySubstringEndingPosition;
} // end of function

Config argparse(int argc, char* argv[]) {
    Config conf;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-') {
            const char * arg = argv[i] + (argv[i][1] == '-' ? 2 : 1);
            if (strcmp(arg, "data") == 0) {
                conf.trainingFileName = argv[++i];
                conf.testingFileName = argv[++i];
            } else if (strcmp(arg, "method") == 0) {
                conf.thresholdMethod = (strcmp(argv[++i], "KDE") == 0 ? 2 : 1);
            } else if (strcmp(arg, "boundThreshold") == 0) {
                conf.boundThreshold = std::stod(argv[++i]);
            } else if (strcmp(arg, "alpha") == 0) {
                conf.alpha = std::stoi(argv[++i]);
            } else if (strcmp(arg, "probablityThreshold") == 0) {
                conf.probablityThreshold = std::stod(argv[++i]);
            } else if (strcmp(arg, "minK") == 0) {
                conf.minK = std::stoi(argv[++i]);
            } else if (strcmp(arg, "maxK") == 0) {
                conf.maxK = std::stoi(argv[++i]);
            } else if (strcmp(arg, "recallThreshold") == 0) {
                conf.recallThreshold = std::stod(argv[++i]);
            }
        }
    }
    return conf;
}

int main (int argc, char* argv[]) {
    Result res = trainAndClassify(argparse(argc, argv));
    std::cout << "Result: SUCCESS, " << std::setprecision(6) << std::fixed
              << res.time << ", [" << res.earliness
              << ", "  << res.errorRate << "], 0" << std::endl;
}
