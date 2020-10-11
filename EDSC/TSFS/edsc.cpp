// this version is for the non-redundant top-K feature selection
// in this version, ranked by earliness sctore, and classifying all the classes together,
// the default classifier has  been added as the majority class
// output into the text file

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <cstring>
#include "math.h"
#include <algorithm>
#include <vector>
#include <bitset>
#include <time.h>
#include <climits>
#include "DataSetInformation.h"
#include "util.h"

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
    std::bitset<ROWTRAINING> bmap;  // std::bitset does not change
};

struct BitArray {
    std::vector<bool> a;
    BitArray(std::size_t size) : a(size, false) { }
    bool operator[] (std::size_t pos) { return a[pos]; }
    BitArray operator& (BitArray ba) {
        BitArray r(std::max(ba.a.size(), a.size()));
        for (int i = 0; i < a.size() && i < ba.a.size(); ++i) {
            r.a[i] = a[i] & ba.a[i];
        }
        return r;
    }
    void set() { std::fill(a.begin(), a.end(), true); }
    void set(std::size_t pos) { a[pos] = true; }
    void reset() { std::fill(a.begin(), a.end(), false); }
    void reset(std::size_t pos) { a[pos] = false; }
    void flip() { for (int i = 0; i < a.size(); ++i) { a[i] = !a[i]; } }
    void flip(std::size_t pos) { a[pos] = !a[pos]; }
    int count() { int c = 0; for (int i = 0; i < a.size(); ++i) { c += a[i]; } return c;}
};

// global variable
std::vector<std::vector<double> > training;         // training data set
std::vector<int> labelTraining;                     // training data class labels
std::vector<std::vector<double> > testing;          // testing data set
std::vector<int> labelTesting;                      // testing data class labels
std::bitset<ROWTRAINING> totalBmap;                      // the union bit map of a certain length of a certain class
std::bitset<ROWTRAINING> allLengthBmap;                  // the union bit map of all length of a certain class
std::vector<Feature *> finalFset;
std::vector<Feature *> AllClassesFset;


// functions in the same file
double * getSegment(int segIndex, int InstanceIndex);
void DisArray(int classIndex, int instanceIndex, double ** DisArray, int ** EndP, int rows, int columns, int MinK, int MaximalK);
void SaveDisArray(double ** DisArray, int rows, int columns, const char * filename, int ** EndP, const char * filename2);
double getMean( double * arr, int len);  // segmentIndex starting from 1
double getSTD(double * arr, int len, double mean);
Feature * ThresholdLearningAll(int DisArrayIndex, int instanceIndex, int startPosition,  double m, int k, int targetClass, double ** DisA, double RecallThreshold, int ** EndP, int alpha);
double * getFeatureSegment(Feature *f);
void PrintFeature(Feature * f);
void PrintFeature(Feature * f, std::ostream& ResultFile);
void PrintTotalBitMap();
void classification(std::vector<Feature *> &Fs, int classIndex, int k);
void classificationAllLength(std::vector<Feature *> &Fs, int classIndex, std::ofstream& resultFile);
void PrintFeatureMoreInfo(Feature * f);
void ReduceAllLength(std::vector<Feature *> &finalFset, std::vector<Feature *> &fSet );
double OneKDE(double * arr, int len, double q, double h, double constant);
//Feature * ThresholdLearningKDE(int index, int k, int targetClass, double ** DisA, double PrecisionThreshold, double RecallThreshold, double ProbalityThreshold);
double ComputeFScore(double recall, double precision);
Feature * ThresholdLearningKDE(int DisArrayIndex, int Instanceindex, int startPosition, int k, int targetClass, double ** DisA, double RecallThreshold, double ProbalityThreshold, int ** EndP, int alpha);
void classificationAllClasses(std::vector<Feature *> &Fs) ;
void classificationAllClassesWithDefault(std::vector<Feature *> &Fs);

void computeWholeSubstringArray(int iLengthOfArray1,
                                                                int iLengthOfArray2,
                                                                double * pMatrixOriginal,
                                                                double * pArraySubstring, double * pArraySubstringEndingPosition, int iLengthOfSubstringArray, int MaximalLength);

void DisArray2(int classIndex, int instanceIndex, double ** DisArray, int ** EndP, int rows, int columns, int MinK, int MaximalK) ;

void createOriginalMatrix(double * array1, int iLengthOfArray1,
                                                    double * array2, int iLengthOfArray2,
                                                    double * &pMatrix);

int main() {
    clock_t tStart = clock();
    // load data
    util::readUCRData(trainingFileName, training, labelTraining);
    util::readUCRData(trainingFileName, testing, labelTesting);

    // compute the best match distance array for the target class.
    // create the space to store the disArray of length k for a certain class

    int option = 1; // option =1 using thresholdAll, option=2, using the KDE cut
    int DisArrayOption = 2;  // =1 , naive version =2 fast version
    int MaximalK = DIMENSION / 2; // maximal length
    int MinK = 5;
    double boundThrehold = 3;  //define the parameter of the Chebyshev's inequality
    double recallThreshold = 0;
    double probablityThreshold = 0.95;
    int alpha = 3;

    // create the output file for the result
    std::ofstream resultFile("out/log", std::ios::out);
    if (option == 1) {
        std::cout << "Chebyshev bound" << std::endl;
        std::cout << "boundThrehold = " << boundThrehold << std::endl;
        std::cout << "alpha = " << alpha << std::endl;
    } else {
        std::cout << "KDE bound" << std::endl;
        std::cout << "probablityThreshold = " << probablityThreshold << std::endl;
    }

    std::cout << "MinLength=" << MinK << ", MaxLength=" << MaximalK << "(" << (double)MaximalK / DIMENSION << "of " << DIMENSION << ")" << std::endl;
    if (DisArrayOption == 1)
        std::cout << "slow version" << std::endl;
    if (DisArrayOption == 2)
        std::cout << "fast version" << std::endl;


    for (int cl = 0; cl < NofClasses; cl++) {
        std::cout << "class: " << cl << std::endl;
        int classIndex = cl; // pick the class
        std::vector<Feature *> reducedFset; // restore the set of the first round set cover
        int k = 1;
        clock_t startFeature, endFeature;

        // initialize the DisA and EndP
        for (int tIndex = ClassIndexes[cl]; tIndex < ClassIndexes[cl] + ClassNumber[cl]; tIndex++) {
            // start of outer for

            int NumberOfSegments = 0;
            for (int i = DIMENSION - MinK + 1; i >= DIMENSION - MaximalK + 1; i--) {
                NumberOfSegments = NumberOfSegments + i;
            }

            double ** DisA = new double *[NumberOfSegments];
            int ** EndP = new int *[NumberOfSegments];
            for (int i = 0; i < NumberOfSegments; i++) {
                DisA[i] = new double[training.size()];
                EndP[i] = new int[training.size()]; // ending position array
            }

            if (DisArrayOption == 1)
                DisArray(cl, tIndex,  DisA, EndP, NumberOfSegments, training.size(), MinK, MaximalK);
            else if (DisArrayOption == 2)
                DisArray2(cl, tIndex,  DisA, EndP, NumberOfSegments, training.size(), MinK, MaximalK);
            else
                DisArray(cl, tIndex,  DisA, EndP, NumberOfSegments, training.size(), MinK, MaximalK);

            // compute
            startFeature = clock();
            totalBmap.reset();   // reset the Bmap for each length's set cover
            std::vector<Feature *> fSet;
            //int offset=ClassIndexes[classIndex]*(DIMENSION-k+1)+1;
            int DisArrayIndex = 0;
            for (int currentL = MinK; currentL <= MaximalK; currentL++ ) { // for each length

                for (int startPosition = 0; startPosition <= DIMENSION - currentL; startPosition++) {

                    Feature * temp;
                    if (option == 1) {
                        temp = ThresholdLearningAll(DisArrayIndex, tIndex, startPosition, boundThrehold, currentL, classIndex, DisA, recallThreshold, EndP, alpha);
                        // temp=ThresholdLearningAll(offset+s, boundThrehold, k, classIndex, DisA, recallThreshold);
                    } else if (option == 2) {
                        temp = ThresholdLearningKDE(DisArrayIndex, tIndex, startPosition, currentL, classIndex, DisA, recallThreshold, probablityThreshold, EndP, alpha);
                    } else {
                        std::cerr << "Error: Invalid option." << std::endl;
                        exit(0);
                    }

                    if (temp != NULL) {
                        fSet.push_back(temp);
                    }
                    DisArrayIndex++;
                }// end of position for
            } // end of length for

            // compute the non-redundent feature set by non-redundant top-K
            while (totalBmap.count() > 0) {
                double max = -1;
                double maxLength = -1;
                int index = -1;
                for (unsigned int i = 0; i < fSet.size(); i++) {
                    double temp = fSet.at(i)->earlyFScore;
                    if (temp > max ) {
                        max = temp;
                        maxLength = fSet.at(i)->length;
                        index = i;

                    } else if  (temp == max && fSet.at(i)->length > maxLength) {
                        max = temp;
                        maxLength = fSet.at(i)->length;
                        index = i;
                    }
                }
                // move this feature to reducedFset
                if (index >= 0) {
                    Feature * currentFeature = fSet.at(index);
                    fSet.erase(fSet.begin() + index);
                    // check if increase the coverage

                    // check current with the totalBmap if there is 1 in current and 1 in totalBmap, new coverage
                    std::bitset<ROWTRAINING> newCoverage = currentFeature->bmap & totalBmap;

                    if (newCoverage.count() > 0) {
                        reducedFset.push_back(currentFeature);
                    }

                    // using bit operation
                    for (unsigned int j = 0; j < (currentFeature->bmap).size(); j++) { // for the current set, update the other set
                        if ((currentFeature->bmap)[j] == 1) {
                            totalBmap.reset(j);  // update the total covered set
                        }
                    }
                }
            }// end while  , end of the feature selection in each length
            std::cout << "reducedFset of instance: " << tIndex << " = " << reducedFset.size() << std::endl;

            // relase the memory
            for (int i = 0; i < NumberOfSegments; i++) {
                delete [] DisA[i];
            }
            delete [] DisA;

            for (int i = 0; i < NumberOfSegments; i++) {
                delete [] EndP[i];
            }
            delete [] EndP;

            for (unsigned int i = 0; i < fSet.size(); i++) {
                delete fSet.at(i);
            }
            endFeature = clock();
        } // end of outer for, end of feature for each isntance

        // second round of set cover
        std::cout << "Total Coverage Rate: " << (double)allLengthBmap.count() / ClassNumber[classIndex] << std::endl;
        ReduceAllLength(finalFset, reducedFset);
        std::cout << "Final set size: " << finalFset.size() << std::endl;

        for (unsigned int s = 0; s < finalFset.size(); s++) {
            std::cout << "\n" << s;
            PrintFeature(finalFset.at(s));
            PrintFeature(finalFset.at(s), resultFile);
        }

        // UPDATE THE ALL CLASS DATA SET
        while (!finalFset.empty()) {
            Feature * tempFeature = finalFset.at(0);
            finalFset.erase(finalFset.begin() + 0);
            AllClassesFset.push_back(tempFeature);
        }
    }// end of for for each classes

    for (unsigned int s = 0; s < AllClassesFset.size(); s++) {
        std::cout << s << std::endl;
        PrintFeature(AllClassesFset.at(s));
    }

    //classification
    classificationAllClasses(AllClassesFset);
    clock_t classificationStart = clock();
    classificationAllClassesWithDefault(AllClassesFset);
    clock_t classificationEnd = clock();

    std::cout << "Training time: " << (double)(clock() - tStart) / CLOCKS_PER_SEC << std::endl;
    std::cout << "av. classifciation time: " << (double)(classificationEnd - classificationStart) / ROWTESTING / CLOCKS_PER_SEC << " seconds" << std::endl;
}

// second round of feature selection by top-K non-redundant
void ReduceAllLength(std::vector<Feature *> & finalFset, std::vector<Feature *> &fSet) {
    while (allLengthBmap.count() > 0) {
        double max = -1;
        double maxLength = -1;
        int index = -1;
        for (unsigned int i = 0; i < fSet.size(); i++) {
            double temp = fSet.at(i)->earlyFScore;
            if (temp > max ) {
                max = temp;
                maxLength = fSet.at(i)->length;
                index = i;
            } else if  (temp == max && fSet.at(i)->length > maxLength) {
                max = temp;
                maxLength = fSet.at(i)->length;
                index = i;
            }
        }
        // move this feature to reducedFset and reset others
        if (index >= 0) {
            Feature * currentFeature = fSet.at(index);
            fSet.erase(fSet.begin() + index);
            // need to remove the redundancy

            // check current with the totalBmap if there is 1 in current and 1 in totalBmap, new coverage
            std::bitset<ROWTRAINING> newCoverage = currentFeature->bmap & allLengthBmap;

            if (newCoverage.count() > 0) {
                finalFset.push_back(currentFeature);
            }
            for (unsigned int j = 0; j < (currentFeature->bmap).size(); j++) { // for the current set, update the other set
                if ((currentFeature->bmap)[j] == 1) {
                    allLengthBmap.reset(j);  // update the total covered set
                }
            }
        }
    }// end while
    std::cout << "finalFset:" << finalFset.size() << std::endl;

    for (unsigned int i = 0; i < fSet.size(); i++) {
        delete fSet.at(i);
    }
}


void classificationAllClassesWithDefault(std::vector<Feature *> &Fs) {
    int NoClassified = 0;
    int CorrectlyClassified = 0;
    int sumLength = 0;
    //find the default label as the the most frequent class

    int defaultlabel = -1;
    int mostFrequent = -1;
    for (int ci = 0; ci < NofClasses; ci++) {
        if (ClassNumber[ci] > mostFrequent) {
            mostFrequent = ClassNumber[ci];
            defaultlabel = Classes[ci];
        }
    }

    for (int i = 0; i < ROWTESTING; i++) {
        bool matched = 0;
        for (int j = 0; j < DIMENSION; j++) { // j is the current ending position of the stream
            for (unsigned int f = 0; f < Fs.size(); f++) {
                int tempLength = Fs.at(f)->length;
                int startingPosition = j - tempLength + 1;
                if (startingPosition >= 0) {
                    double * currentseg = new double[tempLength];
                    for (int ss = 0; ss < tempLength; ss++) {
                        currentseg[ss] = testing[i][ss + startingPosition];
                    }
                    double * tempFeatureSeg = getFeatureSegment(Fs.at(f));
                    double tempDis = util::euclidean(tempFeatureSeg, currentseg, tempLength);
                    delete [] tempFeatureSeg;
                    delete [] currentseg;
                    if (tempDis <= (Fs.at(f)->threshold)) {
                        matched = 1;
                        sumLength = sumLength + j + 1;
                        NoClassified++;
                        if (Fs.at(f)->label == labelTesting[i])
                            CorrectlyClassified++;
                        break;
                        // break to stop checking more features
                    }
                }
            }
            if (matched == 1) // break the segment loop, finish the current testing example
                break;
        } // end of for , finish classify the current example by the features
        if (matched == 0) { // classified by the default classifier
            if (labelTesting[i] == defaultlabel)
                CorrectlyClassified++;
            sumLength = sumLength + DIMENSION;
        }
    }
    std::cout << "accuracy:  " << (double)CorrectlyClassified / ROWTESTING << std::endl;
    std::cout << "earliness: " << (double)sumLength / ROWTESTING << std::endl;
}
void classificationAllClasses(std::vector<Feature *> &Fs) {
    int NoClassified = 0;
    int CorrectlyClassified = 0;
    int sumLength = 0;

    int n = ROWTESTING;
    int len = DIMENSION;
    std::vector<int> predictedLabel(n);
    for (int i = 0; i < n; i++) {
        bool matched = 0;
        for (int j = 0; j < len; j++) { // j is the current ending position of the stream
            for (unsigned int f = 0; f < Fs.size(); f++) {
                int tempLength = Fs.at(f)->length;
                int startingPosition = j - tempLength + 1;
                if (startingPosition >= 0) {
                    double * currentseg = new double[tempLength];
                    for (int ss = 0; ss < tempLength; ss++)
                    {currentseg[ss] = testing[i][ss + startingPosition];}
                    double * tempFeatureSeg = getFeatureSegment(Fs.at(f));
                    double tempDis = util::euclidean(tempFeatureSeg, currentseg, tempLength);
                    delete [] tempFeatureSeg;
                    delete [] currentseg;
                    if (tempDis <= (Fs.at(f)->threshold)) {
                        predictedLabel[i] = (int)Fs.at(f)->label;
                        matched = 1;
                        sumLength = sumLength + j + 1;
                        NoClassified++;
                        if (Fs.at(f)->label == labelTesting[i]) {
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
    }
    std::cout << "coverage: " << (double)NoClassified / n << std::endl;
    if (NoClassified == 0)
        std::cout << " accuracy: ---" << std::endl;
    else {
        std::cout << " accuracy:  " << (double)CorrectlyClassified / NoClassified << std::endl;
        std::cout << " earliness: " << (double)sumLength / NoClassified << std::endl;
    }

    // compute the precision and recall of each class;
    for (int c = 0; c < NofClasses; c++) {
        int tp = 0;
        int classifiedintoC = 0;
        int hasLabelC = 0;
        for (int i = 0; i < n; i++) {
            if (predictedLabel[i] == Classes[c] && (int)labelTesting[i] == Classes[c])
                tp++; 
            if (predictedLabel[i] == Classes[c])
                classifiedintoC++;
            if ((int)labelTesting[i] == Classes[c] )
                hasLabelC++;
        }
        std::cout << std::endl
            << "Class: " << Classes[c] << std::endl
            << " recall:    " << (double)tp / hasLabelC << std::endl
            << " precision: " << (double)tp / classifiedintoC << std::endl;
    }
}

// print the information of the feature
void PrintFeature(Feature * f) {
    PrintFeature(f, std::cout);
}

// output to the result file
void PrintFeature(Feature * f, std::ostream& of) {
    if (f != NULL) {
        of << "feature = [ ";

        for (int i = 0; i < f->length; i++) {
            of << training[f->instanceIndex][f->startPosition + i] << " ";
        }
        of << " ]" << std::endl
           << "instance Index = " << f->instanceIndex << std::endl
           << "startPosition = " << f->startPosition << std::endl
           << "length = " << f->length << std::endl
           << "threshod = " << f->threshold << std::endl
           << "recall = " << f->recall << std::endl
           << "precision = " << f->precision << std::endl
           << "fscore = " << f->fscore << std::endl
           << "earlyfscore = " << f->earlyFScore << std::endl;
    }
}

double ComputeFScore(double recall, double precision) {
    return 2 * recall * precision / (recall + precision);
}


// learning threshold based on the one tail Chebyshev's inequality
Feature * ThresholdLearningAll(int DisArrayIndex, int instanceIndex, int startPosition,  double m, int k, int targetClass, double ** DisA, double RecallThreshold, int ** EndP, int alpha) { // this index starting from 1, m is the parameter in the bound, k is the length of feature
    int classofset = ClassIndexes[targetClass] * (DIMENSION - k + 1) + 1;
    Feature * currentf = new Feature();
    currentf->instanceIndex = instanceIndex;
    currentf->startPosition = startPosition;
    currentf->length = k;
    currentf->label = Classes[targetClass];

    // get the non-target class part in the distance array
    int nonTargetTotal = 0;
    for (int c = 0; c < NofClasses; c++) {
        if (c != targetClass) {
            nonTargetTotal = nonTargetTotal + ClassNumber[c];
        }
    }

    double * nonTargetDis = new double[nonTargetTotal];

    int i = 0;
    for (int c = 0; c < NofClasses; c++) {
        if (c != targetClass) {
            int offset = ClassIndexes[c];
            for (int e = 0; e < ClassNumber[c]; e++) {
                nonTargetDis[i] = DisA[DisArrayIndex][offset + e];
                i++;
            }
        }
    }
    // compute the mean, standard deviation and the threshold
    double mu = getMean(nonTargetDis, nonTargetTotal);
    double sd = getSTD(nonTargetDis, nonTargetTotal, mu);

    currentf->threshold = mu - m * sd;

    delete [] nonTargetDis; // release the memory

    // compute recall, precision and bitmap
    if (currentf->threshold > 0) {
        int targetCount = 0;
        double weightedRecall = 0;
        int totalCount = 0;
        for (int i = 0; i < ROWTRAINING; i++) {
            double temp = DisA[DisArrayIndex][i];
            if (temp <= currentf->threshold) {
                totalCount++;
                if (labelTraining[i] == Classes[targetClass]) {
                    targetCount++;
                    (currentf->bmap).set(i);  // set the bmap
                    weightedRecall = weightedRecall + pow(((double)1 / EndP[DisArrayIndex][i]), (double)1 / alpha);
                }
            }
        }
        currentf->recall = (double)targetCount / ClassNumber[targetClass]; // it is the absolute recall
        currentf->precision = (double)targetCount / totalCount;
        currentf->fscore = ComputeFScore(currentf->recall, currentf->precision);
        currentf->earlyFScore = ComputeFScore(weightedRecall, currentf->precision);

        if (currentf->recall >= RecallThreshold ) {
            for (unsigned int i = 0; i < (currentf->bmap).size(); i++) {
                if ((currentf->bmap)[i] == 1) {
                    totalBmap.set(i);  //  set the total Bmap for each length's set cover
                    allLengthBmap.set(i); // set the all length Bmap for the second round of set cover
                }
            }
            return currentf;
        } else {
            delete currentf;
            return NULL;
        }
    } else {
        delete currentf;
        return NULL;
    }

}

double OneKDE(double * arr, int len, double q, double h, double constant) {
    double temp = 0;
    for (int i = 0; i < len; i++) {
        temp = temp + exp((arr[i] - q) * (arr[i] - q) * (-1 / (2 * h * h)));
    }
    temp = temp * constant;
    return temp;
}

//learning the threshold by KDE classification.
Feature * ThresholdLearningKDE(int DisArrayIndex, int Instanceindex, int startPosition, int k, int targetClass, double ** DisA, double RecallThreshold, double ProbalityThreshold, int ** EndP, int alpha) {
    int classofset = ClassIndexes[targetClass] * (DIMENSION - k + 1) + 1;
    Feature * currentf = new Feature();
    currentf->instanceIndex = Instanceindex;
    currentf->startPosition = startPosition;
    currentf->length = k;

    currentf->label = Classes[targetClass];

    // get the target class part and non-target part in the distance array
    int nonTargetTotal = 0;
    for (int c = 0; c < NofClasses; c++) {
        if (c != targetClass) {
            nonTargetTotal = nonTargetTotal + ClassNumber[c];
        }
    }
    int TargetTotal = ClassNumber[targetClass];

    double * nonTargetDis = new double[nonTargetTotal];
    double * TargetDis = new double[TargetTotal];
    double * CurrentDis = new double[ROWTRAINING];

    int nonTargeti = 0;
    int Targeti = 0;
    int totali = 0;
    for (int c = 0; c < NofClasses; c++) {
        int offset = ClassIndexes[c];
        if (c != targetClass) {
            for (int e = 0; e < ClassNumber[c]; e++) {
                nonTargetDis[nonTargeti] = DisA[DisArrayIndex][offset + e];
                nonTargeti++;
                CurrentDis[totali] = DisA[DisArrayIndex][offset + e];
                totali++;
            }
        } else {
            for (int e = 0; e < ClassNumber[c]; e++) {
                TargetDis[Targeti] = DisA[DisArrayIndex][offset + e];
                Targeti++;
                CurrentDis[totali] = DisA[DisArrayIndex][offset + e];
                totali++;
            }
        }
    }
    // compute the mean, standard deviation and the threshold, and optimal h
    //  for the nonTarget Classes
    double muNonTarget = getMean(nonTargetDis, nonTargetTotal);

    double sdNonTarget = getSTD(nonTargetDis, nonTargetTotal, muNonTarget);

    double hNonTarget = 1.06 * sdNonTarget / pow (nonTargetTotal, 0.2);

    double constantNT = 1 / (sqrt(2 * 3.14159265) * nonTargetTotal * hNonTarget);
    //  for the TargetClasses
    double muTarget = getMean(TargetDis, TargetTotal);

    double sdTarget = getSTD(TargetDis, TargetTotal, muTarget);

    double hTarget = 1.06 * sdTarget / pow (TargetTotal, 0.2);
    double constantT = 1 / (sqrt(2 * 3.14159265) * TargetTotal * hTarget);

    // sort the totalDis
    util::quicksort( CurrentDis, 0, ROWTRAINING - 1);
    //  compute the Probablity<0;
    double NegativeTestPoint = -CurrentDis[ROWTRAINING - 1] / (ROWTRAINING - 1);
    double densityNonTarget = OneKDE(nonTargetDis, nonTargetTotal, NegativeTestPoint, hNonTarget,  constantNT);
    double densityTarget = OneKDE(TargetDis, TargetTotal, NegativeTestPoint, hTarget,  constantT);
    double tempTarget = ((double)ClassNumber[targetClass] / ROWTRAINING) * densityTarget;
    double tempNonTarget = (1 - ((double)ClassNumber[targetClass] / ROWTRAINING)) * densityNonTarget;
    double ProTarget = tempTarget / ( tempTarget + tempNonTarget);

    if (ProTarget > ProbalityThreshold) {
        // compute the breaking Index
        int breakIndex = 0;
        int i = 0;
        for (i = 0; i < ROWTRAINING; i++) {
            densityNonTarget = OneKDE(nonTargetDis, nonTargetTotal, CurrentDis[i], hNonTarget,  constantNT);
            densityTarget = OneKDE(TargetDis, TargetTotal, CurrentDis[i], hTarget,  constantT);
            tempTarget = ((double)ClassNumber[targetClass] / ROWTRAINING) * densityTarget;
            tempNonTarget = (1 - ((double)ClassNumber[targetClass] / ROWTRAINING)) * densityNonTarget;
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
                densityNonTarget = OneKDE(nonTargetDis, nonTargetTotal, value, hNonTarget,  constantNT);
                densityTarget = OneKDE(TargetDis, TargetTotal, value, hTarget,  constantT);
                tempTarget = ((double)ClassNumber[targetClass] / ROWTRAINING) * densityTarget;
                tempNonTarget = (1 - ((double)ClassNumber[targetClass] / ROWTRAINING)) * densityNonTarget;
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

    delete  nonTargetDis;
    delete TargetDis;
    delete CurrentDis;

    if (currentf->threshold > 0) {
        int targetCount = 0;
        double weightedRecall = 0;
        int totalCount = 0;
        for (int i = 0; i < ROWTRAINING; i++) {
            double temp = DisA[DisArrayIndex][i];
            if (temp <= currentf->threshold) {
                totalCount++;
                if (labelTraining[i] == Classes[targetClass]) {
                    targetCount++;
                    (currentf->bmap).set(i);  // set the bmap
                    weightedRecall = weightedRecall + pow(((double)1 / EndP[DisArrayIndex][i]), (double)1 / alpha);

                }
            }
        }
        currentf->recall = (double)targetCount / ClassNumber[targetClass]; // it is the absolute recall
        currentf->precision = (double)targetCount / totalCount;
        currentf->fscore = ComputeFScore(currentf->recall, currentf->precision);
        currentf->earlyFScore = ComputeFScore(weightedRecall, currentf->precision);

        if (currentf->recall >= RecallThreshold) {
            for (int i = 0; i < (currentf->bmap).size(); i++) {
                if ((currentf->bmap)[i] == 1) {
                    totalBmap.set(i);  //  set the total Bmap for each length's set cover
                    allLengthBmap.set(i); // set the all length Bmap for the second round of set cover
                }
            }
            return currentf;
        } else {
            delete currentf;
            return NULL;
        }
    } else {
        delete currentf;
        return NULL;
    }
}

// compute the mean
double getMean( double * arr, int len) { // segmentIndex starting from 1
    double sum = 0;
    for (int i = 0; i < len; i++) {
        sum = sum + arr[i];
    }
    return sum / len;
}

// compute the standard deviation given the mean
double getSTD(double * arr, int len, double mean) {
    double sum = 0;
    for (int i = 0; i < len; i++) {
        sum = sum + (arr[i] - mean) * (arr[i] - mean);
    }
    return sqrt(sum / (len));
}

// compute the best match distance array of the selected class and the ending position
//  the instanceIndex is the absolute instanceIndex
void DisArray(int classIndex, int instanceIndex, double ** DisArray, int ** EndP, int rows, int columns, int MinK, int MaximalK) {
    // compute the DisArray
    for (int tl = 0; tl < columns; tl++) { // for each training example
        // listing substring, for each lenght
        int DisArrayIndex = 0;

        for (int currentL = MinK; currentL <= MaximalK; currentL++ ) { // for each length

            for (int startPosition = 0; startPosition <= DIMENSION - currentL; startPosition++) {
                // compute the best match using naive early stopping
                // compute the best match
                int bestMatchStartP = -1;
                double minDis = 10000;
                for (int l = 0; l < DIMENSION - currentL + 1; l++) { // a possible match's starting position
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
void createOriginalMatrix(std::vector<double> array1,
                          std::vector<double> array2,
                          double * &pMatrix) {
    if (pMatrix != NULL)
        return;
    pMatrix = new double[array1.size()*array2.size()];
    for (int j = 0; j < array2.size(); j++) {
        for (int i = 0; i < array1.size(); i++) {
            pMatrix[j * array1.size() + i] = (array1[i] - array2[j]) * (array1[i] - array2[j]);
        }
    }
}

void computeWholeSubstringArray(int iLengthOfArray1,
                                int iLengthOfArray2,
                                double * pMatrixOriginal,
                                double * pArraySubstring, int * pArraySubstringEndingPosition, int iLengthOfSubstringArray, int MaximalLength) {
    if (pArraySubstring == NULL)
        return;

    double* tempMatrix = new double[iLengthOfArray1 * iLengthOfArray2];
    int iIndexofSubstring = 0;
    int i, j, k;
    //clear out the substring score array
    for (i = 0; i < iLengthOfArray1 * iLengthOfArray2; i++) {
        tempMatrix[i] = 0;
    }

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
void DisArray2(int classIndex, int instanceIndex, double ** DisArray, int ** EndP, int rows, int columns, int MinK, int MaximalK) {
    // compute the DisArray

    for (int tl = 0; tl < columns; tl++) { // for each training example
        double * pArraySubstring = NULL; // the distances
        int * pArraySubstringEndingPosition = NULL;  // th ending position
        int iArraySubstringArrayLength = 0;
        int iArraySubstringArrayEndingPositionLength = 0;

        double * pMatrixOriginal = NULL;
        createOriginalMatrix(training[instanceIndex], training[tl], pMatrixOriginal);
        int pLengthOfSubstringArray = 0;
        for (int c = DIMENSION; c >= DIMENSION - MaximalK + 1; c--) {
            pLengthOfSubstringArray = pLengthOfSubstringArray + c;
        }
        pArraySubstring = new double[pLengthOfSubstringArray];
        pArraySubstringEndingPosition = new int[pLengthOfSubstringArray];
        computeWholeSubstringArray(training[instanceIndex].size(), training[tl].size(), pMatrixOriginal, pArraySubstring, pArraySubstringEndingPosition, iArraySubstringArrayLength, MaximalK);

        // how many from 1 to MinK
        int offset = 0;
        for (int c = DIMENSION; c >= DIMENSION - (MinK - 1) + 1; c--)
        {offset = offset + c;}

        // copy into the disarray and the pend array
        for (int i = offset; i < pLengthOfSubstringArray; i++) {
            DisArray[i - offset][tl] = pArraySubstring[i];
            EndP[i - offset][tl] = pArraySubstringEndingPosition[i];
        }

        // release the memory
        delete[] pMatrixOriginal;
        pMatrixOriginal = NULL;

        delete[] pArraySubstring;
        delete[] pArraySubstringEndingPosition;
        pArraySubstringEndingPosition = NULL;
        pArraySubstring = NULL;
        pMatrixOriginal = NULL;
    }// end of each training example for
} // end of function

