# include <string>

// OliverOil Dataset
//const int DIMENSION=570; // length of time series
//const int ROWTRAINING=30;  // size of training data
//const int ROWTESTING=30;  // size of testing data
//const char* trainingFileName="OliveOil/OliveOil_TRAIN";
//const char* testingFileName="OliveOil/OliveOil_TEST";
//const char* trainingIndexFileName="OliveOil/Index_Train";
//const char* testingIndexFileName="OliveOil/Test_Traing";
//const char* DisArrayFileName="OliveOil/DisArray";
//const int NofClasses=4;
//const int Classes[]={1,2,3,4};
//const char* ResultfileName="OliveOil/result.txt";


// ECG dataset
namespace ECG {

const int DIMENSION=96; // length of time series
const int ROWTRAINING=100;  // size of training data
const int ROWTESTING=100;  // size of testing data
const char* trainingFileName="ECG/ECG200_TRAIN";
const char* testingFileName="ECG/ECG200_TEST";
const char* trainingIndexFileName="ECG/Index_Train";
const char* testingIndexFileName="ECG/Test_Traing";
const char* DisArrayFileName="ECG/DisArray";
const int NofClasses=2;
const int Classes[]={1,-1};
const char* ResultfileName="ECG/result.txt";

} // namespace ECG


// Gun Point
//const int DIMENSION=150; // length of time series
//const int ROWTRAINING=50;  // size of training data
//const int ROWTESTING=150;  // size of testing data
//const char* trainingFileName="Gun_Point/Gun_Point_TRAIN";
//const char* testingFileName="Gun_Point/Gun_Point_TEST";
//const char* trainingIndexFileName="Gun_Point/Index_Train";
//const char* testingIndexFileName="Gun_Point/Test_Traing";
//const char* DisArrayFileName="Gun_Point/DisArray";
//const char* ResultfileName="Gun_Point/result.txt";
//const int NofClasses=2;
//const int Classes[]={1,2};

// CBF
//const int DIMENSION=128; // length of time series
//const int ROWTRAINING=30;  // size of training data
//const int ROWTESTING=900;  // size of testing data
//const char* trainingFileName="CBF/CBF_TRAIN";
//const char* testingFileName="CBF/CBF_TEST";
//const char* trainingIndexFileName="CBF/Index_Train";
//const char* testingIndexFileName="CBF/Test_Traing";
//const char* DisArrayFileName="CBF/DisArray";
//const char* ResultfileName="CBF/result.txt";
//const int NofClasses=3;
//const int Classes[]={1,2,3};

// synthetic_control
    //const int DIMENSION=60; // length of time series
    //const int ROWTRAINING=300;  // size of training data
    //const int ROWTESTING=300;  // size of testing data
    //const char* trainingFileName="synthetic_control/synthetic_control_TRAIN";
    //const char* testingFileName="synthetic_control/synthetic_control_TEST";
    //const char* trainingIndexFileName="synthetic_control/Index_Train";
    //const char* testingIndexFileName="synthetic_control/Test_Traing";
    //const char* DisArrayFileName="synthetic_control/DisArray";
    //const char* ResultfileName="synthetic_control/result.txt";
    //const int NofClasses=6;
    //const int Classes[]={1,2,3,4,5,6};

// wafer dataset
//const int DIMENSION=152; // length of time series
//const int ROWTRAINING= 1000;  // size of training data
//const int ROWTESTING= 6174;  // size of testing data
//const char* trainingFileName="wafer/wafer_TRAIN";
//const char* testingFileName="wafer/wafer_TEST";
//const char* trainingIndexFileName="wafer/Index_Train";
//const char* testingIndexFileName="wafer/Test_Traing";
//const char* DisArrayFileName="wafer/DisArray";
//const char* ResultfileName="wafer/result.txt";
//const int NofClasses= 2;
//const int Classes[]={1,-1};

// two patterns
//const int DIMENSION=128; // length of time series
//const int ROWTRAINING= 1000;  // size of training data
//const int ROWTESTING= 4000;  // size of testing data
//const char* trainingFileName="Two_Patterns/Two_Patterns_TRAIN";
//const char* testingFileName="Two_Patterns/Two_Patterns_TEST";
//const char* trainingIndexFileName="Two_Patterns/Index_Train";
//const char* testingIndexFileName="Two_Patterns/Test_Traing";
//const char* DisArrayFileName="Two_Patterns/DisArray";
//const char* ResultfileName="Two_Patterns/result.txt";
//const int NofClasses= 4;
//const int Classes[]={1,2,3,4};

// ECG Large
//const int DIMENSION=85; // length of time series
//const int ROWTRAINING= 810;  // size of training data
//const int ROWTESTING= 1216;  // size of testing data
//const char* trainingFileName="ECGLarge/ECGLargeTrain";
//const char* testingFileName="ECGLarge/ECGLargeTest";
//const char* trainingIndexFileName="ECGLarge/Index_Train";
//const char* testingIndexFileName="ECGLarge/Test_Traing";
//const char* DisArrayFileName="ECGLarge/DisArray";
//const char* ResultfileName="ECGLarge/result.txt";
//const int NofClasses= 1;
//const int Classes[]={1,-1};

// synthetic_control (UCI dataset without normalization)
    //const int DIMENSION=60; // length of time series
    //const int ROWTRAINING=300;  // size of training data
    //const int ROWTESTING=300;  // size of testing data
    //const char* trainingFileName="SynUCI/UCISynControlTraining.txt";
    //const char* testingFileName="SynUCI/UCISynControlTesting.txt";
    //const char* trainingIndexFileName="SynUCI/Index_Train";
    //const char* testingIndexFileName="SynUCI/Test_Traing";
    //const char* DisArrayFileName="SynUCI/DisArray";
    //const char* ResultfileName="SynUCI/result.txt";
    //const int NofClasses=6;
    //const int Classes[]={1,2,3,4,5,6};

// wafer 2000
//const int DIMENSION=152; // length of time series
//const int ROWTRAINING= 2000*0.9;  // size of training data
//const int ROWTESTING= 6174;  // size of testing data
//const char* trainingFileName="wafer/wafer_TRAIN_2000";
//const char* testingFileName="wafer/wafer_TEST";
//const char* trainingIndexFileName="wafer/Index_Train";
//const char* testingIndexFileName="wafer/Test_Traing";
//const char* DisArrayFileName="wafer/DisArray";
//const char* ResultfileName="wafer/result2000.txt";
//const int NofClasses= 2;
//const int Classes[]={1,-1};


// yoga 2000
//const int DIMENSION=426; // length of time series
//const int ROWTRAINING= 300;  // size of training data
//const int ROWTESTING= 3000;  // size of testing data
//const char* trainingFileName="yoga/yoga_TRAIN";
//const char* testingFileName="yoga/yoga_TEST";
//const char* trainingIndexFileName="yoga/Index_Train";
//const char* testingIndexFileName="yoga/Test_Traing";
//const char* DisArrayFileName="yoga/DisArray";
//const char* ResultfileName="yoga/result.txt";
//const int NofClasses= 2;
//const int Classes[]={1,2};


// trace
//const int DIMENSION=275; // length of time series
//const int ROWTRAINING= 100;  // size of training data
//const int ROWTESTING= 100;  // size of testing data
//const char* trainingFileName="Trace/Trace_TRAIN";
//const char* testingFileName="Trace/Trace_TEST";
//const char* trainingIndexFileName="Trace/Index_Train";
//const char* testingIndexFileName="Trace/Test_Traing";
//const char* DisArrayFileName="Trace/DisArray";
//const char* ResultfileName="Trace/result.txt";
//const int NofClasses= 4;
//const int Classes[]={1,2,3,4};

