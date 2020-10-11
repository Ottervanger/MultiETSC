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

// Light-7  dataset
//const int DIMENSION=319; // length of time series
//const int ROWTRAINING=70;  // size of training data
//const int ROWTESTING=73;  // size of testing data
//const char* trainingFileName="Lighting7/Lighting7_TRAIN1";
//const char* testingFileName="Lighting7/Lighting7_TEST";
//const char* resultFileName="Lighting7/Result";
//const char* path="Lighting7/";
//const int NofClasses=7;
//const int Classes[]={0,1,2,3,4,5,6};
//const int ClassIndexes[]={0,     8,    16,    24,    36,    41,    60}; // the data is sorted by class
//const int ClassNumber[]={8,     8,     8,    12,     5,    19,    10};

// beef  dataset
//const int DIMENSION=470; // length of time series
//const int ROWTRAINING=30;  // size of training data
//const int ROWTESTING=30;  // size of testing data
//const char* trainingFileName="Beef/Beef_TRAIN";
//const char* testingFileName="Beef/Beef_TEST";
//const char* resultFileName="Beef/Result";
//const char* path="Beef/";
//const int NofClasses=5;
//const int Classes[]={1,2,3,4,5};
//const int ClassIndexes[]={0,     6 ,   12,    18,    24}; // the data is sorted by class
//const int ClassNumber[]={6,     6,     6,     6,     6};



// ECG dataset
const int DIMENSION=96; // length of time series
const int ROWTRAINING=100;  // size of training data
const int ROWTESTING=100;  // size of testing data
const char* trainingFileName="/home/gilles//UCR/ECG200/ECG200_TRAIN.tsv";
const char* testingFileName="/home/gilles//UCR/ECG200/ECG200_TEST.tsv";
const char* resultFileName="ECG200/Result";
const char* path="ECG200/";
const int NofClasses=2;
const int Classes[]={-1,1};
const int ClassIndexes[]={0,31}; // the data is sorted by class
const int ClassNumber[]={31,69};




// Gun Point
// const int DIMENSION=150; // length of time series
// const int ROWTRAINING=50;  // size of training data
// const int ROWTESTING=150;  // size of testing data
// const char* trainingFileName="Gun_Point/Gun_Point_TRAIN1";
// //const char* trainingFileName="Gun_Point/Gun_PointShort";
// const char* testingFileName="Gun_Point/Gun_Point_TEST";
// const char* resultFileName="Gun_Point/Result";
// const char* path="Gun_Point/";
// const int NofClasses= 2;
// const int Classes[]={1,2};
// const int ClassIndexes[]={0,24}; // the data is sorted by class
// const int ClassNumber[]={24,26};

// CBF
//const int DIMENSION=128; // length of time series
//const int ROWTRAINING=30;  // size of training data
//const int ROWTESTING=900;  // size of testing data
//const char* trainingFileName="CBF/CBF_TRAIN1";
//const char* testingFileName="CBF/CBF_TEST";
//const char* resultFileName="CBF/Result";
//const char* path="CBF/";
//const int NofClasses= 3;
//const int Classes[]={1,2,3};

//const int ClassIndexes[]={ 0 ,   10,    22}; // the data is sorted by class
//const int ClassNumber[]={10   , 12   ,  8};

// synthetic_control
  //  const int DIMENSION=60; // length of time series
  //  const int ROWTRAINING=300;  // size of training data
  //  const int ROWTESTING=300;  // size of testing data
  //  const char* trainingFileName="synthetic_control/synthetic_control_TRAIN1";
  //  const char* testingFileName="synthetic_control/synthetic_control_TEST";
  // const char* path="synthetic_control/";
  // const char* resultFileName="synthetic_control/Result";
  //  const int NofClasses=6;
  //  const int Classes[]={1,2,3,4,5,6};
  //const int ClassIndexes[]={ 0 ,   50   ,100 ,  150,   200 ,  250}; // the data is sorted by class
  // const int ClassNumber[]={50 ,   50 ,   50   , 50  ,  50  ,  50};


 //wafer dataset
//const int DIMENSION=152; // length of time series
//const int ROWTRAINING= 1000;  // size of training data
//const int ROWTESTING= 6174;  // size of testing data
//const char* trainingFileName="wafer/wafer_TRAIN1";
//const char* testingFileName="wafer/wafer_TEST";
//const char* resultFileName="wafer/Result";
//const char* path="wafer/";
//const int NofClasses=2;
//const int Classes[]={-1,1};
//const int ClassIndexes[]={0,    97}; // the data is sorted by class
//const int ClassNumber[]={97,   903};

// two patterns
//const int DIMENSION=128; // length of time series
//const int ROWTRAINING= 1000;  // size of training data
//const int ROWTESTING= 4000;  // size of testing data
//const char* trainingFileName="Two_Patterns/Two_Patterns_TRAIN1";
//const char* testingFileName="Two_Patterns/Two_Patterns_TEST";
//const char* path="Two_Patterns/";
//const int NofClasses=4;
//const int Classes[]={1,2,3,4};
//const int ClassIndexes[]={0,   271   ,508 ,  758}; // the data is sorted by class
//const int ClassNumber[]={271 ,  237 ,  250  , 242};

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

// my synthetic data set
//const int DIMENSION=100; // length of time series
//const int ROWTRAINING= 100;  // size of training data
//const int ROWTESTING= 100;  // size of testing data
//const char* trainingFileName="MySyn/MySynTraining.txt";
//const char* testingFileName="MySyn/MySynTesting.txt";
//const char* path="MySyn/";
//const int NofClasses= 2;
//const int Classes[]={-1,1};
//const int ClassIndexes[]={0,50}; // the data is sorted by class
//const int ClassNumber[]={50,50};

// my synthetic data set
//const int DIMENSION=100; // length of time series
//const int ROWTRAINING= 100;  // size of training data
//const int ROWTESTING= 100;  // size of testing data
//const char* trainingFileName="MySyn1/MySynTraining.txt";
////const char* testingFileName="MySyn1/MySynTraining.txt";
//const char* testingFileName="MySyn1/MySynTesting.txt";
//const char* path="MySyn1/";
//const int NofClasses= 2;
//const int Classes[]={-1,1};
//const int ClassIndexes[]={0,50}; // the data is sorted by class
//const int ClassNumber[]={50,50};




