#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "math.h"
#include <algorithm>
#include <vector>
#include <bitset>
#include <time.h>
#include <limits>
#include "DataSetInformation.h"
#include "Euclidean.h"
#include "quickSort.h"
using namespace std; 

// structure of a feature
struct Feature
{
  double * f; 
  int segmentIndex;
  int length; 
  int endPosition;
  double label; 
  double threshold; 
  double recall; 
  double uncoveredRecall;
  double precision; 
  bitset<ROWTRAINING> bmap;  // should change the bmap to only the size of the target class.
  bitset<ROWTRAINING> ubmap; // change in the second round of feature reduction

};
// global variable
double training[ROWTRAINING][DIMENSION]; // training data set
double labelTraining[ROWTRAINING]={0}; // training data class labels
double testing [ROWTESTING][DIMENSION]; //  testing data set
double labelTesting[ROWTESTING]={0}; // testing data class labels
double predictedLabel[ROWTESTING]={0}; // predicted label by the classifier
int predictedLength[ROWTESTING]={0};// predicted length by the classifier
bitset<ROWTRAINING> totalBmap; // the union bit map of a certain length of a certain class
bitset<ROWTRAINING> allLengthBmap;  // the union bit map of all length of a certain class
 vector<Feature *> finalFset;


// functions in the same file
void LoadData(const char * fileName, double Data[][DIMENSION], double Labels[], int len  );
void printData(int index);
double * getSegment(int index, int k);
void DisArray(int classIndex, int k, double ** DisArray, int rows, int columns);
void SaveDisArray(double ** DisArray, int rows, int columns, const char * filename);
void loadDisArray(const char * fileName, double ** Data, int rows, int columns);
double getMean( double * arr, int len);  // segmentIndex starting from 1
double getSTD(double * arr, int len, double mean);
Feature * ThresholdLearningAll(int index, double m, int k, int targetClass, double ** DisA, double RecallThreshold);
void PrintFeature(Feature * f);
void PrintTotalBitMap();
void classification(vector<Feature *> &Fs, int classIndex, int k);
void classificationAllLength(vector<Feature *> &Fs, int classIndex);
string IntToString(int intValue);
void PrintFeatureMoreInfo(Feature * f);
void ReduceAllLength(vector<Feature *> &finalFset, vector<Feature *> &fSet );
double OneKDE(double * arr, int len, double q, double h, double constant);
Feature * ThresholdLearningFirst(int index, double m, int k, int targetClass, double ** DisA, double PrecisionThreshold);  
Feature * ThresholdLearningKDE(int index, int k, int targetClass, double ** DisA, double PrecisionThreshold, double RecallThreshold, double ProbalityThreshold);



void main()
{
    cout<<"\n The naive method for time series feature selection";
     // load training data
	LoadData(trainingFileName, training,labelTraining, ROWTRAINING);
	// load the testing data
    LoadData(testingFileName, testing,labelTesting, ROWTESTING);

    // compute the best match distance array for the target class.
    // create the space to store the disArray of length k for a certain class

	int option=1; // option =1 using thresholdAll, option=2, using thresholdFirst
    int MaximalK=DIMENSION/3*2;  // maximal length
	int MinK=10;
    int classIndex=0;    // pick the class
	//double frequentThreshold=0;
	double PrecisionThresold=0;   // define the precision threshold
	double boundThrehold=3;  //define the parameter of the Chebyshev's inequality
	double boundThreshold2=3.5; // unimodal bound   95% percent
	double recallThreshold=0.1; 
	double probablityThreshold=0.9;
	//double boundThreshold2=2.11;// 10%

    vector<Feature *> reducedFset; // restore the set of the first round set cover
	
   int k=10;
   for (k=MinK;k<=MaximalK;k++)
   { // start of outer for
		int NumberOfSegments=(DIMENSION-k+1)*ClassNumber[classIndex];
		double ** DisA=new double *[NumberOfSegments]; 
		for (int i=0;i<NumberOfSegments;i++)
		{
			DisA[i]=new double[ROWTRAINING];
		}

    string filename="DisAc"; 
	string Cstring=IntToString(classIndex) ;
    filename=filename+Cstring+"k"; 
	string Kstring=IntToString(k) ;
    filename=filename+Kstring; 
    filename.insert(0,path);
    char * f;
    f = new char [filename.size()+1];
    strcpy (f, filename.c_str());

    ifstream inputFile( f, ios::in);
	if ( !inputFile )
	{
		//cerr << "file does not exist" << endl;
        DisArray(classIndex, k, DisA, NumberOfSegments, ROWTRAINING);
        SaveDisArray(DisA, NumberOfSegments, ROWTRAINING, f);
	}
    else
    {
		loadDisArray(f, DisA, NumberOfSegments, ROWTRAINING);
    }

	// test  a segment and its threshold
	//Feature * testFeature=ThresholdLearningAll(1676, 3, k, 0,  DisA, 0) ;
	//PrintFeature(testFeature);
	//testFeature=ThresholdLearningFirst(1676, 2.98, k, 0,  DisA, 0) ;
	//PrintFeature(testFeature);
	//testFeature=ThresholdLearningKDE(1676,k,  classIndex, DisA,  0, recallThreshold,  probablityThreshold);
	//PrintFeature(testFeature);

    // compute 
    totalBmap.reset();   // reset the Bmap for each length's set cover
    vector<Feature *> fSet; 
    int offset=ClassIndexes[classIndex]*(DIMENSION-k+1)+1;
    for (int s=0;s<NumberOfSegments;s++)
    {  
		Feature * temp;
		if (option==1)
		{
			 temp=ThresholdLearningAll(offset+s, boundThrehold, k, classIndex, DisA, recallThreshold);
		}
		else if (option==2)
		{
		    temp=ThresholdLearningFirst(offset+s, boundThreshold2, k, classIndex, DisA, PrecisionThresold);
		}
		else if (option==3)
		{
		  temp=ThresholdLearningKDE(offset+s,k,  classIndex, DisA,  PrecisionThresold, recallThreshold,  probablityThreshold);
		
		}
		else
		{
		    cout<<"\n invalide option";
			exit(0);
		}

	
	
       if (temp!=NULL)
       { fSet.push_back(temp);}
    }
   
    cout<<"\n the size of feature set of length k="<<k<<" is "<<fSet.size();

    // compute the non-redundent feature set by greedy
	while (totalBmap.count()>0)
	{ 
		int max=0; 
		int index=-1; 
		for (int i=0;i<fSet.size();i++)
		{
			int temp=fSet.at(i)->uncoveredRecall; 
			if (temp>max)
			{
				max=temp; 
				index=i;
			}
		}
       // move this feature to reducedFset and reset others 
       if (index>=0)
       {
		   Feature * currentFeature=fSet.at(index);
			fSet.erase(fSet.begin()+index); 
			reducedFset.push_back(currentFeature);
       
			for (int j=0;j<(currentFeature->bmap).size();j++)  // for the current set, update the other set
			{ 
				if ((currentFeature->bmap).at(j)==1) 
				{
					for (int jj=0;jj<fSet.size();jj++)
					{  
						if ((fSet.at(jj)->bmap).at(j)==1)
						{
							(fSet.at(jj)->bmap).reset(j); 
							fSet.at(jj)->uncoveredRecall=(fSet.at(jj)->uncoveredRecall)-1;
						}
					}
					totalBmap.reset(j);  // update the total covered set
				}
			}
       }
     }// end while
	  cout<<"\n reducedFset:"<<reducedFset.size();

	  // relase the memory
	 for (int i=0;i<NumberOfSegments;i++)
	 {
		delete [] DisA[i];
	 }
	  delete [] DisA;
	  for (int i=0;i< fSet.size();i++)
	  {   
	     delete [] fSet.at(i)->f; 
		 delete fSet.at(i);
	  }
	 
 } // end of outer for, end of the feature selection for each length

   //    for (int s=0;s<reducedFset.size();s++)
   // {
   // //   PrintFeature(reducedFset.at(s));
	  // PrintFeatureMoreInfo(reducedFset.at(s));
   //} 


   // second round of set cover
   ReduceAllLength(finalFset, reducedFset);
    for (int s=0;s<finalFset.size();s++)
    {  cout<<"\n"<<s;
       PrintFeature(finalFset.at(s));
	   //PrintFeatureMoreInfo(finalFset.at(s));
   } 


    //classification
    //classification(reducedFset, classIndex,MaximalK);
  classificationAllLength(finalFset, classIndex);
}

// second round of feature selection
void ReduceAllLength(vector<Feature *> & finalFset, vector<Feature *> &fSet )
{   
	// reset the uncovered reall after the first round feature reduction
	 for (int i=0;i<fSet.size(); i++)
	 {  fSet.at(i)->uncoveredRecall=(fSet.at(i)->ubmap).count();}
	
	 while (allLengthBmap.count()>0)
     { 
       //cout<<"\n"<<allLengthBmap.count();
		 int max=0; 
       int index=-1; 
       for (int i=0;i<fSet.size();i++)
       { int temp=fSet.at(i)->uncoveredRecall; 
         if (temp>max)
         {max=temp; index=i;}
       }
       // move this feature to reducedFset and reset others 
       if (index>=0)
       {Feature * currentFeature=fSet.at(index);
        fSet.erase(fSet.begin()+index); 
        finalFset.push_back(currentFeature);
		//cout<<"\n add feature";
		
       
        for (int j=0;j<(currentFeature->ubmap).size();j++)  // for the current set, update the other set
        { if ((currentFeature->ubmap).at(j)==1) 
           {
              for (int jj=0;jj<fSet.size();jj++)
              {  if ((fSet.at(jj)->ubmap).at(j)==1)
                  {
                      (fSet.at(jj)->ubmap).reset(j); 
                       fSet.at(jj)->uncoveredRecall=(fSet.at(jj)->uncoveredRecall)-1;
                  }
              }
              allLengthBmap.reset(j);  // update the total covered set
			 // cout<<"\n reset";
           }
        }

       }
     }// end while
	  cout<<"\n finalFset:"<<finalFset.size();
	   
	  for (int i=0;i< fSet.size();i++)
	  {   
	     delete [] fSet.at(i)->f; 
		 delete fSet.at(i);
	  }
}

// classification of one length
void classification(vector<Feature *> &Fs, int classIndex, int k)
{   cout<<"\nk="<<k;
    int TotalCount=0;
    int countDetected=0;
    int countDetectedInTarget=0;
    for (int i=0;i<ROWTESTING;i++)
    {
          bool matched=0;
            for (int j=0;j<=DIMENSION-k;j++)
            {    
                double * currentSegment=new double[k]; 

                for(int jj=0;jj<k;jj++)
                    currentSegment[jj]=testing[i][jj+j];

                for (int f=0;f<Fs.size();f++)
                { double temp=Euclidean(Fs.at(f)->f, currentSegment, k);
                  //cout<<"\n instance "<<i<<", the distance is "<<temp;
                   if (temp<=(Fs.at(f)->threshold))
                   {   matched=1;
                       countDetected++;
                       if(Classes[classIndex]==labelTesting[i])
                       {countDetectedInTarget++;}
                       break;}
                }
               if (matched==1)
               { break;}
            }
    }

	// count the total number in the target class
	int TargetClassTotal=0;
	
	
	for (int i=0;i<ROWTESTING;i++)
	{
	    if (labelTesting[i]==Classes[classIndex])
			TargetClassTotal++;

	}

    cout<<"\nthe recall of class "<<Classes[classIndex]<<" is:"<<(double)countDetectedInTarget/TargetClassTotal; 
    if (countDetected==0)
    {cout<<"\nthe precision of class "<<Classes[classIndex]<<" is:"<<0;}
    else
    cout<<"\nthe precision of class "<<Classes[classIndex]<<" is:"<<(double)countDetectedInTarget/countDetected; 

}


// online classification of all lengths
void classificationAllLength(vector<Feature *> &Fs, int classIndex)
{
    int TotalCount=0;

    int countDetected=0;
    int countDetectedInTarget=0;
	int sumLength=0;
    for (int i=0;i<ROWTESTING;i++)
    {   
          bool matched=0;
            for (int j=0;j<DIMENSION;j++)// j is the current ending position of the stream
            {    
               for (int f=0;f<Fs.size();f++)
               { 
                  int tempLength=Fs.at(f)->length;
                  int startingPosition=j-tempLength+1;
                  if (startingPosition>=0)
                  {   
                      double * currentseg=new double[tempLength];
                      for (int ss=0;ss<tempLength;ss++)
                      {currentseg[ss]=testing[i][ss+startingPosition];}
                      double tempDis=Euclidean(Fs.at(f)->f, currentseg, tempLength);
					  delete [] currentseg;
                      if (tempDis<=(Fs.at(f)->threshold))
                      {  cout<<"\n Instance "<<i<<"("<<labelTesting[i]<<") classified by segment "<<Fs.at(f)->segmentIndex<<"of length "<<Fs.at(f)->length<< "at ending position "<<j<<" as "<<Fs.at(f)->label;
                         matched=1;
						 predictedLength[i]=j; // store the classified length
						 sumLength=sumLength+j;
                         countDetected++;
                         if(Classes[classIndex]==labelTesting[i])
                            {countDetectedInTarget++;}
						 break;
                           // break to stop checking more features
                      }
                     
                  }
               }
                if (matched==1) // break the segment loop, finish the current testing example
               { break;}
            }
    }

	// count the total number in the target class
	int TargetClassTotal=0;
	
	for (int i=0;i<ROWTESTING;i++)
	{
	    if (labelTesting[i]==Classes[classIndex])
			TargetClassTotal++;

	}

    cout<<"\nthe recall of class "<<Classes[classIndex]<<" is:"<<(double)countDetectedInTarget/TargetClassTotal; 
    if (countDetected==0)
    {cout<<"\nthe precision of class "<<Classes[classIndex]<<" is:"<<0;}
    else
    cout<<"\nthe precision of class "<<Classes[classIndex]<<" is:"<<(double)countDetectedInTarget/countDetected; 
	cout<<"\n the averaged detection length "<<(double)sumLength/countDetected;

}

// print the information of the feature
void PrintFeature(Feature * f)
{
   if (f!=NULL)
   {
    cout<<"\n the feature=";
        for (int i=0;i<f->length;i++)
            cout<<(f->f)[i]<<" ";
    cout<<"\n segment Index="<<f->segmentIndex;
	cout<<"\n length="<<f->length;
    cout<<"\n threshod="<<f->threshold;
    cout<<"\n recall="<<f->recall;
    cout<<"\n precision="<<f->precision;
    //  cout<<"\n bitmap=";
    // for (int i=0;i<ROWTRAINING;i++)
    //   {cout<<(f->bmap).at(i)<<" ";}
   }
}

// this function is used to study the precision of best match and all matches
void PrintFeatureMoreInfo(Feature * f)
{
   if (f!=NULL)
   {
    cout<<"\n the feature=";
        for (int i=0;i<f->length;i++)
            cout<<(f->f)[i]<<" ";
    cout<<"\n segment Index="<<f->segmentIndex;
    cout<<"\n threshod="<<f->threshold;
    cout<<"\n recall="<<f->recall<<","<<(f->bmap).count();
	cout<<"\n recall="<<f->recall<<","<<(f->ubmap).count();
    cout<<"\n precision="<<f->precision;
	// compute the precision of all matches
	int countTotal=0; 
	int countTarget=0;
	for (int i=0;i<ROWTRAINING;i++)
		{for (int j=0;j<=DIMENSION-f->length;j++)
		{
		     // compute the Euclidean distance
			double * temp= new double[f->length]; 
			for (int ii=0;ii<f->length;ii++)
			{temp[ii]= training[i][j+ii];}
			double tempDis= Euclidean(f->f, temp, f->length);
			if (tempDis<=f->threshold)
			{
				  countTotal++; 
				  if (labelTraining[i]==f->label)
				  {
					  countTarget++;
				  }
			}
		}
	} // end of outer for
	cout<<"\n precision of all segments on training="<<(double)countTarget/countTotal;
   } // end of if
} // end of fuction


// print the bitmap
void PrintTotalBitMap()
{
    cout<<"\n the total bitmap=";
    for (int i=0;i<ROWTRAINING;i++)
    {
		 cout<<totalBmap.at(i)<<" ";
	} 
}

// learning threshold based on the one tail Chebyshev's inequality
Feature * ThresholdLearningAll(int index, double m, int k, int targetClass, double ** DisA, double RecallThreshold)  // this index starting from 1, m is the parameter in the bound, k is the length of feature
{
  int classofset=ClassIndexes[targetClass]*(DIMENSION-k+1)+1;
   Feature * currentf= new Feature(); 
   currentf->f=getSegment(index,k); 
   currentf->length=k;
   currentf->segmentIndex=index;
   currentf->label=Classes[targetClass];
   
   // get the non-target class part in the distance array
   int nonTargetTotal=0; 
   for (int c=0;c<NofClasses;c++)
   {
       if (c!=targetClass)
       {
           nonTargetTotal=nonTargetTotal+ClassNumber[c]; 
       }
   }

   double * nonTargetDis=new double[nonTargetTotal]; 
     
     int i=0;
     for (int c=0;c<NofClasses;c++)
	 {
		   if (c!=targetClass)
		   {   
			   int offset=ClassIndexes[c];
			   for (int e=0;e<ClassNumber[c];e++)
			   {
				   nonTargetDis[i]=DisA[index-classofset][offset+e];
				   i++;
			   }
		   }
	 }
   // compute the mean, standard deviation and the threshold
   double mu=getMean(nonTargetDis, nonTargetTotal); 
   double sd=getSTD(nonTargetDis, nonTargetTotal,mu);
   //cout<<"\nsdNonTarget="<<sd;

   currentf->threshold=mu-m*sd; 
   
   
   delete [] nonTargetDis; // release the memory

   // compute recall, precision and bitmap
   if (currentf->threshold>0)
   {
      int targetCount=0; 
      int totalCount=0;
      for (int i=0;i<ROWTRAINING;i++)
      {  double temp=DisA[index-classofset][i]; 
         if (temp<=currentf->threshold)
         {
            totalCount++; 
            if (labelTraining[i]==Classes[targetClass])
            {
				targetCount++;
                (currentf->bmap).set(i);  // set the bmap
			    (currentf->ubmap).set(i);// set the ubmap for the second round of set covering
            }
         }
      }
       currentf->recall=(double)targetCount/ClassNumber[targetClass]; // it is the absolute recall
       currentf->uncoveredRecall=targetCount; // this recall is used for the set cover computation
      
       currentf->precision=(double)targetCount/totalCount;
      
       if (currentf->recall>=RecallThreshold )
       {   
           for (int i=0;i<(currentf->bmap).size();i++)
           {
              if ((currentf->bmap).at(i)==1)
			  {
				  totalBmap.set(i);  //  set the total Bmap for each length's set cover
			      allLengthBmap.set(i); // set the all length Bmap for the second round of set cover
			  }
           }
           return currentf;
	   }
       else
       {   delete [] currentf->f; // release the memory
           delete currentf; 
           return NULL;
	   }
   }
   else
   {   
	   delete [] currentf->f;  // release the memory
       delete currentf;
       return NULL;
   }
   
}

double OneKDE(double * arr, int len, double q, double h, double constant)
{ 
	double temp=0;
	for (int i=0;i<len;i++)
	{
	     temp=temp+exp((arr[i]-q)*(arr[i]-q)* (-1/(2*h*h)));
	}
	
	temp=temp*constant;
	return temp;


}

// learning the threshold by KDE classification. 
Feature * ThresholdLearningKDE(int index, int k, int targetClass, double ** DisA, double PrecisionThreshold, double RecallThreshold, double ProbalityThreshold)
{
	 int classofset=ClassIndexes[targetClass]*(DIMENSION-k+1)+1;
	 Feature * currentf= new Feature(); 
	 currentf->f=getSegment(index,k); 
	 currentf->length=k;
	 currentf->segmentIndex=index;
	 currentf->label=Classes[targetClass];

	 // get the target class part and non-target part in the distance array
   int nonTargetTotal=0; 
   for (int c=0;c<NofClasses;c++)
   {
       if (c!=targetClass)
       {
           nonTargetTotal=nonTargetTotal+ClassNumber[c]; 
       }
   }
   int TargetTotal=ClassNumber[targetClass];

   double * nonTargetDis=new double[nonTargetTotal]; 
   double * TargetDis=new double[TargetTotal];
   double * CurrentDis=new double[ROWTRAINING];

     int nonTargeti=0; 
	 int Targeti=0;
	 int totali=0;
     for (int c=0;c<NofClasses;c++)
	 {    
		   int offset=ClassIndexes[c];
		   if (c!=targetClass)
		   {   
			   for (int e=0;e<ClassNumber[c];e++)
			   {
				   nonTargetDis[nonTargeti]=DisA[index-classofset][offset+e];
				   nonTargeti++;
				   CurrentDis[totali]=DisA[index-classofset][offset+e];
				   totali++;
			   }
		   }
		   else
		   {
		      for (int e=0;e<ClassNumber[c];e++)
			   {
				   TargetDis[Targeti]=DisA[index-classofset][offset+e];
				   Targeti++;
				   CurrentDis[totali]=DisA[index-classofset][offset+e];
				   totali++;
			   }
		   }
	 }
   // compute the mean, standard deviation and the threshold, and optimal h
	 // for the nonTarget Classes
   double muNonTarget=getMean(nonTargetDis, nonTargetTotal); 
   
   
   double sdNonTarget=getSTD(nonTargetDis, nonTargetTotal,muNonTarget);
   //cout<<"\nsdNonTarget="<<sdNonTarget;
  
   double hNonTarget=1.06* sdNonTarget/ pow (nonTargetTotal,0.2);

   double constantNT=1/(sqrt(2*3.14159265)*nonTargetTotal*hNonTarget);
   // for the TargetClasses
    double muTarget=getMean(TargetDis, TargetTotal); 
   
   double sdTarget=getSTD(TargetDis, TargetTotal,muTarget);
  
   double hTarget=1.06* sdTarget/ pow (TargetTotal,0.2);
   double constantT=1/(sqrt(2*3.14159265)*TargetTotal*hTarget);

   // sort the totalDis
   quicksort( CurrentDis, 0, ROWTRAINING-1);
   // compute the Probablity<0; 
   double NegativeTestPoint=-CurrentDis[ROWTRAINING-1]/(ROWTRAINING-1);
   double densityNonTarget=OneKDE(nonTargetDis, nonTargetTotal, NegativeTestPoint, hNonTarget,  constantNT); 
  // cout<<"\n densityNonTarget="<<densityNonTarget;
   double densityTarget=OneKDE(TargetDis, TargetTotal, NegativeTestPoint, hTarget,  constantT);
  // cout<<"\n densityTarget="<<densityTarget;
   double tempTarget=((double)ClassNumber[targetClass]/ROWTRAINING)*densityTarget;
 //  cout<<"\n tempTarget="<<tempTarget;
   double tempNonTarget=(1-((double)ClassNumber[targetClass]/ROWTRAINING))*densityNonTarget;
 //  cout<<"\n tempNonTarget="<<tempNonTarget;
   double ProTarget=tempTarget/( tempTarget+tempNonTarget);
  // cout<<"\nProbaNegative="<<ProTarget;

   if (ProTarget>ProbalityThreshold)
   {  
	   // compute the breaking Index
       int breakIndex=0;
	   for (int i=0;i<ROWTRAINING;i++)
	   {  
		  // cout<<"\n"<<i;
		   densityNonTarget=OneKDE(nonTargetDis, nonTargetTotal, CurrentDis[i], hNonTarget,  constantNT); 
		   densityTarget=OneKDE(TargetDis, TargetTotal, CurrentDis[i], hTarget,  constantT); 
		   tempTarget=((double)ClassNumber[targetClass]/ROWTRAINING)*densityTarget; 
		   tempNonTarget= (1-((double)ClassNumber[targetClass]/ROWTRAINING))*densityNonTarget; 
		   ProTarget= tempTarget/( tempTarget+tempNonTarget); 
		   //cout<<"\n"<<i<<" dis="<<CurrentDis[i]<<" Proba="<<ProTarget;
		   if (ProTarget<ProbalityThreshold) // belong to the non-target class
		   {
			  
			  breakIndex=i;
			  break; 
		   }
	   }
	     // compute the breaking point between breakingIndex and the previous point
	   if (breakIndex>=1)
	   {
		   int NonofBreakingPoint=20;
		   double value=0;
		   for (value=CurrentDis[breakIndex-1]; value<CurrentDis[breakIndex];value=value+(CurrentDis[breakIndex]-CurrentDis[breakIndex-1])/NonofBreakingPoint)
		   {  
			   //cout<<"\n"<<value;
			   densityNonTarget=OneKDE(nonTargetDis, nonTargetTotal, value, hNonTarget,  constantNT); 
			   densityTarget=OneKDE(TargetDis, TargetTotal, value, hTarget,  constantT); 
			   tempTarget=((double)ClassNumber[targetClass]/ROWTRAINING)*densityTarget; 
			   tempNonTarget= (1-((double)ClassNumber[targetClass]/ROWTRAINING))*densityNonTarget; 
			   ProTarget= tempTarget/( tempTarget+tempNonTarget); 
			   if (ProTarget<ProbalityThreshold) // belong to the non-target class
			   {
				//   cout<<"\nProba="<<ProTarget;
				  currentf->threshold=value;
				  break; 
			   }
	             
		   }
		   if (value>=CurrentDis[breakIndex])
		   {currentf->threshold=CurrentDis[breakIndex];}
	   }
	   else
	   {
	      currentf->threshold=-1;
	   }
	  
   }
   else
   {
      currentf->threshold=-1;
   }
   
   delete  nonTargetDis; 
   delete TargetDis;
   delete CurrentDis;
 
      if (currentf->threshold>0)
   {
      int targetCount=0; 
      int totalCount=0;
      for (int i=0;i<ROWTRAINING;i++)
      {  double temp=DisA[index-classofset][i]; 
         if (temp<=currentf->threshold)
         {
            totalCount++; 
            if (labelTraining[i]==Classes[targetClass])
            {
				targetCount++;
                (currentf->bmap).set(i);  // set the bmap
			    (currentf->ubmap).set(i);// set the ubmap for the second round of set covering
            }
         }
      }
       currentf->recall=(double)targetCount/ClassNumber[targetClass]; // it is the absolute recall
       currentf->uncoveredRecall=targetCount; // this recall is used for the set cover computation
      
       currentf->precision=(double)targetCount/totalCount;
      
       if (currentf->precision>=PrecisionThreshold  && currentf->recall>=RecallThreshold)
       {   
           for (int i=0;i<(currentf->bmap).size();i++)
           {
              if ((currentf->bmap).at(i)==1)
			  {
				  totalBmap.set(i);  //  set the total Bmap for each length's set cover
			      allLengthBmap.set(i); // set the all length Bmap for the second round of set cover
			  }
           }
           return currentf;
	   }
       else
       {   delete [] currentf->f; // release the memory
           delete currentf; 
           return NULL;
	   }
   }
   else
   {   
	   delete [] currentf->f;  // release the memory
       delete currentf;
       return NULL;
   }
}
// learning threshold based on the first unimode and the tigher bound
Feature * ThresholdLearningFirst(int index, double m, int k, int targetClass, double ** DisA, double PrecisionThreshold)  // this index starting from 1, m is the parameter in the bound, k is the length of feature
{
  int classofset=ClassIndexes[targetClass]*(DIMENSION-k+1)+1;
   Feature * currentf= new Feature(); 
   currentf->f=getSegment(index,k); 
   currentf->length=k;
   currentf->segmentIndex=index;
   currentf->label=Classes[targetClass];
   
   // get the non-target class part in the distance array
   int nonTargetTotal=0; 
   for (int c=0;c<NofClasses;c++)
   {
       if (c!=targetClass)
       {
           nonTargetTotal=nonTargetTotal+ClassNumber[c]; 
       }
   }

   double * nonTargetDis=new double[nonTargetTotal]; 
     
     int i=0;
     for (int c=0;c<NofClasses;c++)
	 {
		   if (c!=targetClass)
		   {   
			   int offset=ClassIndexes[c];
			   for (int e=0;e<ClassNumber[c];e++)
			   {
				   nonTargetDis[i]=DisA[index-classofset][offset+e];
				   i++;
			   }
		   }
	 }
   // compute the mean, standard deviation and the threshold
   double mu=getMean(nonTargetDis, nonTargetTotal); 
   
   double sd=getSTD(nonTargetDis, nonTargetTotal,mu);
  

   // optimal h
   double h=1.06* sd/ pow (nonTargetTotal,0.2);

   // find the first modal
   // sort the non target Dis
   quicksort( nonTargetDis, 0, nonTargetTotal-1);
 
  
   double * densities=new double[nonTargetTotal]; 
   double constant=1/(sqrt(2*3.14159265)*nonTargetTotal*h);
  // compute the kernel density and stop at the local minimum
   int firstUnimodalIndex=-1;
   for (int i=0;i<nonTargetTotal;i++)
   {
       densities[i]=OneKDE(nonTargetDis, nonTargetTotal, nonTargetDis[i], h,  constant); 
	   if (i>=2 && densities[i-1]<densities[i] &&densities[i-1]<densities[i-2])
	   {
	       firstUnimodalIndex=i-1; 
		   break;
	   }
	   if (i==nonTargetTotal-1)
	   {
	      firstUnimodalIndex=i;
	   }
   }
   //cout<<"\n firstUnimodalIndex="<<firstUnimodalIndex;
   double mu2=getMean(nonTargetDis, firstUnimodalIndex+1); 
    double sd2=getSTD(nonTargetDis, firstUnimodalIndex+1,mu2);

   
   currentf->threshold=mu2-m*sd2; 
   
   delete [] nonTargetDis; // release the memory
   delete [] densities;

   // compute recall, precision and bitmap
   if (currentf->threshold>0)
   {
      int targetCount=0; 
      int totalCount=0;
      for (int i=0;i<ROWTRAINING;i++)
      {  double temp=DisA[index-classofset][i]; 
         if (temp<=currentf->threshold)
         {
            totalCount++; 
            if (labelTraining[i]==Classes[targetClass])
            {
				targetCount++;
                (currentf->bmap).set(i);  // set the bmap
			    (currentf->ubmap).set(i);// set the ubmap for the second round of set covering
            }
         }
      }
       currentf->recall=(double)targetCount/ClassNumber[targetClass]; // it is the absolute recall
       currentf->uncoveredRecall=targetCount; // this recall is used for the set cover computation
      
       currentf->precision=(double)targetCount/totalCount;
      
       if (currentf->precision>=PrecisionThreshold )
       {   
           for (int i=0;i<(currentf->bmap).size();i++)
           {
              if ((currentf->bmap).at(i)==1)
			  {
				  totalBmap.set(i);  //  set the total Bmap for each length's set cover
			      allLengthBmap.set(i); // set the all length Bmap for the second round of set cover
			  }
           }
           return currentf;
	   }
       else
       {   delete [] currentf->f; // release the memory
           delete currentf; 
           return NULL;
	   }
   }
   else
   {   
	   delete [] currentf->f;  // release the memory
       delete currentf;
       return NULL;
   }
   
}

// compute the mean
double getMean( double * arr, int len)  // segmentIndex starting from 1
{  
     double sum=0;
     for (int i=0;i<len;i++)
     {
         sum=sum+arr[i];
     }
     return sum/len;
}

// compute the standard deviation given the mean
double getSTD(double * arr, int len, double mean)
{
     double sum=0;
     for (int i=0;i<len;i++)
     {
         sum=sum+(arr[i]-mean)*(arr[i]-mean);
     }
     return sqrt(sum/(len));
}

 // compute the best match distance array of the selected class
void DisArray(int classIndex, int k, double ** DisArray, int rows, int columns) 
{   
    // compute the DisArray
    int offset=ClassIndexes[classIndex]*(DIMENSION-k+1)+1;
    for (int i=0;i<rows;i++)
	{
      for (int j=0;j<columns;j++)
      {
          // compute the best match and store the distance.
          int segmentIndex=offset+i; 
          double * segment=getSegment(segmentIndex, k); 
            // compute the best match

          double minDis=10000; 
          for (int l=0;l<DIMENSION-k+1;l++)
          { 
             double ret = 0;
             for (int ii=0; ii<k;ii++)
             {
				  double dist = segment[ii]-training[j][l+ii];
				  ret += dist * dist;
				  if (ret>=minDis)
					  {
						  break; // early stopping
					  }
              }

             if (ret<minDis)
			 {
				 minDis=ret;
			 }
          }
		  delete [] segment;
          DisArray[i][j]=sqrt(minDis); 
      } // end of inner for
  //  cout<<i<<" ";
    } // end of outer for
}

// load the best match distance matrix 
void loadDisArray(const char * fileName, double ** Data, int rows, int columns)
{
  ifstream inputFile( fileName, ios::in);
	if ( !inputFile )
	{
		cerr << "file could not be opened" << endl;
		exit(1);
	} // end if
    
	int row=0;
	int col=0;
	while( !inputFile.eof() )
	{
		for ( row=0; row < rows; row++)
			for ( col=0; col < columns; col++)
			{
					inputFile>>Data[row][col];
            }	
	}
	inputFile.close(); 
}

// save the best match distance matrix to a text file
void SaveDisArray(double ** DisArray, int rows, int columns, const char * filename)
  // save it after compute
{ 
  ofstream outputFile(filename, ios::out);
  for (int row=0; row < rows; row++)
  {
     for (int col=0; col < columns; col++)
    {
       outputFile<<DisArray[row][col];
       outputFile<<" ";
     }
       outputFile<<endl;
  }   
    outputFile.close();
}

// get the n's segment of length k 
double * getSegment(int index, int k) // index starting from 1 to consistent with the matlab version
{
   // compute the starting index and return the index's segment
    if (k<=0 || k>DIMENSION)
      return NULL; 
    else
    {
       int eachLinebase=DIMENSION-k+1; 
       int row=index/eachLinebase; 
       int column=(index % eachLinebase)-1; 
       if (column==-1)
       {
		   row=row-1; 
           column=DIMENSION-k;
	   }
       double * segment=new double[k]; 
      //cout<<"\n segment "<<index<<" of length "<<k<<"\n"; 
       for (int i=0;i<k;i++)
       {
          segment[i]=training[row][column+i];
         // cout<<segment[i]<<" ";
       }
       return segment;
    }
}



// print training data
void printData(int index)
{
  cout<<"\nlabel="<<labelTraining[index]<<" "; 
  for (int i=0;i<DIMENSION;i++)
  {cout<<training[index][i]<<" ";}
}

// load the training and the testing data from text file
void LoadData(const char * fileName, double Data[][DIMENSION], double Labels[],int len  )
{
    
	ifstream inputFile( fileName, ios::in);
	if ( !inputFile )
	{
		cerr << "file could not be opened" << endl;
		exit(1);
	} // end if

	int row=0;
	int col=0;
	while( !inputFile.eof() )
	{
		for ( row=0; row < len; row++)
			for ( col=0; col < DIMENSION+1; col++)
			{
				if (col==0)
				{  
					inputFile>>Labels[row];
				}
				else
				{
					inputFile>>Data[row][col-1];
				}
			}
	}

	inputFile.close();
}

// the function of int to string
string IntToString(int intValue) 
{
	  char *myBuff;
	  string strRetVal;
	  // Create a new char array
	  myBuff = new char[100];
	  // Set it to empty
	  memset(myBuff,'\0',100);
	  // Convert to string
	  itoa(intValue,myBuff,10);
	  // Copy the buffer into the string object
	  strRetVal = myBuff;
	  // Delete the buffer
	  delete[] myBuff;
	  return(strRetVal);
}
