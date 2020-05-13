#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "math.h"
#include <set>
#include <algorithm>
#include <vector>
#include <time.h>
#include <limits>



#include "DataSetInformation.h"
#include "Euclidean.h"
#include "minValue.h"
#include "find.h"










using namespace std;
using namespace ECG;


// ALgorithm parameters: minimal support
int prefix=DIMENSION;

double MinimalSupport=0;

int strictversion=1; // 1 strict veresion , 0 loose version

// global variable
double training[ROWTRAINING][DIMENSION]; // training data set
double labelTraining[ROWTRAINING]={0}; // training data class labels
double testing [ROWTESTING][DIMENSION]; //  testing data set
double labelTesting[ROWTESTING]={0}; // testing data class labels
double predictedLabel[ROWTESTING]={0}; // predicted label by the classifier
int predictedLength[ROWTESTING]={0};// predicted length by the classifier
int  TrainingIndex[ROWTRAINING][DIMENSION];//  store the 1NN for each space, no ranking tie
double DisArray[ROWTRAINING][ROWTRAINING]={0}; //  the pairwise distance array of full length
int classDistri[NofClasses]={0};
double classSupport[NofClasses]={0};
int FullLengthClassificationStatus[ROWTRAINING];// 0 can not be classified correctly, 1 can be classified correctly
int PredictionPrefix[ROWTRAINING];
vector<set<int, less<int> >> NNSetList;// store the MNCS;
vector<int> NNSetListLength;
vector<int> NNSetListClassMap;// store the class label of the MNCS
int correct=0; // the unlabeled sequence which is correctly classified.
double classificationTime; // classification time of one instance
double trainingTime; // classification time of one instance



// functions in the same file
void LoadData(const char * fileName, double Data[][DIMENSION], double Labels[], int len  );
void getClassDis();
void LoadDisArray(const char * fileName, double Data[ROWTRAINING][ROWTRAINING]  );
void LoadTrIndex(const char * fileName, int Data[ROWTRAINING][DIMENSION]  );
set<int,less<int>> SetRNN(int l, set<int,less<int>>& s);
set<int,less<int>> SetNN(int l, set<int,less<int>>& s);
int NNConsistent(int l, set<int,less<int>>& s);
//void testUnion();
int getMPL(set<int,less<int>>& s); // loose version
int getMPL(set<int,less<int>>& s,int la, int lb); // loose version
//void getMNCS(int onenode,double label, set<int,less<int>>& PreMNCS);
int updateMPL(set<int,less<int>> s);
//bool sharePrefix(   set<int, less<int> >& s1, set<int, less<int> >& s2, set<int, less<int> >& s3, int level        );
void printSet(set<int, less<int> > & s);
void printSetList(vector<set<int, less<int> >> & v) ;
int findMin( int data[], int len);
void classification();
double  mean(int data[], int len);
void report();
double SetDis( set<int, less<int> > A, set<int, less<int> > B );
vector<int> RankingArray(vector<set<int,less<int>>> & List);
int GetMN(vector<set<int,less<int>>> & List, int ClusterIndex); // get the NN of cluster, if there is no MN, return -1, others , return the value;

int getMPLStrict(set<int,less<int>>& s,int la, int lb);
int getMPLStrict(set<int,less<int>>& s);
int updateMPLStrict(set<int,less<int>> s);


int getMPLTest(set<int,less<int>>& s,int la, int lb);
int getMPLTest(set<int,less<int>>& s);
int updateMPLTest(set<int,less<int>> s);
double NNclassification(int prefix, int option);
double LeaveOneOut(int le);
int FixedPrefix();
void Discretization(double s [DIMENSION], double label);
double findMin( double data[], int len);
double findMean( double data[], int len);
double findMax( double data[], int len);
int findMembership(double a, double means[3]);


int main ()
{
  

 //   // load training data
	LoadData(trainingFileName, training,labelTraining, ROWTRAINING);

     
   for (int i=0; i< ROWTRAINING;i++)
    {
        double temp[DIMENSION];
        
        for(int j=0;j<DIMENSION;j++)
        {temp[j]=training[i][j];
        
        
        
        }
        
         Discretization(temp, labelTraining[i]);

       
    }




   /*LoadData(testingFileName, testing,labelTesting, ROWTESTING);
     
    for (int i=0; i< ROWTESTING;i++)
    {
        double temp[DIMENSION];
        
        for(int j=0;j<DIMENSION;j++)
        {temp[j]=testing[i][j];
        
        
        
        }
        
         Discretization(temp, labelTesting[i]);

       
    }*/

}// end main



void Discretization(double s [DIMENSION],double label)
{
   
  

  //for(int j=0;j<DIMENSION;j++)
  // {cout<<s[j];}

  //
  //cout<<"\n";
  
   
  int membership[DIMENSION];
  double means[3];
  double newmeans[3]={0,0,0};
  means[0]=findMin(s, DIMENSION);
  means[1]=findMean(s, DIMENSION);
  means[2]=findMax(s, DIMENSION);

  cout<<"The minimal value is "<< means[0];
  cout<<"The max value is "<< means[2];
  cout<<"The average value is "<<means[1];

  while (1)
  {
      int count[3]={0,0,0};
      for(int j=0;j<DIMENSION;j++)
       {
           membership[j]=findMembership(s[j],means);
          //cout<<membership[j]<<" ";

           newmeans[membership[j]-1]=newmeans[membership[j]-1]+s[j];
           count[membership[j]-1]++;
  
       }
     
      if (count[0]>0)
      {newmeans[0]= newmeans[0]/count[0];}
      else
      {
       newmeans[0]=means[0];
      
      }
      if (count[1]>0)
      {
          newmeans[1]= newmeans[1]/count[1];
      }
      else
      {
      
      newmeans[1]=means[1];
      
      
      }
      
       if (count[2]>0)
       {newmeans[2]= newmeans[2]/count[2];}
       else
       { newmeans[2]=means[2];}
      


      if (means[0]==newmeans[0] && means[1]==newmeans[1] && means[2]==newmeans[2])
      {
         break;
      
      }
      else
      {

         // cout<<"\n oldmeans"<<means[0]<<","<<means[1]<<","<<means[2];
         // cout<<"\n newmeans"<<newmeans[0]<<","<<newmeans[1]<<","<<newmeans[2];
         // char c;
        //  cin>>c;
       means[0]=newmeans[0];
       means[1]=newmeans[1];
       means[2]=newmeans[2];
       newmeans[0]=0;
       newmeans[1]=0;
       newmeans[2]=0;
      
      
      }
  
  }
  
  ofstream outputFile("Two_Patterns/TrDisCBF.txt", ios::out|ios::app);
  outputFile<<int(label)<<",";
  
  for(int j=0;j<DIMENSION;j++)
       {
          
          cout<<membership[j];
          outputFile<<membership[j];
  
       }
  
  cout<<"\n";
  outputFile<<"\n";

}



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


void LoadDisArray(const char * fileName, double Data[ROWTRAINING][ROWTRAINING]  )
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
		for ( row=0; row < ROWTRAINING; row++)
			for ( col=0; col < ROWTRAINING; col++)
			{

				
				
					inputFile>>Data[row][col];
				
			}
			
	}

	inputFile.close();
}

void LoadTrIndex(const char * fileName, int Data[ROWTRAINING][DIMENSION]  )
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
		for ( row=0; row < ROWTRAINING; row++)
			for ( col=0; col < DIMENSION; col++)
			{

				
				
					inputFile>>Data[row][col];
				
			}
			
	}

	inputFile.close();
}



void getClassDis()
{
   for (int i=0; i<NofClasses;i++)
   {  
	 
   
       classDistri[i]=find(labelTraining, Classes[i], ROWTRAINING );
	   classSupport[i]= classDistri[i]*MinimalSupport;
   
   }

  


}








set<int,less<int>> SetRNN(int l, set<int,less<int>>& s) // find a set's RNN on prefix l
{
    set<int, less<int>> index;

    set<int, less<int> >::iterator i;
    for (i=s.begin(); i!=s.end(); i++)
	{
	   int element=*i;

	   for (int j=0;j<ROWTRAINING;j++)
	   {
	       if (TrainingIndex[j][l-1]==element && s.count(j)==0)
		   {
		     index.insert(j);
		   }
	   
	   
	   
	   
	   }
	
	
	}

	 
	return index;

	


}

set<int,less<int>> SetNN(int l, set<int,less<int>>& s)
{
    set<int, less<int>> index;

    set<int, less<int> >::iterator i;
    for (i=s.begin(); i!=s.end(); i++)
	{
	   int element=*i;

	   int NNofelement=TrainingIndex[element][l-1];
		index.insert(NNofelement);
		   
	   
	   
	}

	 
	return index;

	


}
int NNConsistent(int l, set<int,less<int>>& s)// return 1, if it is NN consistent, return 0, if not
{
 set<int, less<int> >::iterator i;
 int consistent=1;
  for (i=s.begin(); i!=s.end(); i++)
  {
      int element=*i;

	  int NNofelement=TrainingIndex[element][l-1]; // get the NN
  
      if (s.count(NNofelement)==0)
	  {
		  consistent=0;
		  break;
	  
	  }
  
  
  }


 return consistent;

}







int getMPL(set<int,less<int>>& s)
{ 
    
  int MPL=DIMENSION;

  set<int, less<int> > ::iterator i;
  if (s.size()==1) // simple RNN Method
  {
    set<int,less<int>> LastRNN=SetRNN(DIMENSION,s); // get full length RNN
    
	
             
	
    
	int FirstElement=*(s.begin());
	double label=labelTraining[FirstElement];
    int labelIndex=0;
	if (label==-1)
	   {labelIndex=1;}
	else
	   {labelIndex=label-1;}


	// get the intersect(RNN(S,L), T')=RNN(S,L) in the same class as S. 
    set<int,less<int>> LastUsefulRNN;
    for (i=LastRNN.begin(); i!=LastRNN.end(); i++)
	{
	    int element=*i;

		if ( FullLengthClassificationStatus[element]==1)
		{
		
		   LastUsefulRNN.insert(element);
		
		}
	
	
	
	}
	// compute the support of the sequence
	int Support=s.size()+LastUsefulRNN.size();
	

	if (Support>=classSupport[labelIndex])
	{
	   if (LastUsefulRNN.size()>0)
	   {
           set<int,less<int>> PreviousRNN;
	      for (int le=DIMENSION-1;le>=1;le--)
		  {
		     

             PreviousRNN=SetRNN(le,s);
			  

             set<int,less<int>> PreviousUsefulRNN;


		     for (i=PreviousRNN.begin(); i!=PreviousRNN.end(); i++)
				{
					int element=*i;

					if (FullLengthClassificationStatus[element]==1)
					{
					
					   PreviousUsefulRNN.insert(element);
					   
					
					}
				
				
				
				}
            


			 if (  LastUsefulRNN==PreviousUsefulRNN )
			 {
				 

                
                   // CurrentUsefulRNN=PreviousUsefulRNN;

                    

                  
			         PreviousRNN.clear();
			        PreviousUsefulRNN.clear();
				 
				
			  
			 }
			 else
			 { 
                 /*printSet(s);
                 printSet(PreviousUsefulRNN);
                 cout<<"Support="<<support<<"\n";*/
				 
				 MPL=le+1;
                 break;
			 
			 
			 }

		  
		  
		  
		  
		  }// end of for
	   
	   
	   
	   
	   }
	   else
	   {MPL=DIMENSION;}
	}
	else
	{
	  MPL=DIMENSION;
	}

  
  
  
  }
  else // super-sequence Method
  {
    set<int,less<int>> LastRNN=SetRNN(DIMENSION,s); // get full length RNN
	
    
	int FirstElement=*(s.begin());
	double label=labelTraining[FirstElement];
    int labelIndex=0;
	if (label==-1)
	   {labelIndex=1;}
	else
	   {labelIndex=label-1;}


	// get the intersect(RNN(S,L), T')=RNN(S,L) in the same class as S. 
    set<int,less<int>> LastUsefulRNN;
    for (i=LastRNN.begin(); i!=LastRNN.end(); i++)
	{
	    int element=*i;

		if (FullLengthClassificationStatus[element]==1)
		{
		
		  LastUsefulRNN.insert(element);
		
		}
	
	
	
	}
	// compute the support of the sequence
	int Support=s.size()+LastUsefulRNN.size();

	if (Support>=classSupport[labelIndex])
	{
       
        set<int,less<int>> PreviousRNN;
		for (int le=DIMENSION-1;le>=1;le--)
		  {
		     if (NNConsistent(le,s)==0)
			 {
			   MPL=le+1;
			   break;
			 }
			 else
			 {
			  
			  
			  
			  
			  
			

             PreviousRNN=SetRNN(le,s);

             set<int,less<int>> PreviousUsefulRNN;


		     for (i=PreviousRNN.begin(); i!=PreviousRNN.end(); i++)
				{
					int element=*i;

					if ( FullLengthClassificationStatus[element]==1)
					{
					
					   PreviousUsefulRNN.insert(element);
					   
					
					}
				
				
				
				}

			 if ( LastUsefulRNN==PreviousUsefulRNN)
			 {
				 
				   
			         PreviousRNN.clear();
			        PreviousUsefulRNN.clear();
				 
				 
			  
			 }
			 else
			 { 
				 /*printSet(s);
                 printSet(PreviousUsefulRNN);
                 cout<<"Support="<<support<<"\n";*/
				 MPL=le+1;
                 break;
			 
			 
			 }

		  
			 }// end of else
		  
		  
		  }// end of for
	   
	   
	   
	   
	   
	}
	else
	{
	  MPL=DIMENSION;
	}

  
  
  
  }
      














  return MPL;
  
  
  }






int getMPLStrict(set<int,less<int>>& s)
{ 
    
  int MPL=DIMENSION;

  set<int, less<int> > ::iterator i;
  if (s.size()==1) // simple RNN Method
  {
    set<int,less<int>> LastRNN=SetRNN(DIMENSION,s); // get full length RNN
    
	
             
	
    
	int FirstElement=*(s.begin());
	double label=labelTraining[FirstElement];
    int labelIndex=0;
	if (label==-1)
	   {labelIndex=1;}
	else
	   {labelIndex=label-1;}


	
	// compute the support of the sequence
	int Support=s.size()+LastRNN.size();
	

	if (Support>=classSupport[labelIndex])
	{
	   if (LastRNN.size()>0)
	   {
           set<int,less<int>> PreviousRNN;
	      for (int le=DIMENSION-1;le>=1;le--)
		  {
		     

             PreviousRNN=SetRNN(le,s);
			  



			 if (  LastRNN==PreviousRNN )
			 {
				 

			         PreviousRNN.clear();
			       
				 
				
			  
			 }
			 else
			 { 
                
				 
				 MPL=le+1;
                 break;
			 
			 
			 }

		  
		  
		  
		  
		  }// end of for
	   
	   
	   
	   
	   }
	   else
	   {MPL=DIMENSION;}
	}
	else
	{
	  MPL=DIMENSION;
	}

  
  
  
  }
  else // super-sequence Method
  {
    set<int,less<int>> LastRNN=SetRNN(DIMENSION,s); // get full length RNN
	
    
	int FirstElement=*(s.begin());
	double label=labelTraining[FirstElement];
    int labelIndex=0;
	if (label==-1)
	   {labelIndex=1;}
	else
	   {labelIndex=label-1;}


	
    
	// compute the support of the sequence
	int Support=s.size()+LastRNN.size();

	if (Support>=classSupport[labelIndex])
	{
       
        set<int,less<int>> PreviousRNN;
		for (int le=DIMENSION-1;le>=1;le--)
		  {
		     if (NNConsistent(le,s)==0)
			 {
			   MPL=le+1;
			   break;
			 }
			 else
			 {
			  
			  
			  
			  
			  
			

             PreviousRNN=SetRNN(le,s);

           
			 if ( LastRNN==PreviousRNN)
			 {
				 
				   
			         PreviousRNN.clear();
			     
				 
				 
			  
			 }
			 else
			 { 
				 
				 MPL=le+1;
                 break;
			 
			 
			 }

		  
			 }// end of else
		  
		  
		  }// end of for
	   
	   
	   
	   
	   
	}
	else
	{
	  MPL=DIMENSION;
	}

  
  
  
  }
      














  return MPL;
  
  
  }







int updateMPL(set<int,less<int>> s)
{
 
      int length= getMPL(s);
    

	  set<int, less<int> > ::iterator jj;
      for (jj=s.begin(); jj!=s.end(); jj++)
	  {
	        if (PredictionPrefix[*jj]>length)
			{
			    PredictionPrefix[*jj]=length;
			
			}
	  
	  
	  
	
   
       }

  return length;


}


 
int updateMPLStrict(set<int,less<int>> s)
{
 
      int length= getMPLStrict(s);
    

	  set<int, less<int> > ::iterator jj;
      for (jj=s.begin(); jj!=s.end(); jj++)
	  {
	        if (PredictionPrefix[*jj]>length)
			{
			    PredictionPrefix[*jj]=length;
			
			}
	  
	  
	  
	
   
       }

  return length;


}

   
  
bool sharePrefix(   set<int, less<int> >& s1, set<int, less<int> >& s2, set<int, less<int> >& s3, int level        )

{
    
 set<int, less<int> > ::iterator i,j;
 bool SharePrefix=1;
 int count=0;
 for ( count=0, i=s1.begin(),j=s2.begin(); count<level-2;count++, i++,j++)
 {
   if ((*i) != (*j))
   {   
	   SharePrefix=0;
	   break;
   }
 
 }
      
  
if (SharePrefix==1)
 {
    set_union(s1.begin(), s1.end(), s2.begin(), s2.end(),
           insert_iterator<set<int, less<int> >> (s3,s3.begin()) );
	/*cout<<"Merged set:\n";
    for(i=s3.begin();i!=s3.end();i++)
   {
   cout<<(*i)<<" ";
    }
   cout<<"\n";*/
	return SharePrefix;
 }
 else

 {
	 
	//cout<<"\n can not merge";
	 return SharePrefix;
 
 }
 
}
void printSetList(vector<set<int, less<int> >> & v) 
{ cout<<"\n";
   
   for (int i=0;i<v.size();i++)
   {
	   cout<<"Set "<<i<<":";

      set<int, less<int> > temp=v[i];
	  //printSet(temp);

   
   
   
   
   
   }

 
 
 
 
}

void printSet(set<int, less<int> > & s)
 {
 
	 set<int, less<int> > :: iterator i;

	 for (i=s.begin();i!=s.end();i++)
		 cout<<*i<<" ";
	 cout<<endl;
 
 
 }
int findMin( int data[], int len)
{
 int Min=DIMENSION;
 for (int i=0;i<len;i++)
 {
   if (data[i]<Min)
   {
	   Min=data[i];
   
   }
 
 
 }
 return Min;



}


double findMin( double data[], int len)
{
 double Min=data[0];
 for (int i=0;i<len;i++)
 {
   if (data[i]<Min)
   {
	   Min=data[i];
   
   }
 
 
 }
 return Min;



}


double findMax( double data[], int len)
{
 double Max=data[0];
 for (int i=0;i<len;i++)
 {
   if (data[i]>Max)
   {
	   Max=data[i];
   
   }
 
 
 }
 return Max;



}


double findMean( double data[], int len)
{
 double sum=0;
 for (int i=0;i<len;i++)
 {
   sum=sum+data[i];
 
 
 }
 return sum/len;



}

int findMembership(double a, double means[3])
{
   
    double temp0=a-means[0];
    if (temp0<0)
    {temp0=-temp0;}

    double temp1=a-means[1];
    if (temp1<0)
    {temp1=-temp1;}
  
    double temp2=a-means[2];
    if (temp2<0)
    {temp2=-temp2;}

    if (temp0<=temp1 && temp0<=temp2)
        return 1;
    if (temp1<temp0 && temp1<temp2)
        return 2;
    if (temp2<temp0 && temp2<temp1)
        return 3;


}


int findNN(int index,int len)
{
   int indexOfNN=-1;

   double Mindis=100000;


   for (int i=0;i<ROWTRAINING;i++)
   {
	  double tempdis=Euclidean( testing[index], training[i], len );

		  if (tempdis<Mindis)
		  {
		     Mindis=tempdis;
			 indexOfNN=i;
		  
		  }
	   
	}
	   
	   
	  return indexOfNN;
}

void classification()
{

  LoadData(testingFileName, testing,labelTesting, ROWTESTING);
  int startfrom=findMin( PredictionPrefix, ROWTRAINING);
  cout<<"\n Start from: "<<startfrom<<endl;

  clock_t t1, t2;

  t1 = clock();
  
  cout<<(double)t1;





  for (int i=0;i<ROWTESTING;i++)
  {
       for (int j=startfrom;j<=DIMENSION;j++)
	   {
	      
	      int tempNN=findNN(i,j);

         // cout<<"\n instance "<<i<<" 's NN is "<<tempNN<<" at length "<<j;

		  if (PredictionPrefix[tempNN]<=j)
		  {
		    
			predictedLabel[i]=labelTraining[tempNN];
			predictedLength[i]=j;

			if (predictedLabel[i]==labelTesting[i])
			{correct++;}

			//cout<<"\n instance "<<i<<" ("<<labelTesting[i]<<")  is classified by instance "<<tempNN <<" at length "<<predictedLength[i] <<"as  "<<predictedLabel[i];
		    break; // can be classified.
		  }
	   
	   
	   }
  
  
  
  }

  t2 = clock();
  cout<<"\n finish time: "<<(double)t2<<endl;

  classificationTime = (double)(t2 - t1)/CLOCKS_PER_SEC  ;

  
}



double  mean(int data[], int len)
{   
	double sum=0;
    for (int i=0;i<len;i++)
	{
	  sum=sum+data[i];
	}

	return sum/len;


}





void reportSynUCI()
{
   for (int i=0;i<6;i++)
   {
       double sum=0; correct=0;
       for(int j=i*50;j<i*50+50;j++)
       {
       
          sum=sum+predictedLength[j];
          if (predictedLabel[j]==i+1)
          {
             correct++;
          }
       
       
       }
       cout<<"Prediction length of class "<<i<<" is "<<sum/50<<endl;
       cout<<"Accuracy of class "<<i<<" is "<<(double)correct/50<<endl;
   
   
   }
   
  /* cout<<"predictedLabel"<<endl;
   for (int i=0;i<300;i++)
   {cout<<predictedLabel[i]<<" ";}*/
   



}
void report()
  {
      cout<<"\n NN classifier<< prefix: "<<prefix<<"\n";

      cout<<" Heuristic Algorithm Report\n";
	  cout<< "The averaged predicted length of testing data is "<<mean(predictedLength,ROWTESTING)<<"\n";
	  cout<< "The averaged prediction prefix of training data is "<<mean(PredictionPrefix,ROWTRAINING)<<"\n";
	  cout<<" The accuracy is "<< double(correct)/ROWTESTING<<"\n";
	  cout<<" The error rate is"<<1-double(correct)/ROWTESTING<<"\n";
	  cout<<"Average time of classifying one instance "<<classificationTime/ROWTESTING<<" seconds\n";
	  cout<<"Training Time "<<trainingTime<<" seconds\n";
      //reportSynUCI();

      // compute the false positive  and true positve
      int FP=0, TP=0, TC=0,FC=0;
      for (int i=0;i<ROWTESTING;i++)
      {
      
         if (predictedLabel[i]==+1 &&labelTesting[i]==-1)
         {
             FP++;
         }

         if (predictedLabel[i]==1 &&labelTesting[i]==1)
         {
             TP++;
         }

         if(labelTesting[i]==1)
         { TC++;}
         else
         { FC++;}
      
      
      }
      
      double FPRate=(double)FP/FC;
      double TPRate=(double)TP/TC;
      
      cout<<"FalsePostive: "<<FPRate<<"\n";
      cout<<"True Postive: "<<TPRate<<"\n";

	  ofstream outputFile(ResultfileName, ios::out|ios::app);
      outputFile<<"\n NN classifier<< prefix: "<<prefix<<"\n";
      outputFile<<"\n Report (Heuristic) with Minimal support="<<MinimalSupport<<"\n";
	  outputFile<< "The averaged predicted length of testing data is "<<mean(predictedLength,ROWTESTING)<<"\n";
      outputFile<< "The averaged prediction prefix of training data is "<<mean(PredictionPrefix,ROWTRAINING)<<"\n";
	  outputFile<<" The accuracy is "<< double(correct)/ROWTESTING<<"\n";
	  outputFile<<" The error rate is"<<1-double(correct)/ROWTESTING<<"\n";
	  outputFile<<"Average time of classifying one instance "<<classificationTime/ROWTESTING<<" seconds\n";
	  outputFile<<"Training Time "<<trainingTime<<" seconds \n";
      outputFile<<"FalsePostive: "<<FPRate<<"\n";
      outputFile<<"True Postive: "<<TPRate<<"\n";

	  outputFile.close();

    /*
     
if NofClasses==2
disp(['True positive Rate ',num2str( double(TP)/classDistri(1) )   ])
disp(['The false positive rates ',num2str(double(FP)/classDistri(2)   )])
end

for i=1:NofClasses
   temp= length(find(labelTesting==Classes(i)));
disp([num2str(Classes(i)), ': ', num2str(countingClass(i)/temp)])
end */
  
  
  
  
  }


double SetDis( set<int, less<int> > A, set<int, less<int> > B )
{

 
	  double minimal=10000000;

	  

     set<int, less<int> > :: iterator iA;
	 set<int, less<int> > :: iterator iB;

	  for ( iA=A.begin();iA!=A.end();iA++)
		  
	  {
		  for (iB=B.begin();iB!=B.end();iB++)
		  {

		    double temp=DisArray[*iA][*iB];
			//cout<<"dis "<<temp<<endl;
			
            if (minimal>temp)
            {
               minimal=temp;
            
            }
		  
		  }
	  
	  }

	  return minimal;

	  
	
}












vector<int> RankingArray(vector<set<int,less<int>>> & List)

{
   
	vector<int> result;

	for (int i=0;i<List.size();i++)
    {
	    double Mindis=100000;
		int MinIndex=-1;
		
		for (int j=0;j<List.size();j++)
		{
		  if(j!=i)
		  {
		     double tempdis= SetDis( List[i], List[j]);
		  
		      if(tempdis<Mindis)
			  {
			      Mindis=tempdis;
                  MinIndex=j;

			  }
		  
		  
		  }


		 } // end for inside
	
	
	result.push_back(MinIndex);
	
	
	
	}// endfor outside

  return result;

}




int GetMN(vector<set<int,less<int>>> & List, int ClusterIndex) // get the NN of cluster, if there is no MN, return -1, others , return the value
{
   // find the NN of ClusterIndex
    double Mindis=100000;
	int NN=-1;
   for (int i=0;i<List.size();i++)
   {
       
      if (i!=ClusterIndex)
      {
      
            double tempdis= SetDis( List[ClusterIndex], List[i]);
		  
		      if(tempdis<Mindis)
			  {
			      Mindis=tempdis;
                  NN=i;

			  }
      }
   
   
   }

   // find the NN of MinIndex

   Mindis=100000;
	int MinIndex=-1;
     for (int i=0;i<List.size();i++)
   {
       
      if (i!=NN)
      {
      
            double tempdis= SetDis( List[NN], List[i]);
		  
		      if(tempdis<Mindis)
			  {
			      Mindis=tempdis;
                  MinIndex=i;

			  }
      }
   
   
   }

  if (MinIndex==ClusterIndex)
  {return NN;}
  else
  {return -1;}



}
int getMPL(set<int,less<int>>& s,int la, int lb)
{
  int startFrom=0;

  if (la<=lb)
      startFrom=lb;
  else
      startFrom=la;

  int MPL=DIMENSION;

  set<int, less<int> > ::iterator i;
  if (s.size()==1) // simple RNN Method
  {
    set<int,less<int>> LastRNN=SetRNN(DIMENSION,s); // get full length RNN

	
             
	
    
	int FirstElement=*(s.begin());
	double label=labelTraining[FirstElement];
    int labelIndex=0;
	if (label==-1)
	   {labelIndex=1;}
	else
	   {labelIndex=label-1;}


	// get the intersect(RNN(S,L), T')=RNN(S,L) in the same class as S. 
    set<int,less<int>> LastUsefulRNN;
    for (i=LastRNN.begin(); i!=LastRNN.end(); i++)
	{
	    int element=*i;

		if ( FullLengthClassificationStatus[element]==1)
		{
		
		   LastUsefulRNN.insert(element);
		
		}
	
	
	
	}
	// compute the support of the sequence
	int Support=s.size()+LastUsefulRNN.size();
	

	if (Support>=classSupport[labelIndex])
	{
	   if (LastUsefulRNN.size()>0)

	   {
	      set<int,less<int>> PreviousRNN;
           for (int le=DIMENSION-1;le>=1;le--)
		  {
		   

             PreviousRNN=SetRNN(le,s);
			  

             set<int,less<int>> PreviousUsefulRNN;


		     for (i=PreviousRNN.begin(); i!=PreviousRNN.end(); i++)
				{
					int element=*i;

					if (FullLengthClassificationStatus[element]==1)
					{
					
					   PreviousUsefulRNN.insert(element);
					   
					
					}
				
				
				
				}
            


			 if (  LastUsefulRNN==PreviousUsefulRNN )
			 {
				 
				   
			         PreviousRNN.clear();
			        PreviousUsefulRNN.clear();
				 
				
			  
			 }
			 else
			 { 
                 /*printSet(s);
                 printSet(PreviousUsefulRNN);
                 cout<<"Support="<<support<<"\n";*/
				 
				 MPL=le+1;
                 break;
			 
			 
			 }

		  
		  
		  
		  
		  }// end of for
	   
	   
	   
	   
	   }
	   else
	   {MPL=DIMENSION;}
	}
	else
	{
	  MPL=DIMENSION;
	}

  
  
  
  }
  else // super-sequence Method
  {
    set<int,less<int>> LastRNN=SetRNN(DIMENSION,s); // get full length RNN
	
    
	int FirstElement=*(s.begin());
	double label=labelTraining[FirstElement];
    int labelIndex=0;
	if (label==-1)
	   {labelIndex=1;}
	else
	   {labelIndex=label-1;}


	// get the intersect(RNN(S,L), T')=RNN(S,L) in the same class as S. 
    set<int,less<int>> LastUsefulRNN;
    for (i=LastRNN.begin(); i!=LastRNN.end(); i++)
	{
	    int element=*i;

		if (FullLengthClassificationStatus[element]==1)
		{
		
		   LastUsefulRNN.insert(element);
		
		}
	
	
	
	}
	// compute the support of the sequence
	int Support=s.size()+LastUsefulRNN.size();
    set<int,less<int>> PreviousRNN;
    set<int,less<int>> PreviousUsefulRNN;
	if (Support>=classSupport[labelIndex])
	{
		for (int le=startFrom-1;le>=1;le--)
		  {
		     if (NNConsistent(le,s)==0)
			 {
			   MPL=le+1;
			   break;
			 }
			 else
			 {
			  
			  
			  
			  
			  
			 

             PreviousRNN=SetRNN(le,s);

           


		     for (i=PreviousRNN.begin(); i!=PreviousRNN.end(); i++)
				{
					int element=*i;

					if ( FullLengthClassificationStatus[element]==1)
					{
					
					   PreviousUsefulRNN.insert(element);
					   
					
					}
				
				
				
				}

			 if ( LastUsefulRNN==PreviousUsefulRNN)
			 {
				 
				    PreviousRNN.clear();
			        PreviousUsefulRNN.clear();
				 
				 
			  
			 }
			 else
			 { 
				 /*printSet(s);
                 printSet(PreviousUsefulRNN);
                 cout<<"Support="<<support<<"\n";*/
				 MPL=le+1;
                 break;
			 
			 
			 }

		  
			 }// end of else
		  
		  
		  }// end of for
	   
	   
	   
	   
	   
	}
	else
	{
	  MPL=DIMENSION;
	}

  
  
  
  }
      














  return MPL;
  



}







int getMPLStrict(set<int,less<int>>& s,int la, int lb)
{
  int startFrom=0;

  if (la<=lb)
      startFrom=lb;
  else
      startFrom=la;

  int MPL=DIMENSION;

  set<int, less<int> > ::iterator i;
  if (s.size()==1) // simple RNN Method
  {
    set<int,less<int>> LastRNN=SetRNN(DIMENSION,s); // get full length RNN

	
             
	
    
	int FirstElement=*(s.begin());
	double label=labelTraining[FirstElement];
    int labelIndex=0;
	if (label==-1)
	   {labelIndex=1;}
	else
	   {labelIndex=label-1;}


	
	// compute the support of the sequence
	int Support=s.size()+LastRNN.size();
	

	if (Support>=classSupport[labelIndex])
	{
	   if (LastRNN.size()>0)

	   {
	      set<int,less<int>> PreviousRNN;
           for (int le=DIMENSION-1;le>=1;le--)
		  {
		   

             PreviousRNN=SetRNN(le,s);
			  
			 if (  LastRNN==PreviousRNN )
			 {
				 
				   
			         PreviousRNN.clear();
			       
			 }
			 else
			 { 
                
				 
				 MPL=le+1;
                 break;
			 
			 
			 }

		  
		  
		  
		  
		  }// end of for
	   
	   
	   
	   
	   }
	   else
	   {MPL=DIMENSION;}
	}
	else
	{
	  MPL=DIMENSION;
	}

  
  
  
  }
  else // super-sequence Method
  {
    set<int,less<int>> LastRNN=SetRNN(DIMENSION,s); // get full length RNN
	
    
	int FirstElement=*(s.begin());
	double label=labelTraining[FirstElement];
    int labelIndex=0;
	if (label==-1)
	   {labelIndex=1;}
	else
	   {labelIndex=label-1;}

	// compute the support of the sequence
	int Support=s.size()+LastRNN.size();
    set<int,less<int>> PreviousRNN;
   
	if (Support>=classSupport[labelIndex])
	{
		for (int le=startFrom-1;le>=1;le--)
		  {
		     if (NNConsistent(le,s)==0)
			 {
			   MPL=le+1;
			   break;
			 }
			 else
			 {
			  
			  
			  
			  
			  
			 

             PreviousRNN=SetRNN(le,s);

           



			 if ( LastRNN==PreviousRNN)
			 {
				 
				    PreviousRNN.clear();
			      
				 
				 
			  
			 }
			 else
			 { 
				 /*printSet(s);
                 printSet(PreviousUsefulRNN);
                 cout<<"Support="<<support<<"\n";*/
				 MPL=le+1;
                 break;
			 
			 
			 }

		  
			 }// end of else
		  
		  
		  }// end of for
	   
	   
	   
	   
	   
	}
	else
	{
	  MPL=DIMENSION;
	}

  
  
  
  }
      














  return MPL;
  



}





int getMPLTest(set<int,less<int>>& s,int la, int lb)
{
  int startFrom=0;

  if (la<=lb)
      startFrom=lb;
  else
      startFrom=la;

  int MPL=DIMENSION;

  set<int, less<int> > ::iterator i;
  if (s.size()==1) // simple RNN Method
  {
    set<int,less<int>> LastRNN=SetRNN(DIMENSION,s); // get full length RNN

	
             
	
    
	int FirstElement=*(s.begin());
	double label=labelTraining[FirstElement];
    int labelIndex=0;
	if (label==-1)
	   {labelIndex=1;}
	else
	   {labelIndex=label-1;}


	
	// compute the support of the sequence
	int Support=s.size()+LastRNN.size();
	

	if (Support>=classSupport[labelIndex])
	{
	   if (LastRNN.size()>0)

	   {
	      set<int,less<int>> PreviousRNN;
           for (int le=DIMENSION-1;le>=1;le--)
		  {
		   

             PreviousRNN=SetRNN(le,s);
			  
			 if (  LastRNN==PreviousRNN )
			 {
				 
				   
			         PreviousRNN.clear();
			       
			 }
			 else
			 { 
                
				 
				 MPL=le+1;
                 break;
			 
			 
			 }

		  
		  
		  
		  
		  }// end of for
	   
	   
	   
	   
	   }
	   else
	   {MPL=DIMENSION;}
	}
	else
	{
	  MPL=DIMENSION;
	}

  
  
  
  }
  else // super-sequence Method
  {
    set<int,less<int>> LastRNN=SetRNN(DIMENSION,s); // get full length RNN
	
    
	int FirstElement=*(s.begin());
	double label=labelTraining[FirstElement];
    int labelIndex=0;
	if (label==-1)
	   {labelIndex=1;}
	else
	   {labelIndex=label-1;}

	// compute the support of the sequence
	int Support=s.size()+LastRNN.size();
    set<int,less<int>> PreviousRNN;
   
	if (Support>=classSupport[labelIndex])
	{
		for (int le=startFrom-1;le>=1;le--)
		  {
		     if (NNConsistent(le,s)==0)
			 {
			   MPL=le+1;
			   break;
			 }
			 //else
			 //{
			 // 
			 // 
			 // 
			 // 
			 // 
			 //

    //         PreviousRNN=SetRNN(le,s);

    //       



			 //if ( LastRNN==PreviousRNN)
			 //{
				// 
				//    PreviousRNN.clear();
			 //     
				// 
				// 
			 // 
			 //}
			 //else
			 //{ 
				// /*printSet(s);
    //             printSet(PreviousUsefulRNN);
    //             cout<<"Support="<<support<<"\n";*/
				// MPL=le+1;
    //             break;
			 //
			 //
			 //}

		  //
			 //}// end of else
		  
		  
		  }// end of for
	   
	   
	   
	   
	   
	}
	else
	{
	  MPL=DIMENSION;
	}

  
  
  
  }
      














  return MPL;
  



}







int getMPLTest(set<int,less<int>>& s)
{ 
    
  int MPL=DIMENSION;

  set<int, less<int> > ::iterator i;
  if (s.size()==1) // simple RNN Method
  {
    set<int,less<int>> LastRNN=SetRNN(DIMENSION,s); // get full length RNN
    
	
             
	
    
	int FirstElement=*(s.begin());
	double label=labelTraining[FirstElement];
    int labelIndex=0;
	if (label==-1)
	   {labelIndex=1;}
	else
	   {labelIndex=label-1;}


	
	// compute the support of the sequence
	int Support=s.size()+LastRNN.size();
	

	if (Support>=classSupport[labelIndex])
	{
	   if (LastRNN.size()>0)
	   {
           set<int,less<int>> PreviousRNN;
	      for (int le=DIMENSION-1;le>=1;le--)
		  {
		     

             PreviousRNN=SetRNN(le,s);
			  



			 if (  LastRNN==PreviousRNN )
			 {
				 

			         PreviousRNN.clear();
			       
				 
				
			  
			 }
			 else
			 { 
                
				 
				 MPL=le+1;
                 break;
			 
			 
			 }

		  
		  
		  
		  
		  }// end of for
	   
	   
	   
	   
	   }
	   else
	   {MPL=DIMENSION;}
	}
	else
	{
	  MPL=DIMENSION;
	}

  
  
  
  }
  else // super-sequence Method
  {
    set<int,less<int>> LastRNN=SetRNN(DIMENSION,s); // get full length RNN
	
    
	int FirstElement=*(s.begin());
	double label=labelTraining[FirstElement];
    int labelIndex=0;
	if (label==-1)
	   {labelIndex=1;}
	else
	   {labelIndex=label-1;}


	
    
	// compute the support of the sequence
	int Support=s.size()+LastRNN.size();

	if (Support>=classSupport[labelIndex])
	{
       
        set<int,less<int>> PreviousRNN;
		for (int le=DIMENSION-1;le>=1;le--)
		  {
		     if (NNConsistent(le,s)==0)
			 {
			   MPL=le+1;
			   break;
			 }
			 //else
			 //{
			 // 
			 // 
			 // 
			 // 
			 // 
			

    //         PreviousRNN=SetRNN(le,s);

    //       
			 //if ( LastRNN==PreviousRNN)
			 //{
				// 
				//   
			 //        PreviousRNN.clear();
			 //    
				// 
				// 
			 // 
			 //}
			 //else
			 //{ 
				// 
				// MPL=le+1;
    //             break;
			 //
			 //
			 //}

		  //
			 //}// end of else
		  
		  
		  }// end of for
	   
	   
	   
	   
	   
	}
	else
	{
	  MPL=DIMENSION;
	}

  
  
  
  }
      














  return MPL;
  
  
  }





  int updateMPLTest(set<int,less<int>> s)
{
 
      int length= getMPLTest(s);
    

	  set<int, less<int> > ::iterator jj;
      for (jj=s.begin(); jj!=s.end(); jj++)
	  {
	        if (PredictionPrefix[*jj]>length)
			{
			    PredictionPrefix[*jj]=length;
			
			}
	  
	  
	  
	
   
       }

  return length;


}


double LeaveOneOut(int le)
{
  double accuracy;
  int correctlyClassified=0;

  for (int i=0;i<ROWTRAINING;i++)
  {
  
    int NN= TrainingIndex[i][le-1];
    if (labelTraining[NN]==labelTraining[i])
    {
      correctlyClassified++;
    }
   
   
  
  
  }

   return accuracy=double(correctlyClassified)/ROWTRAINING;
}

int FixedPrefix()
{

  double Accuracy=LeaveOneOut(DIMENSION);
  int le;
  for (le=DIMENSION-1;le>1;le--)
  {
     double temp=LeaveOneOut(le);

     if(temp<Accuracy)
     {break;}
  
  
  }
  return le+1;

}




double NaiveSolution(int prefix, int option)
{

  LoadData(testingFileName, testing,labelTesting, ROWTESTING);
  
  //LoadData(trainingFileName, training,labelTraining, ROWTRAINING);

  clock_t t1, t2;

  t1 = clock();
  
 // cout<<(double)t1;




  int FP=0;
  int TP=0;
  int TC=0;
  int FC=0;
  for (int i=0;i<ROWTESTING;i++)
  {
     
	      int tempNN=findNN(i,prefix);

         // cout<<"\n instance "<<i<<" 's NN is "<<tempNN<<" at length "<<j;

		 
		    
			predictedLabel[i]=labelTraining[tempNN];
			 if (predictedLabel[i]==+1 &&labelTesting[i]==-1)
            {
             FP++;
            }

           if (predictedLabel[i]==1 &&labelTesting[i]==1)
             {
             TP++;
             }
            

           if(labelTesting[i]==1)
            { TC++;}
           else
            { FC++;}

			if (predictedLabel[i]==labelTesting[i])
			{correct++;}

			
		
  
  
  
  }

  t2 = clock();
 // cout<<"\n finish time: "<<(double)t2<<endl;

  classificationTime = (double)(t2 - t1)/CLOCKS_PER_SEC  ;
  if (option==1)
  {
      return double(correct)/ROWTESTING;
  }
  else
      return  (double)FP/FC;
  
}





































double NNclassification(int prefix, int option)
{

  LoadData(testingFileName, testing,labelTesting, ROWTESTING);
  
  

  clock_t t1, t2;

  t1 = clock();
  
 // cout<<(double)t1;


  correct=0;

  int FP=0;
  int TP=0;
  int TC=0;
  int FC=0;
  for (int i=0;i<ROWTESTING;i++)
  {
     
	      int tempNN=findNN(i,prefix);

         // cout<<"\n instance "<<i<<" 's NN is "<<tempNN<<" at length "<<j;

		 
		    
			predictedLabel[i]=labelTraining[tempNN];
			 if (predictedLabel[i]==+1 &&labelTesting[i]==-1)
            {
             FP++;
            }

           if (predictedLabel[i]==1 &&labelTesting[i]==1)
             {
             TP++;
             }
            

           if(labelTesting[i]==1)
            { TC++;}
           else
            { FC++;}

			if (predictedLabel[i]==labelTesting[i])
			{
                
                
                correct++;
            }
            else
            {
            
              //  cout<<"\n instance "<<i <<"class "<<labelTesting[i]<<" is miss classified by its  NN is "<<tempNN+1<<" at length "<<prefix;
            
            }

			
		
  
  
  
  }

  t2 = clock();
 // cout<<"\n finish time: "<<(double)t2<<endl;

  classificationTime = (double)(t2 - t1)/CLOCKS_PER_SEC  ;
  if (option==1)
  {
      return double(correct)/ROWTESTING;
  }
  else
      return  (double)FP/FC;
  
}


