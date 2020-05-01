


#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "math.h"
#include <set>
#include <algorithm>
#include <vector>
#include <time.h>



#include "DataSetInformation.h"
#include "Euclidean.h"
#include "minValue.h"
#include "find.h"










using namespace std;


// ALgorithm parameters: minimal support
double MinimalSupport=0;

// global variable
double training[ROWTRAINING][DIMENSION]={0}; // training data set
double labelTraining[ROWTRAINING]={0}; // training data class labels
double testing [ROWTESTING][DIMENSION]={0}; //  testing data set
double labelTesting[ROWTESTING]={0}; // testing data class labels
double predictedLabel[ROWTESTING]={0}; // predicted label by the classifier
int predictedLength[ROWTESTING]={0};// predicted length by the classifier
int  TrainingIndex[ROWTRAINING][DIMENSION]={0};//  store the 1NN for each space, no ranking tie
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
void testUnion();
int getMPL(set<int,less<int>>& s);
int getMPL(set<int,less<int>>& s,int la, int lb);
void getMNCS(int onenode,double label, set<int,less<int>>& PreMNCS);
void updateMPL(vector<set<int,less<int>>> & List);
bool sharePrefix(   set<int, less<int> >& s1, set<int, less<int> >& s2, set<int, less<int> >& s3, int level        );
void printSet(set<int, less<int> > & s);
void printSetList(vector<set<int, less<int> >> & v) ;
int findMin( int data[], int len);
void classification();
double  mean(int data[], int len);
void report();





int main ()
{
  
	// load training data
	LoadData(trainingFileName, training,labelTraining, ROWTRAINING);

	////output class labels
	//for ( int row=0; row < ROWTRAINING; row++)
	//	cout<<labelTraining[row]<<"\n ";

	//// output data

	//for ( int row=0; row < 1; row++)
	//{
	//	for (int col=0; col<DIMENSION; col++)
	//	{
	//		cout<<training[row][col]<<" ";
	//	}
	//	cout<<"\n";
	//}

    // load testing data
	
   
	////output class labes
	//for ( int row=0; row < ROWTRAINING; row++)
	//	cout<<labelTesting[row]<<"\n ";

	//// output data

	//for ( int row=0; row < 1; row++)
	//{
	//	for (int col=0; col<DIMENSION; col++)
	//	{
	//		cout<<testing[row][col]<<" ";
	//	}
	//	cout<<"\n";
	//}



	//for (int i=0;i< NofClasses;i++)
	//{
	//   cout<< Classes[i]<<" ";
	//
	//
	//}


	getClassDis();

	/*for (int i=0;i<NofClasses;i++)
	{cout<<Classes[i]<<" :"<<classDistri[i]<<" Minimal Support: "<<classSupport[i]<<"\n";}*/
    
   // LoadDisArray(DisArrayFileName, DisArray  );
     
	/*for ( int row=0; row < ROWTRAINING; row++)
	{
		for (int col=0; col<ROWTRAINING; col++)
		{
			cout<<DisArray[row][col]<<" ";
		}
		cout<<"\n";
	}*/

   LoadTrIndex(trainingIndexFileName, TrainingIndex  );
   /*for ( int row=0; row < ROWTRAINING; row++)
	{
		for (int col=0; col<DIMENSION; col++)
		{
			cout<<TrainingIndex[row][col]<<" ";
		}
		cout<<"\n";
	}*/

   // compute the full length classification status, 0 incorrect, 1 correct

   clock_t t3, t4;

  t3 = clock();
   for (int i=0;i<ROWTRAINING;i++)
   {
       
	   int NNofi=TrainingIndex[i][DIMENSION-1];
	  

       if (labelTraining[i]==labelTraining[NNofi])
	   {
	      FullLengthClassificationStatus[i]=1;
	   }

	 

   }

   // intialize the prediction prefix

   for (int i=0;i<ROWTRAINING;i++)
   {
	   PredictionPrefix[i]=DIMENSION;
	  
   }
   
  

   // testUnion();

   // simple RNN method
   for (int i=0;i<ROWTRAINING;i++)
   {
	   
	   set<int, less<int> >  s;
	   s.insert(i);
	   PredictionPrefix[i]=getMPL(s);
	  
   }

   // for (int i=0;i<ROWTRAINING;i++)
   //{
	  // 
	  // 
	  // cout<<PredictionPrefix[i]<<" ";
	  //
   //}
   

   // find the pure bi-roots

   vector<set<int, less<int> >> biRoots; // is a vector of sets
   
   bool UsedInBiRoots[ROWTRAINING]={0}; // 0 not used, 1 used
    
   for (int i=0;i<ROWTRAINING;i++)
   {
      if (UsedInBiRoots[i]==0)
	  {
	    int NNofi=TrainingIndex[i][DIMENSION-1];
		if (TrainingIndex[NNofi][DIMENSION-1]==i && labelTraining[i]==labelTraining[NNofi])
		{
           set<int, less<int> > temp;
		   temp.insert(i);
		   temp.insert(NNofi);

		   biRoots.push_back(temp);

           UsedInBiRoots[i]=1;
		   UsedInBiRoots[NNofi]=1;
		}
	  
	  
	  }
   
   
   
   } // end for

    
   /*for (int i=0; i<biRoots.size(); i++)
   {
     set<int, less<int> > temp=biRoots[i];

     set<int, less<int> > ::iterator j;

	 cout << " bi-root ";
     for (j=temp.begin(); j!=temp.end(); j++) cout << *j << ' ';
     cout << endl;


   
   
   
   }
   */
    // get all the pure MNCS
   for (int k=0;k<biRoots.size();k++)
   {
   
       NNSetList.push_back(biRoots[k]);
	   double biRootlabel=labelTraining[*(biRoots[k].begin())];
	   NNSetListClassMap.push_back(biRootlabel);
	   set<int, less<int> > NextlayerofCom=SetRNN(DIMENSION, biRoots[k]);
	   set<int, less<int> > ::iterator j;

	  for (j=NextlayerofCom.begin();j!=NextlayerofCom.end();j++)
	  {
	  
	     int node=*j;

		 getMNCS(node,biRootlabel,biRoots[k]);


	  
	  
	  }

   
   
   
   }
   

     
  // for (int i=0; i<NNSetList.size(); i++)
  // {
  //   set<int, less<int> > temp=NNSetList[i];

  //   set<int, less<int> > ::iterator j;

	 //cout << " MNCS ";
  //   for (j=temp.begin(); j!=temp.end(); j++) cout << *j << ' ';
  //   cout << endl;


	 //cout<<"label: "<<NNSetListClassMap[i];
  // 
  // 
  // }
  // 
	 
   // comput the prediction length for MNCS
  updateMPL(NNSetList);
   
  
  

  //  Start to build the second leve, APRIOR

   for (int i=0;i<NofClasses;i++)
   {
   
	   double templabel=Classes[i];
       
	  

       vector<set<int, less<int> >>  NNSetListMinimal, NNSetListAll;
	   //NNSetListMinimal: a list of MNCS in classes[i]
	   // NNSetListALl: store all the MNCS in classes[i] from the second level

	   for (int j=0;j<NNSetListClassMap.size();j++)
	   {
	     
	      if (NNSetListClassMap[j]==templabel)
		  {
		  
      		     NNSetListMinimal.push_back(NNSetList[j]);
		  
		  }
	   }
	
	   int latticeLevel=1;

	   vector<set<int, less<int> >>  currentIndex, nextIndex;
       vector<int> currentIndexLength=NNSetListLength;
       vector<int> nextIndexLength;

	   for(int j=0;j<NNSetListMinimal.size();j++)
	   {
	   
	        set<int, less<int> > temp;
			temp.insert(j);
			currentIndex.push_back(temp);
	   
	   
	   }
	  // cout<<"NNSetListMinimal";
	   //printSetList(NNSetListMinimal);

       
	   while (true)
	   {
	      int size=currentIndex.size();
		  cout<<"Number of Sets: "<<size<<endl;
		  if (size==1)
		  {break;}
		  else
		  {
		    latticeLevel=latticeLevel+1;
			cout<<"Lattice Level: "<<latticeLevel<<endl;
            
			 for (int i1=0;i1<size-1;i1++)
			 {
			 
			    for (int j1=i1+1;j1<size;j1++)
				{
				    // check if two indexes can be merged
                     set<int, less<int> > unionindex;
                     bool result=sharePrefix(  currentIndex[i1] , currentIndex[j1] , unionindex, latticeLevel);  
                     


					 if (result==1)// can be merged
					 {
					  

						 // compute the first sequence set
                         set<int, less<int> > tempSet1;
						 set<int, less<int> > indexListSet1=currentIndex[i1];
						 set<int, less<int> > ::iterator it;

						 for (it=indexListSet1.begin();it!=indexListSet1.end();it++)
						 {
						    int indexofMNCS=(*it);
                            set<int, less<int> > tempMNCS=NNSetListMinimal[indexofMNCS];

							set<int, less<int> > ::iterator it1;
							for (it1=tempMNCS.begin();it1!=tempMNCS.end();it1++)
							{
							    tempSet1.insert(*it1);
							
							}


						 
						 
						 }
                          

						  // compute the second sequence set
                         set<int, less<int> > tempSet2;
						 set<int, less<int> > indexListSet2=currentIndex[j1];
						 

						 for (it=indexListSet2.begin();it!=indexListSet2.end();it++)
						 {
						    int indexofMNCS=(*it);
                            set<int, less<int> > tempMNCS=NNSetListMinimal[indexofMNCS];

							set<int, less<int> > ::iterator it2;
							for (it2=tempMNCS.begin();it2!=tempMNCS.end();it2++)
							{
							    tempSet2.insert(*it2);
							
							}

                        
					    }
					     
						 // check if one is the other's subset
                       bool issubset=0;
					   if (tempSet1.size()>=tempSet2.size())
					   { issubset=includes(tempSet1.begin(),tempSet1.end(),tempSet2.begin(),tempSet2.end()); }
					   else
					   { issubset=includes(tempSet2.begin(),tempSet2.end(),tempSet1.begin(),tempSet1.end());}
                       
					   
					   if (issubset==0)// not subset of each other
					   {
					     nextIndex.push_back(unionindex);

                         set<int, less<int> > unionset;

						 set_union(tempSet1.begin(), tempSet1.end(), tempSet2.begin(),tempSet2.end(),
                            insert_iterator<set<int, less<int> >> (unionset,unionset.begin()) );
						 //cout<<"\n set: ";
						 //printSet(unionset);
                         
                         int tempLength=getMPL(unionset,currentIndexLength[i1] ,currentIndexLength[j1] );
                       // update the length
                        
                              set<int, less<int> > ::iterator jj;
                              for (jj=unionset.begin(); jj!=unionset.end(); jj++)
	                          {
	                                if (PredictionPrefix[*jj]>tempLength)
			                        {
			                            PredictionPrefix[*jj]=tempLength;
                        			
			                        }
                        	  
                        	  
                        	  
	                          }
                              
                              nextIndexLength.push_back(tempLength);




                        


						 NNSetListAll.push_back(unionset);
					   
					    
					   
					   
					   }
					   
					   
					   
					   
					   
					   
					 
					 
					 
					 
					 } // end can be merged or not


				
				
				
				}// end for
			 
			 
			 
			 
			 }// end for
		  
		  
		  
		  
		  
		  }// end of size >1
         
         

          currentIndex.clear();
		  for (int kk=0;kk<nextIndex.size();kk++)
		  {currentIndex.push_back(nextIndex[kk]);}
		  nextIndex.clear();

           currentIndexLength.clear();
		  for (int kk=0;kk<nextIndexLength.size();kk++)
		  {currentIndexLength.push_back(nextIndexLength[kk]);}
		  nextIndexLength.clear();




	      
		  if(currentIndex.size()<=1)
		  {   cout<<"\nI am here";
			  break;}
	   




	   
	   } // end while


	   //  update the prediction prefix
    // cout<<"\nstart to compute length";
     
      
	// printSetList(NNSetListAll);
	// cout<<"\nSize :"<<NNSetListAll.size();

	// updateMPL(NNSetListAll); // will be updated in inside of the loop
    
	 






   }// end for classes

    t4 = clock();
    trainingTime = (double)(t4 - t3)/CLOCKS_PER_SEC  ;
   cout<<"\PredictionPrefix\n";
   for (int i=0;i<ROWTRAINING;i++)
   {
    cout<<PredictionPrefix[i]<<" ";
   
   
   }
   
    classification();
	report();

}// end main

   


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

void testUnion()
{
	 // test how to use set

 //   set<int, less<int> >  s, s2, s4;
 //   set<int, less<int> > ::iterator i;
	//

 //  
 //   s.insert(19);
	//s.insert(26);
	//s.insert(21);

	//
 //   
	//

	//s2.insert(5);
	//s2.insert(12);

 //   cout << "set contains the elements: ";
 //   for (i=s.begin(); i!=s.end(); i++) cout << *i << ' ';
 //   cout << endl;

	//int result=getMPL(s);
	//cout<<"MPL of s: "<<result;

	//cout << " set2 contains the elements: ";
 //   for (i=s2.begin(); i!=s2.end(); i++) cout << *i << ' ';
 //   cout << endl;

	//cout<<"they are equal: "<<(s==s2);
	//s2.clear();
	//cout << " after clear, set2 contains the elements: ";
 //   for (i=s2.begin(); i!=s2.end(); i++) cout << *i << ' ';
 //   cout << endl;


	//bool test = includes(s.begin(),s.end(),s2.begin(),s2.end()); // check if s2 is s1's subset, return bool
 //   cout << "s2 is subset of s is " << test << endl;


 //     
	// set_union(s.begin(), s.end(), s2.begin(), s2.end(),
 //           insert_iterator<set<int, less<int> >> (s4,s4.begin()) );
 //
 //   cout << "the union is ";
 //   for (i=s4.begin(); i!=s4.end(); i++) cout << *i << ' ';
 //   cout << endl;

	//cout<<"The size of s4 is :"<<s4.size();
 //   
	//
	//cout<<"the first element of s4 is: "<<*(s4.begin());




 
 
 }

int getMPL(set<int,less<int>>& s)
{ int MPL=DIMENSION;
  set<int, less<int> > ::iterator i;
  if (s.size()==1) // simple RNN Method
  {
    set<int,less<int>> CurrentRNN=SetRNN(DIMENSION,s); // get full length RNN

	
             
	
    
	int FirstElement=*(s.begin());
	double label=labelTraining[FirstElement];
    int labelIndex=0;
	if (label==-1)
	   {labelIndex=1;}
	else
	   {labelIndex=label-1;}


	// get the intersect(RNN(S,L), T')=RNN(S,L) in the same class as S. 
    set<int,less<int>> CurrentUsefulRNN;
    for (i=CurrentRNN.begin(); i!=CurrentRNN.end(); i++)
	{
	    int element=*i;

		if ( FullLengthClassificationStatus[element]==1)
		{
		
		   CurrentUsefulRNN.insert(element);
		
		}
	
	
	
	}
	// compute the support of the sequence
	int Support=s.size()+CurrentUsefulRNN.size();
	

	if (Support>=classSupport[labelIndex])
	{
	   if (CurrentUsefulRNN.size()>0)
	   {
	      for (int le=DIMENSION-1;le>=1;le--)
		  {
		     set<int,less<int>> PreviousRNN;

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
            


			 if (  CurrentUsefulRNN==PreviousUsefulRNN )
			 {
				 
				    CurrentUsefulRNN=PreviousUsefulRNN;
			        PreviousUsefulRNN.clear();
				 
				
			  
			 }
			 else
			 {  
                 /* printSet(s);
                 printSet(PreviousUsefulRNN);
                 cout<<"Support="<<classSupport[labelIndex]<<"\n";*/
				 
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
    set<int,less<int>> CurrentRNN=SetRNN(DIMENSION,s); // get full length RNN
	
    
	int FirstElement=*(s.begin());
	double label=labelTraining[FirstElement];
    int labelIndex=0;
	if (label==-1)
	   {labelIndex=1;}
	else
	   {labelIndex=label-1;}


	// get the intersect(RNN(S,L), T')=RNN(S,L) in the same class as S. 
    set<int,less<int>> CurrentUsefulRNN;
    for (i=CurrentRNN.begin(); i!=CurrentRNN.end(); i++)
	{
	    int element=*i;

		if (FullLengthClassificationStatus[element]==1)
		{
		
		   CurrentUsefulRNN.insert(element);
		
		}
	
	
	
	}
	// compute the support of the sequence
	int Support=s.size()+CurrentUsefulRNN.size();

	if (Support>=classSupport[labelIndex])
	{
		for (int le=DIMENSION-1;le>=1;le--)
		  {
		     if (NNConsistent(le,s)==0)
			 {
			    /*printSet(s);
                 printSet(CurrentUsefulRNN);
                 cout<<"Support="<<classSupport[labelIndex]<<"\n";*/
                MPL=le+1;
			   break;
			 }
			 else
			 {
			  
			  
			  
			  
			  
			 set<int,less<int>> PreviousRNN;

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

			 if ( CurrentUsefulRNN==PreviousUsefulRNN)
			 {
				 
				    CurrentUsefulRNN=PreviousUsefulRNN;
			        PreviousUsefulRNN.erase(PreviousUsefulRNN.begin(), PreviousUsefulRNN.end());
				 
				 
			  
			 }
			 else
			 { 
				/*  printSet(s);
                 printSet(PreviousUsefulRNN);
                 cout<<"Support="<<classSupport[labelIndex]<<"\n";*/
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
      









 //if (MPL<50)

 //{cout<<"MPL is "<<MPL<<endl;


 //char a;
 //cout<<"\n continue..\n";
 //cin>>a;}

  return MPL;
  
  
  }




void getMNCS(int onenode,double label, set<int,less<int>>& PreMNCS)
{

    if (labelTraining[onenode]!=label)
	{   return;  }
	else
	{  
        set<int, less<int> > currentMNCS;
	    set<int, less<int> > ::iterator i;
	    for (i=PreMNCS.begin(); i!=PreMNCS.end(); i++)
		{
		  currentMNCS.insert(*i);
		
		
		}
		currentMNCS.insert(onenode);
		NNSetList.push_back(currentMNCS);
		NNSetListClassMap.push_back(label);
        
		set<int, less<int> > temp,RNNofNode;
		temp.insert(onenode);
		RNNofNode=SetRNN(DIMENSION, temp);


		for (i=RNNofNode.begin(); i!=RNNofNode.end(); i++)
		{
		   int element=*i;

		   if (labelTraining[element]==label)
		   {
		       getMNCS(element,label,currentMNCS);
		   
		   
		   }
		
		
		}


	
	   
	
	
	}




}
void updateMPL(vector<set<int,less<int>>> & List)
{
  for (int i=0;i<List.size();i++)
   {
       
      set<int, less<int> > temp=List[i];

	  int MPLofMNCS=getMPL(temp);

      NNSetListLength.push_back(MPLofMNCS);

    

	  set<int, less<int> > ::iterator jj;
      for (jj=temp.begin(); jj!=temp.end(); jj++)
	  {
	        if (PredictionPrefix[*jj]>MPLofMNCS)
			{
			    PredictionPrefix[*jj]=MPLofMNCS;
			
			}
	  
	  
	  
	  }
   
   
   
   
   }




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




void report()
  {
	 
      cout<<"\n Report\n--Exact Algorithm";
	  cout<< "The averaged predicted length of testing data is "<<mean(predictedLength,ROWTESTING)<<"\n";
	  cout<< "The averaged prediction prefix of training data is "<<mean(PredictionPrefix,ROWTRAINING)<<"\n";
	  cout<<" The accuracy is "<< double(correct)/ROWTESTING<<"\n";
	  cout<<" The error rate is"<<1-double(correct)/ROWTESTING<<"\n";
	  cout<<"Average time of classifying one instance "<<classificationTime/ROWTESTING<<"seconds \n";
	  cout<<"Training Time "<<trainingTime<<"second \n";

	  ofstream outputFile(ResultfileName, ios::out|ios::app);

      outputFile<<"\n Report (Exact Algorithm) Minimal support="<<MinimalSupport<<"\n";
	  outputFile<< "The averaged predicted length of testing data is "<<mean(predictedLength,ROWTESTING)<<"\n";
      outputFile<< "The averaged prediction prefix of training data is "<<mean(PredictionPrefix,ROWTRAINING)<<"\n";
	  outputFile<<" The accuracy is "<< double(correct)/ROWTESTING<<"\n";
	  outputFile<<" The error rate is"<<1-double(correct)/ROWTESTING<<"\n";
	 
	  outputFile<<"Average time of classifying one instance "<<classificationTime/ROWTESTING<<" seconds\n";
	  outputFile<<"Training Time "<<trainingTime<<" seconds\n";

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
    set<int,less<int>> CurrentRNN=SetRNN(DIMENSION,s); // get full length RNN

	
             
	
    
	int FirstElement=*(s.begin());
	double label=labelTraining[FirstElement];
    int labelIndex=0;
	if (label==-1)
	   {labelIndex=1;}
	else
	   {labelIndex=label-1;}


	// get the intersect(RNN(S,L), T')=RNN(S,L) in the same class as S. 
    set<int,less<int>> CurrentUsefulRNN;
    for (i=CurrentRNN.begin(); i!=CurrentRNN.end(); i++)
	{
	    int element=*i;

		if ( FullLengthClassificationStatus[element]==1)
		{
		
		   CurrentUsefulRNN.insert(element);
		
		}
	
	
	
	}
	// compute the support of the sequence
	int Support=s.size()+CurrentUsefulRNN.size();
	

	if (Support>=classSupport[labelIndex])
	{
	   if (CurrentUsefulRNN.size()>0)
	   {
	      for (int le=DIMENSION-1;le>=1;le--)
		  {
		     set<int,less<int>> PreviousRNN;

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
            


			 if (  CurrentUsefulRNN==PreviousUsefulRNN )
			 {
				 
				    CurrentUsefulRNN=PreviousUsefulRNN;
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
    set<int,less<int>> CurrentRNN=SetRNN(DIMENSION,s); // get full length RNN
	
    
	int FirstElement=*(s.begin());
	double label=labelTraining[FirstElement];
    int labelIndex=0;
	if (label==-1)
	   {labelIndex=1;}
	else
	   {labelIndex=label-1;}


	// get the intersect(RNN(S,L), T')=RNN(S,L) in the same class as S. 
    set<int,less<int>> CurrentUsefulRNN;
    for (i=CurrentRNN.begin(); i!=CurrentRNN.end(); i++)
	{
	    int element=*i;

		if (FullLengthClassificationStatus[element]==1)
		{
		
		   CurrentUsefulRNN.insert(element);
		
		}
	
	
	
	}
	// compute the support of the sequence
	int Support=s.size()+CurrentUsefulRNN.size();

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
			  
			  
			  
			  
			  
			 set<int,less<int>> PreviousRNN;

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

			 if ( CurrentUsefulRNN==PreviousUsefulRNN)
			 {
				 
				    CurrentUsefulRNN=PreviousUsefulRNN;
			        PreviousUsefulRNN.erase(PreviousUsefulRNN.begin(), PreviousUsefulRNN.end());
				 
				 
			  
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