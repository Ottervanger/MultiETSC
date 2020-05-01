
#include "minValue.h"
double minValue (double  a[], int length)
{   
	double min=10000;
    for (int i=0;i<length;i++)
	{
	  if (a[i]>0 && min>a[i])
	  {
	     min=a[i];
	  
	  }
	
	}

  return min;

}