#include <cmath>
#include "util.h"

namespace util {

void quicksort(double a[], int start, int end) {
    if (start>=end)
        return;
    double pivot=a[end];
     
    int storedIndex = start;
    for (int i=start;i<end;i++) {
        if (a[i] < pivot) {
            double temp=a[storedIndex]; 
            a[storedIndex] = a[i]; 
            a[i] = temp; 
            storedIndex++;
        }
    }
    double temp=a[storedIndex]; 
    a[storedIndex]=a[end]; 
    a[end]=temp; 
    quicksort(a, start, storedIndex-1); 
    quicksort(a, storedIndex+1, end); 
}


double euclidean(double * a, double * b, int length) {
  double ret = 0;
  for (int i=0; i<length;i++) {
    double dist = a[i]-b[i];
    ret += dist * dist;
  }
  return ret > 0 ? sqrt(ret) : 0;
}

} // namespace util
