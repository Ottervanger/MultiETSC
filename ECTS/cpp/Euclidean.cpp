#include "Euclidean.h"

double Euclidean(double * a, double * b, int length) {
    double ret = 0;
    for (int i = 0; i < length; i++) {
        double dist = a[i] - b[i];
        ret += dist * dist;
    }
    // since we only use the distance to order 
    // we can leave out taking the root
    return ret;
}
