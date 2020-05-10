#include "find.h"

int find(double a[], double target, int len) {
    int count = 0;
    for  (int i = 0; i < len; i++) {
        if (a[i] == target) {
            count++;
        }
    }
    return count;
}
