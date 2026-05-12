#include "mex.h"
#ifdef _OPENMP
  #include <omp.h>
#endif

static int get_int(const mxArray *a){
    if(!mxIsDouble(a) || mxIsComplex(a) || mxIsEmpty(a))
        mexErrMsgIdAndTxt("omp_threads_mex:arg","Expected a real scalar.");
    double v = mxGetScalar(a);
    if(!(v >= 1.0)) mexErrMsgIdAndTxt("omp_threads_mex:range","threads must be >= 1.");
    return (int)v;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
#ifdef _OPENMP
    if(nrhs==0){
        plhs[0] = mxCreateDoubleScalar((double)omp_get_max_threads());
        return;
    } else if(nrhs==1){
        int n = get_int(prhs[0]); if(n<1) n=1;
        omp_set_num_threads(n);
        if(nlhs>0) plhs[0] = mxCreateDoubleScalar((double)omp_get_max_threads());
        return;
    } else {
        mexErrMsgIdAndTxt("omp_threads_mex:arity","0 or 1 inputs only.");
    }
#else
    if(nrhs==0){
        plhs[0] = mxCreateDoubleScalar(1.0);
    } else {
        mexErrMsgIdAndTxt("omp_threads_mex:noopenmp","Built without OpenMP.");
    }
#endif
}
