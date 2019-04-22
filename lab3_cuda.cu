#include "lab3_cuda.h"
#include <cmath>
#include <malloc.h>
#include <math.h>
#include <algorithm>

using namespace std;

///////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////Helper Functions///////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

double compare_matrices(double* A, double* B, int M, int N){
    double diff = 0; int p, q;
    for(int i = 0; i < M; i++)
        for(int j = 0; j < N; j++){
            if(fabs(fabs(A[i*N+j]) - fabs(B[i*N+j])) > diff){
                diff = fabs(fabs(A[i*N+j]) - fabs(B[i*N+j]));
                p = i; q = j; 
            }
        }
    return diff;
}

void reverse_array(double* a, int N){
    double* temp = new double[N];
    for(int i = 0; i < N; i++)
        temp[i] = a[i];
    for(int i = 0; i < N; i++)
        a[i] = temp[N-i-1];
}

void copy_matrix(double* to, double* from, int n, int m){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++)  
            to[i*n+j] = from[i*n+j];
    }
}

double* null_matrix(int m, int n){
    double* A;
    cudaMallocManaged((void**)&A, sizeof(double)*m*n);
    return A;
}

double* empty_matrix(int m, int n){
    double* A;
    cudaMallocManaged((void**)&A, sizeof(double)*m*n);
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++)  
            A[i*n+j] = 0;
    }
    return A;
}

double** empty_matrix_2d(int m, int n){
    double** A = new double*[m];
    for(int i = 0; i < m; i++){
        A[i] = new double[n];
        for(int j = 0; j < n; j++)  
            A[i][j] = 0;
    }
    return A;
}

double** null_matrix_2d(int m, int n){
    double** A = new double*[m];
    for(int i = 0; i < m; i++){
        A[i] = new double[n];
    }
    return A;
}

void copy_matrix_to2d(double** to, double* from, int m, int n){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++)  
            to[i][j] = from[i*n+j];
    }
}

void copy_matrix_from2d(double* to, double** from, int m, int n){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++)  
            to[i*n+j] = from[i][j];
    }
}

double* diagonal_matrix(int n){
    double* A;
    cudaMallocManaged((void**)&A, sizeof(double)*n*n);
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++)  
            A[i*n+j] = (i == j) ? 1 : 0;
    }
    return A;
}

void matrix_multiply(double* res, double* A, double* B, int N, int M, int N1){
    // Matrices shapes: A = NxM, B = MxN1, res = NxN1
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N1; j++){
            res[i*N1+j] = 0;
            for(int k = 0; k < M; k++)
                res[i*N1+j] += A[i*M+k] * B[k*N1+j];
        }
    }
}

void print_matrix(double* A, int M, int N, char* name){
    printf("\nMatrix %s\n", name);
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            printf("%f ", A[i*N+j]);
        }
        printf("\n");
    }
}

void print_matrix_2d(double** A, int M, int N, char* name){
    printf("\nMatrix %s\n", name);
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            printf("%f ", A[i][j]);
        }
        printf("\n");
    }
}


///////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////Jacobi////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

#define TOLERANCE 0.001
#define JACOBI_UPDATE_TOLERANCE 0.001

double **S; //Symmetric matrix (input)
double  *e; //eigenvalues
double **E; //eigenvectors
int  *ind;
bool *changed;
int  state;
int  N;

double** mat1;
double** mat2;
double** mat3;
double ek_prev;
int m;

void mat_mul(double** C, double** A, int Am, int An, 
                 double** B, int Bm, int Bn){
    for (int i=0; i<Am; i++){
        for (int j=0; j<Bn; j++){
            C[i][j] = 0;
            for (int k=0; k<An; k++){
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int maxind(int k) {
    m = k+1;

    for (int i = k+2; i < N; i++){
        if (fabs(S[k][i]) > fabs(S[k][m])){
            m = i;
        }
    }

    return m;
}

void update(int k, double t) {
    ek_prev = e[k];
    e[k] = ek_prev + t;

    if (e[k] < 0) e[k] = 0;

    if (changed[k] && fabs(ek_prev - e[k]) < JACOBI_UPDATE_TOLERANCE) {
        changed[k] = false;
        state = state - 1;
    }
    else if ((! changed[k]) && fabs(ek_prev - e[k]) > JACOBI_UPDATE_TOLERANCE) {
        changed[k] = true;
        state = state + 1;
    }
}

void rotate(int k, int l, int i, int j, double c, double s,
            bool eigenvectors){

    mat1[0][0] = c; mat1[0][1] = -s;
    mat1[1][0] = s; mat1[1][1] = c;

    if (eigenvectors){
        mat2[0][0] = E[i][k];
        mat2[1][0] = E[i][l];
    }
    else {
        mat2[0][0] = S[k][l];
        mat2[1][0] = S[i][j];
    }

    mat_mul(mat3, mat1, 2, 2, mat2, 2, 1);

    if (eigenvectors){
        E[i][k] = mat3[0][0];
        E[i][l] = mat3[1][0];
    }
    else{
        S[k][l] = mat3[0][0];
        S[i][j] = mat3[1][0];
    }
}

void init_jacobi() {
    E = (double**)malloc(__SIZEOF_POINTER__*N);
    for (int i=0; i<N; i++){
        E[i] = (double*)malloc(__SIZEOF_DOUBLE__*N);
        for (int j=0; j<N; j++){
            E[i][j] = 0;
        }
        E[i][i] = 1;
    }

    state = N;

    mat1 = (double**)malloc(__SIZEOF_POINTER__*2);
    mat1[0] = (double*)malloc(__SIZEOF_DOUBLE__*2);
    mat1[1] = (double*)malloc(__SIZEOF_DOUBLE__*2);
    mat2 = (double**)malloc(__SIZEOF_POINTER__*2);
    mat2[0] = (double*)malloc(__SIZEOF_DOUBLE__*1);
    mat2[1] = (double*)malloc(__SIZEOF_DOUBLE__*1);
    mat3 = (double**)malloc(__SIZEOF_POINTER__*2);
    mat3[0] = (double*)malloc(__SIZEOF_DOUBLE__*1);
    mat3[1] = (double*)malloc(__SIZEOF_DOUBLE__*1);

    e = (double*)malloc(__SIZEOF_DOUBLE__*N);
    ind = (int*)malloc(__SIZEOF_INT__*N);
    changed = (bool*)malloc(sizeof(bool)*N);

    for (int k=0; k<N; k++){
        ind[k]     = maxind(k);
        e[k]       = S[k][k];
        changed[k] = true;
    }
}

void Jacobi(double **input_matrix, int n, 
            double **eigenvalues, double ***eigenvectors) {
    N = n;
    S = input_matrix;

    init_jacobi();

    int k, l, i, m;
    double p, y, d, r, c, s, t;

    while(state != 0){
        m = 0;

        for (k=1; k<N-1; k++){
            if (fabs(S[k][ind[k]]) > fabs(S[m][ind[m]])){
                m = k;
            }
        }

        k = m;
        l = ind[m];
        p = S[k][l];
        y = (e[l] - e[k]) / 2.0;
        d = fabs(y) + sqrt(p*p + y*y);
        r = sqrt(p*p + d*d);
        c = d / r;
        s = p / r;
        t = (p*p) / d;

        if (y < 0.0) { s = -s; t = -t; }

        S[k][l] = 0.0;
        update(k, -t);
        update(l, t);

        for (i=0; i<k; i++)  { rotate(i, k, i, l, c, s, false); }
        for (i=k+1; i<l; i++){ rotate(k, i, i, l, c, s, false); }
        for (i=l+1; i<N; i++)  { rotate(k, i, l, i, c, s, false); }

        for (i=0; i<N; i++){
            rotate(k, l, i, i, c, s, true);
        }

        ind[k] = maxind(k);
        ind[l] = maxind(l);
    }

    *eigenvalues = e;
    *eigenvectors = E;
}

///////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////CUDA/////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

__global__ void mult_cuda(double* res, double* a, double* b, int N, int M, int N1)
{
    int i = blockIdx.x;
    int j = threadIdx.x;

    res[i*N1+j] = 0;
    for(int k = 0; k < M; k++)
        res[i*N1+j] += a[i*M+k] * b[k*N1+j];
}

void matrix_multiply_cuda(double* res, double* a, double* b, int N, int M, int N1){
    // Matrices shapes: A = NxM, B = MxN1, res = NxN1
    mult_cuda<<<N, N1>>>(res, a, b, N, M, N1);
    cudaDeviceSynchronize();
}

///////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////SVD//////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

void SVD_and_PCA (
    int M, 
    int N, 
    double* D, 
    double** U, 
    double** SIGMA, 
    double** V_T, 
    int *SIGMAm, 
    int *SIGMAn, 
    double** D_HAT, 
    int *K, 
    int retention) {

    *SIGMAm = N;
    *SIGMAn = M;

    // printf("Starting SVD\n");
    // Dt is D transpose = NxM
    double* Dt = empty_matrix(N, M);
    // Dc is copy of D = MxN
    double* Dc = empty_matrix(M, N);

    // DtD is Dt.D = NxN, so are Q and R
    double* DtD = empty_matrix(N, N);
    
    // Compute Dt and Dc
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            Dt[j*M+i] = D[i*N + j];
            Dc[i*N+j] = D[i*N + j];
        }
    }

    // Multiply Dt.D = NxM . MxN = NxN
    matrix_multiply_cuda(DtD, Dt, Dc, N, M, N);

    // print_matrix(DtD, N, N, "DtD\0");

    // Get Eigenvalues of DtD i.e. Q and R
    double* Ei = null_matrix(N, N);
    double* Ei_temp = null_matrix(N, N);
    double* eigenvalues;
    double** eigenvectors;

    // Convert DtD to 2d matrix for Jacobi
    double** DtDJ = null_matrix_2d(N, N);
    copy_matrix_to2d(DtDJ, DtD, N, N);
    // print_matrix_2d(DtDJ, N, N, "DtDJ\0");

    // printf("Starting jacobi\n");
    Jacobi(DtDJ, N, &eigenvalues, &eigenvectors);
    // printf("End jacobi\n");

    // Convert Eigenvectors from 2d to 1d
    copy_matrix_from2d(Ei, eigenvectors, N, N);

    // Sorting and reordering eigenvectors
    double* eigenvalues1 = new double[N];
    for(int i = 0; i < N; i++){
        eigenvalues1[i] = eigenvalues[i];
    }
    
    std::sort(eigenvalues, eigenvalues + N);
    reverse_array(eigenvalues, N);
    // for(int i = 0; i < N; i++){
    //     printf("Eigenvals = %f, \t\t %f\n", eigenvalues[i], eigenvalues1[i]);
    // }

    // Update Ei
    for(int j = 0; j < N; j++){
        int p = 0;
        // Find p i.e. index of jth max eigenvalue
        for(p = 0; p < N; p++){
            if(eigenvalues1[p] == eigenvalues[j])
                break;
        }
        // printf("p=%d, j=%d\n",p,j);
        for(int i = 0; i < N; i++){
            Ei_temp[i*N+j] = Ei[i*N+p];
        }
    }
    // print_matrix(Ei, N, N, "Ei\0");
    // print_matrix(Ei_temp, N, N, "Ei_temp\0");

    copy_matrix(Ei, Ei_temp, N, N);

    // Compute Sigma
    double* sigma = empty_matrix(M, N);
    double* sigma_inv = empty_matrix(N, M);
    double* sigma_vals = new double[N];
    for(int i = 0; i < N; i++){
        sigma_vals[i] = sqrt(eigenvalues[i]);
        sigma[i*N+i] = sqrt(eigenvalues[i]);
        sigma_inv[i*M+i] = (1.0 / sqrt(eigenvalues[i]));
    }

    *SIGMA = sigma_vals;

    double* V_temp = null_matrix(M, M);
    double* U_temp = null_matrix(N, N);

    // Compute U
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            U_temp[i*N+j] = Ei[i*N+j];
        }
    }
    
    *U = U_temp;

    double* temp = null_matrix(M, N);
    double* temp2 = null_matrix(M, M);
    matrix_multiply_cuda(temp, Dc, Ei, M, N, N);
    matrix_multiply_cuda(temp2, temp, sigma_inv, M, N, M);

    // Compute V_T
    for(int i = 0; i < M; i++)
        for(int j = 0; j < M; j++){
            V_temp[j*M+i] = temp2[i*M+j];
        }

    *V_T = V_temp;

    // Test U = M * V * Sigma-1
    // matrix_multiply_cuda(temp, U_temp, sigma, N, N, M);
    // matrix_multiply_cuda(temp2, temp, V_temp, N, M, M);

    // printf("Comparison result diff = %f\n", compare_matrices(temp2, Dt, N, M));


///////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////PCA//////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

    double ret = double(retention)/100;
    double sumeigen = 0;
    for(int i = 0; i < N; i++){
        sumeigen += sigma[i*N+i] * sigma[i*N+i];
        // printf("Sigma %d is %f\n", i, *(*SIGMA + i));
    }
    double sumret = 0; int k = 0;
    for(k = 0; k < N; k++){
        sumret += (sigma[k*N+k] * sigma[k*N+k]/ sumeigen);
        if(sumret >= ret)
            break;
    }

    *K = k+1;
    // printf("K = %d\n", *K);
    double* W = empty_matrix(N, k+1);
    for(int i = 0; i < N; i++){
        for(int j = 0; j <= k; j++)
            W[i*(k+1)+j] = U_temp[i*N+j];
    }

    // Print W
    // print_matrix(W, N, *K, "W\0");

    // printf("D-Hat:\n");
    double* DHatTemp = null_matrix(M, k+1);

    matrix_multiply_cuda(DHatTemp, Dc, W, M, N, (k+1));

    // for(int i = 0; i < M; i++){
    //     for(int j = 0; j <= k; j++){
    //         printf("%f ", DHatTemp[i*(k+1) + j]);
    //     }
    //     printf("\n");
    // }

    *D_HAT = DHatTemp;
}

