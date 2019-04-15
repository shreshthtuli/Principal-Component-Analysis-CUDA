#include "lab3_cuda.h"
#include <cmath>
#include <malloc.h>
#include <math.h>
#include <algorithm>


using namespace std;

///////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////Helper Functions///////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

double compare_matrices(double** A, double** B, int M, int N){
    double diff = 0; int p, q;
    for(int i = 0; i < M; i++)
        for(int j = 0; j < N; j++){
            if(fabs(fabs(A[i][j]) - fabs(B[i][j])) > diff){
                diff = fabs(fabs(A[i][j]) - fabs(B[i][j]));
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

void copy_matrix(double** to, double** from, int n, int m){
    #pragma omp parallel for
    for(int i = 0; i < m; i++){
        #pragma omp parallel for
        for(int j = 0; j < n; j++)  
            to[i][j] = from[i][j];
    }
}

double** empty_matrix(int m, int n){
    double** A = new double*[m];
    for(int i = 0; i < m; i++){
        A[i] = new double[n];
        for(int j = 0; j < n; j++)  
            A[i][j] = 0;
    }
    return A;
}

double** diagonal_matrix(int n){
    double** A = new double*[n];
    for(int i = 0; i < n; i++){
        A[i] = new double[n];
        for(int j = 0; j < n; j++)  
            A[i][j] = (i == j) ? 1 : 0;
    }
    return A;
}

float matrix_multiply(double** res, double** A, double** B, int N, int M, int N1){
    // Matrices shapes: A = NxM, B = MxN1, res = NxN1
    double diff = 0; double old;
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N1; j++){
            old = res[i][j];
            res[i][j] = 0;
            for(int k = 0; k < M; k++)
                res[i][j] += A[i][k] * B[k][j];
            diff = max(diff, fabs(fabs(res[i][j]) - fabs(old)));
        }
    }
    return (float)diff;
}

void print_matrix(double** A, int M, int N, char* name){
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

double** mat_transpose(double** A, int Am, int An) {
    double **B;
    B = (double**)malloc(__SIZEOF_POINTER__*An);
    for (int i=0; i<An; i++)
        B[i] = (double*)malloc(__SIZEOF_DOUBLE__*Am);

    for (int i=0; i<Am; i++){
        for (int j=0; j<An; j++){
            B[j][i] = A[i][j];
        }
    }

    return B;
}

double** mat_mul(double** A, int Am, int An, 
                 double** B, int Bm, int Bn){
    double **C;
    C = (double**)malloc(__SIZEOF_POINTER__*Am);
    for (int i=0; i<Am; i++)
        C[i] = (double*)malloc(__SIZEOF_DOUBLE__*Bn);

    for (int i=0; i<Am; i++){
        for (int j=0; j<Bn; j++){
            C[i][j] = 0;
            for (int k=0; k<An; k++){
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

int maxind(int k) {
    int m = k+1;

    for (int i = k+2; i < N; i++){
        if (fabs(S[k][i]) > fabs(S[k][m])){
            m = i;
        }
    }

    return m;
}

void update(int k, double t) {
    double ek_prev = e[k];
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
    double** mat1;
    double** mat2;
    double** mat3;

    mat1 = (double**)malloc(__SIZEOF_POINTER__*2);
    mat1[0] = (double*)malloc(__SIZEOF_DOUBLE__*2);
    mat1[1] = (double*)malloc(__SIZEOF_DOUBLE__*2);
    mat1[0][0] = c; mat1[0][1] = -s;
    mat1[1][0] = s; mat1[1][1] = c;

    mat2 = (double**)malloc(__SIZEOF_POINTER__*2);
    mat2[0] = (double*)malloc(__SIZEOF_DOUBLE__*1);
    mat2[1] = (double*)malloc(__SIZEOF_DOUBLE__*1);
    if (eigenvectors){
        mat2[0][0] = E[i][k];
        mat2[1][0] = E[i][l];
    }
    else {
        mat2[0][0] = S[k][l];
        mat2[1][0] = S[i][j];
    }

    mat3 = mat_mul(mat1, 2, 2, mat2, 2, 1);

    if (eigenvectors){
        E[i][k] = mat3[0][0];
        E[i][l] = mat3[1][0];
    }
    else{
        S[k][l] = mat3[0][0];
        S[i][j] = mat3[1][0];
    }

    free(mat1[0]);
    free(mat1[1]);
    free(mat1);
    free(mat2[0]);
    free(mat2[1]);
    free(mat2);
    free(mat3[0]);
    free(mat3[1]);
    free(mat3);
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

    while(state != 0){
        int m = 0;

        for (int k=1; k<N-1; k++){
            if (fabs(S[k][ind[k]]) > fabs(S[m][ind[m]])){
                m = k;
            }
        }

        int k = m;
        int l = ind[m];
        double p = S[k][l];
        double y = (e[l] - e[k]) / 2.0;
        double d = fabs(y) + sqrt(p*p + y*y);
        double r = sqrt(p*p + d*d);
        double c = d / r;
        double s = p / r;
        double t = (p*p) / d;

        if (y < 0.0) { s = -s; t = -t; }

        S[k][l] = 0.0;
        update(k, -t);
        update(l, t);

        for (int i=0; i<k; i++)  { rotate(i, k, i, l, c, s, false); }
        for (int i=k+1; i<l; i++){ rotate(k, i, i, l, c, s, false); }
        for (int i=l+1; i<N; i++)  { rotate(k, i, l, i, c, s, false); }

        for (int i=0; i<N; i++){
            rotate(k, l, i, i, c, s, true);
        }

        ind[k] = maxind(k);
        ind[l] = maxind(l);
    }

    *eigenvalues = e;
    *eigenvectors = E;
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

    *SIGMAm = M;
    *SIGMAm = N;

    printf("Starting SVD\n");
    // Dt is D transpose = NxM
    double** Dt = empty_matrix(N, M);
    // Dc is copy of D = MxN
    double** Dc = empty_matrix(M, N);

    // DtD is Dt.D = NxN, so are Q and R
    double** DtD = empty_matrix(N, N);
    
    // Compute Dt and Dc
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            Dt[j][i] = D[i*N + j];
            Dc[i][j] = D[i*N + j];
        }
    }

    // Multiply Dt.D = NxM . MxN = NxN
    matrix_multiply(DtD, Dt, Dc, N, M, N);

    print_matrix(DtD, N, N, "DtD\0");

    // Get Eigenvalues of DtD i.e. Q and R
    double** Ei = empty_matrix(N, N);
    double** Ei_temp = empty_matrix(N, N);

    printf("Starting jacobi\n");

    // Jacobi
    double **prod, *eigenvalues, **eigenvectors;

    Jacobi(DtD, N, &eigenvalues, &eigenvectors);

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            Ei[j][i] = eigenvectors[j][i];

    // Extract eigenvalues into an array
    // double* eigenvalues = new double[N];
    double* eigenvalues1 = new double[N];
    for(int i = 0; i < N; i++){
        eigenvalues[i] = eigenvalues[i];
        eigenvalues1[i] = eigenvalues[i];
    }
    
    std::sort(eigenvalues, eigenvalues + N);
    reverse_array(eigenvalues, N);
    for(int i = 0; i < N; i++){
        printf("Eigenvals = %f, \t\t %f\n", eigenvalues[i], eigenvalues1[i]);
    }

    // Update Ei
    Ei_temp = empty_matrix(N, N);
    for(int j = 0; j < N; j++){
        int p = 0;
        // Find p i.e. index of jth max eigenvalue
        for(p = 0; p < N; p++){
            if(eigenvalues1[p] == eigenvalues[j])
                break;
        }
        printf("p=%d, j=%d\n",p,j);
        for(int i = 0; i < N; i++){
            Ei_temp[i][j] = Ei[i][p];
        }
    }
    print_matrix(Ei, N, N, "Ei\0");
    print_matrix(Ei_temp, N, N, "Ei_temp\0");

    copy_matrix(Ei, Ei_temp, N, N);

    double** sigma = empty_matrix(M, N);
    double** sigma_inv = empty_matrix(N, M);
    double* sigma_vals = new double[N];
    for(int i = 0; i < N; i++){
        sigma_vals[i] = sqrt(eigenvalues[i]);
        sigma[i][i] = sqrt(eigenvalues[i]);
        sigma_inv[i][i] = (1.0 / sqrt(eigenvalues[i]));
    }

    SIGMA = &sigma_vals;

    double** Vt = empty_matrix(M, M);
    double** U_temp = empty_matrix(N, N);
    double* Ui = new double[N*N];
    printf("U\n");
    // Compute U
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            Ui[N*i + j] = Ei[i][j];
            U_temp[i][j] = Ei[i][j];
        }
    }
    
    printf("U\n");
    U = &Ui;
    double** temp = empty_matrix(M, N);
    double** temp2 = empty_matrix(M, M);
    matrix_multiply(temp, Dc, Ei, M, N, N);
    matrix_multiply(temp2, temp, sigma_inv, M, N, M);

    // V_T
    double* Vi = new double[M*M];
    for(int i = 0; i < M; i++)
        for(int j = 0; j < M; j++){
            Vi[M*j+i] = temp2[i][j];
            Vt[j][i] = temp2[i][j];
        }

    double ret = double(retention)/100;
    double sumeigen = 0;
    for(int i = 0; i < N; i++){
        sumeigen += sigma[i][i] * sigma[i][i];
        printf("Sigma %d is %f\n", i, *(*SIGMA + i));
    }

    V_T = &Vi;

    // Test U = M * V * Sigma-1
    matrix_multiply(temp, U_temp, sigma, N, N, M);
    matrix_multiply(temp2, temp, Vt, N, M, M);

    printf("Comparison result diff = %f\n", compare_matrices(temp2, Dt, N, M));


///////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////PCA//////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////


    double sumret = 0; int k = 0;
    for(k = 0; k < N; k++){
        sumret += (sigma[k][k] * sigma[k][k]/ sumeigen);
        if(sumret >= ret)
            break;
    }

    *K = k+1;
    printf("K = %d\n", *K);
    double** W = empty_matrix(N, k+1);
    for(int i = 0; i < N; i++){
        for(int j = 0; j <= k; j++){
            W[i][j] = U_temp[i][j];
        }
    }

    // Print W
    print_matrix(W, N, *K, "W\0");

    printf("D-Hat:\n");
    double* DHatTemp = (double *)malloc(sizeof(double)*((k+1) * M));
    for(int i = 0; i < M; i++){
        for(int j = 0; j <= k; j++){
            DHatTemp[i*(k+1) + j] = 0;
            for(int p = 0; p < N; p++){
                *(DHatTemp + i*(k+1) + j) += *(D + i*N + p) * W[p][j];
            }
            printf("%f ", DHatTemp[i*(k+1) + j]);
        }
        printf("\n");
    }

    D_HAT = &DHatTemp;
}

