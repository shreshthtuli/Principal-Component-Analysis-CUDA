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

class SymmetricMatrix
{
    private:
        size_t m_size;
        double* m_mat;
        double precision;
    
    public:
        double* eigenvalues;
        double** eVects;
        SymmetricMatrix(size_t mat_size, double prec);

        class Row
        {
            friend class SymmetricMatrix;
            private:
                SymmetricMatrix& m_mat;
                size_t m_row;
                Row(SymmetricMatrix& mat, size_t row) : m_mat(mat), m_row(row) {}
            public:
                double& operator[](size_t index);
        };

        Row operator[](size_t index);
        void calculateEigens();
};

void maxInd(size_t *p, size_t *q, double *Apq, size_t size, SymmetricMatrix& matA)
{
    double Aij;
    *p = 0, *q = 1;
    *Apq = abs(matA[0][1]);
    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < i; j++)
        {
            Aij = abs(matA[i][j]);
            if (Aij > *Apq)
            {
                *Apq = Aij;
                *p = i;
                *q = j;
            }
        }
    }
}

void calcPhiTCS(double *phi, size_t p, size_t q, double *t, double *c, double *s, SymmetricMatrix& matA)
{
    *phi = (matA[q][q] - matA[p][p]) / (2 * matA[p][q]);
    *t = *phi == 0 ? 1 : (1 / (*phi + (*phi > 0 ? 1 : -1) * sqrt(*phi * *phi + 1)));
    *c = 1 / sqrt(1 + *t * *t);
    *s = *t / sqrt(1 + *t * *t);
}

void populateA(double **A, size_t p, size_t q, double c, double s, SymmetricMatrix& matA, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            if (i == p)
                A[i][j] = matA[p][j] * c - matA[q][j] * s;
            else if (i == q)
                A[i][j] = matA[p][j] * s + matA[q][j] * c;
            else
                A[i][j] = matA[i][j];
        }
    }
}

void populateMatA(double **A, size_t p, size_t q, double c, double s, SymmetricMatrix& matA, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j <= i; j++)
        {
            if (j == p)
                matA[i][p] = A[i][p] * c - A[i][q] * s;
            else if (j == q)
                matA[i][q] = A[i][p] * s + A[i][q] * c;
            else
                matA[i][j] = A[i][j];
        }
    }
}

void computeNext(double **now, double **next, size_t p, size_t q, double c, double s, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            if (j == p)
                next[i][j] = now[i][p] * c - now[i][q] * s;
            else if (j == q)
                next[i][j] = now[i][p] * s + now[i][q] * c;
            else
                next[i][j] = now[i][j];
        }
    }
}

SymmetricMatrix::SymmetricMatrix(size_t mat_size, double prec) 
{
    m_size = mat_size;
    m_mat = new double[(mat_size - 1) * mat_size / 2 + mat_size];
    precision = abs(prec);
    eigenvalues = new double[mat_size];
    eVects = diagonal_matrix(mat_size);
}

SymmetricMatrix::Row SymmetricMatrix::operator[](size_t row)
{
    return Row(*this, row);
}

double& SymmetricMatrix::Row::operator[](size_t col)
{
    size_t r = max(m_row, col);
    size_t c = min(m_row, col);
    return m_mat.m_mat[(r + 1) * r / 2 + c];
}

void SymmetricMatrix::calculateEigens()
{
    size_t size = m_size;

    SymmetricMatrix* matAp = this;
    SymmetricMatrix& matA = *matAp;
    double Aij, Apq = 100, phi, t, c, s;
    double** A = empty_matrix(size, size);
    double** nextEVects = empty_matrix(size, size);
    size_t p, q;

    while(Apq > precision)
    {
        maxInd(&p, &q, &Apq, size, matA);

        calcPhiTCS(&phi, p, q, &t, &c, &s, matA);

        populateA(A, p, q, c, s, matA, size);
    
        populateMatA(A, p, q, c, s, matA, size);

        populateMatA(A, p, q, c, s, matA, size);

        computeNext(eVects, nextEVects, p, q, c, s ,size);        

        copy_matrix(eVects, nextEVects, size, size);
    }

    for (size_t i = 0; i < size; i++)
        eigenvalues[i] = matA[i][i];
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
    SymmetricMatrix mat(N, 1e-10);

    for (int i = 0; i < N; i++)
        for (int j = 0; j <= i; j++)
            mat[i][j] = DtD[i][j];

    mat.calculateEigens();
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            Ei[j][i] = mat.eVects[j][i];

    // Extract eigenvalues into an array
    double* eigenvalues = new double[N];
    double* eigenvalues1 = new double[N];
    for(int i = 0; i < N; i++){
        eigenvalues[i] = mat.eigenvalues[i];
        eigenvalues1[i] = mat.eigenvalues[i];
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

