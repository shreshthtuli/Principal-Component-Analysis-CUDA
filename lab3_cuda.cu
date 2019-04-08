#include "lab3_cuda.h"
#include <cmath>
#include <malloc.h>
#include <math.h>
#include <algorithm>


using namespace std;

///////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////Jacobi////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

typedef struct struct_eigen
{
    double value;
    double* vector;
} eigen;


class SymmetricMatrix
{
    public:
        SymmetricMatrix();
        SymmetricMatrix(size_t mat_size);
        virtual ~SymmetricMatrix();
        SymmetricMatrix(const SymmetricMatrix& other);

        size_t Getsize() const { return m_size; }

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
        eigen* calculateEigens(double precision);

    protected:

    private:
        const size_t m_size;
        double* const m_mat;
};

SymmetricMatrix::SymmetricMatrix() : m_size(0), m_mat(nullptr)
{
}

SymmetricMatrix::SymmetricMatrix(size_t mat_size) : m_size(mat_size), m_mat(new double[(mat_size - 1) * mat_size / 2 + mat_size])
{
}

SymmetricMatrix::~SymmetricMatrix()
{
    delete[] m_mat;
}

SymmetricMatrix::SymmetricMatrix(const SymmetricMatrix& other) : SymmetricMatrix(other.m_size)
{
    size_t arr_size = (m_size - 1) * m_size / 2 + m_size;
    for (size_t i = 0; i < arr_size; i++)
        m_mat[i] = other.m_mat[i];
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

eigen* SymmetricMatrix::calculateEigens(double precision)
{
    precision = abs(precision);
    size_t size = Getsize();

    if (size == 0)
        return nullptr;
    if (size == 1)
    {
        eigen* e = new eigen[1];
        e[0].value = m_mat[0];
        e[0].vector = new double[1];
        e[0].vector[0] = 1;
        return e;
    }

    double* eVects = new double[size * size];
    for (size_t i = 0; i < size; i++)
        for (size_t j = 0; j < size; j++)
            eVects[i * size + j] = i == j ? 1 : 0;

    SymmetricMatrix* matAp = this;
    while(true)
    {
        SymmetricMatrix& matA = *matAp;

        // Computing max Aij -> Apq, p, q
        size_t p = 0, q = 1;
        double Apq = abs(matA[0][1]);
        for (size_t i = 0; i < size; i++)
        {
            for (size_t j = 0; j < i; j++)
            {
                double Aij = abs(matA[i][j]);
                if (Aij > Apq)
                {
                    Apq = Aij;
                    p = i;
                    q = j;
                }
            }
        }

        if (Apq < precision)
            break;

        double phi = (matA[q][q] - matA[p][p]) / (2 * matA[p][q]);
        double t = phi == 0 ? 1 : (1 / (phi + (phi > 0 ? 1 : -1) * sqrt(phi * phi + 1)));
        double c = 1 / sqrt(1 + t * t);
        double s = t / sqrt(1 + t * t);

        // Computing Jacob rotation
        double* A_ = new double[size * size];
        for (size_t i = 0; i < size; i++)
        {
            for (size_t j = 0; j < size; j++)
            {
                size_t index = i * size + j;
                if (i == p)
                    A_[index] = matA[p][j] * c - matA[q][j] * s;
                else if (i == q)
                    A_[index] = matA[p][j] * s + matA[q][j] * c;
                else
                    A_[index] = matA[i][j];
            }
        }
        SymmetricMatrix* nextMatA = new SymmetricMatrix(size);
        SymmetricMatrix& A__ = *nextMatA;
        for (size_t i = 0; i < size; i++)
        {
            for (size_t j = 0; j <= i; j++)
            {
                if (j == p)
                    A__[i][p] = A_[i * size + p] * c - A_[i * size + q] * s;
                else if (j == q)
                    A__[i][q] = A_[i * size + p] * s + A_[i * size + q] * c;
                else
                    A__[i][j] = A_[i * size + j];
            }
        }

        delete[] A_;
        if (matAp != this)
            delete matAp;

        matAp = nextMatA;

        // Computing eigenvectors
        double* nextEVects = new double[size * size];
        for (size_t i = 0; i < size; i++)
        {
            for (size_t j = 0; j < size; j++)
            {
                size_t k = i * size + j;
                if (j == p)
                    nextEVects[k] = eVects[i * size + p] * c - eVects[i * size + q] * s;
                else if (j == q)
                    nextEVects[k] = eVects[i * size + p] * s + eVects[i * size + q] * c;
                else
                    nextEVects[k] = eVects[k];
            }
        }
        delete[] eVects;
        eVects = nextEVects;
    }

    eigen* e = new eigen[size];
    for (size_t i = 0; i < size; i++)
    {
        e[i].value = (*matAp)[i][i];
        e[i].vector = new double[size];
        for (size_t j = 0; j < size; j++)
        {
            e[i].vector[j] = eVects[j * size + i];
        }
    }

    if (matAp != this)
        delete matAp;
    delete[] eVects;

    return e;
}

///////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////SVD//////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

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


void SVD_and_PCA (int M, 
        int N, 
        double* D, 
        double** U, 
        double** SIGMA, 
        double** V_T, 
        double** D_HAT, 
        int *K,
        int retention) {

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
    // Di has diagonal entries as eigenvalues
    // Ei has eigenvectors as columns
    SymmetricMatrix mat(N);

    for (int i = 0; i < N; i++)
        for (int j = 0; j <= i; j++)
            mat[i][j] = DtD[i][j];

    eigen* e = mat.calculateEigens(1e-10);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            Ei[j][i] = e[i].vector[j];

    // Extract eigenvalues into an array
    double* eigenvalues = new double[N];
    double* eigenvalues1 = new double[N];
    for(int i = 0; i < N; i++){
        eigenvalues[i] = e[i].value;
        eigenvalues1[i] = e[i].value;
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
    for(int i = 0; i < N; i++){
        *(*SIGMA+i) = sqrt(eigenvalues[i]);
        sigma[i][i] = sqrt(eigenvalues[i]);
        sigma_inv[i][i] = (1.0 / sqrt(eigenvalues[i]));
        // printf("Sigma %d = %f\n", i, *(*SIGMA+i));
    }

    double** Vt = empty_matrix(M, M);
    double** U_temp = empty_matrix(N, N);
    // Compute U
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            *(*U + N*i + j) = Ei[i][j];
            U_temp[i][j] = Ei[i][j];
        }
    }
    
    double** temp = empty_matrix(M, N);
    double** temp2 = empty_matrix(M, M);
    matrix_multiply(temp, Dc, Ei, M, N, N);
    matrix_multiply(temp2, temp, sigma_inv, M, N, M);

    // V_T
    for(int i = 0; i < M; i++)
        for(int j = 0; j < M; j++){
            *(*V_T + M*j + i) = temp2[i][j]; 
            Vt[j][i] = temp2[i][j];
        }

    double ret = double(retention)/100;
    double sumeigen = 0;
    for(int i = 0; i < N; i++){
        sumeigen += sigma[i][i] * sigma[i][i];
        printf("Sigma %d is %f\n", i, *(*SIGMA + i));
    }

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

