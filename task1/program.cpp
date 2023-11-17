#include <omp.h>
#include <iostream>
#include <iostream>
#include <cstdlib>
#include <ctime>

#define N 2


double** init_matrix() {
    double** matrix = new double*[N];
    for (int i = 0; i < N; ++i) {
        matrix[i] = new double[N];
    }
    return matrix;
}


double** copy_matrix(double** matrix) {
    double** copy = init_matrix();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            copy[i][j] = matrix[i][j];
        }
    }
    return matrix;
}

void print_matrix(double** matrix) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


double** generate_random_square_matrix(unsigned int seed) {
    std::srand(seed);
    double** matrix = new double*[N];
    for (int i = 0; i < N; ++i) {
        matrix[i] = new double[N];
        for (int j = 0; j < N; ++j) {
            matrix[i][j] = std::rand() % 10;
        }
    }
    print_matrix(matrix);
    return matrix;
}


double** matmul(double** a, double** b) 
{
    double** res = init_matrix();
    int i, j, k;
    // #pragma omp parallel for private(i,j,k)
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            for (k = 0; k < N; ++k) {
                res[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return res;
} 


double** deg(double** matrix, int deg) {
    double** mulcopy = copy_matrix(matrix);
    for (int i = 0; i < deg - 1; i++) {
        mulcopy = matmul(mulcopy, matrix);
        print_matrix(mulcopy);
    }
    return mulcopy;
}

double** sum(double** a, double** b) {
    double** res = init_matrix();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            res[i][j] = a[i][j] + b[i][j];
        }
    }
    return res;
}

double** composite_calc(double** B_3, double** C_3) {
    double** res = init_matrix();
    for (int i = 0; i < N; i++) {
        int C = B_3[i][i] + B_3[i][i];
        for (int j = 0; j < N; j++) {
            res[i][j] = C;
        }
    }
    return res;
}

// A = B^4 + C^4 + Tr(B^3 + C^3)E

int main()
{
    omp_set_num_threads(omp_get_num_procs());

    double** B = generate_random_square_matrix(1);
    double** C = generate_random_square_matrix(2);

    double** B_3 = deg(B, 3);
    double** C_3 = deg(C, 3);
    double** D = composite_calc(B_3, C_3);

    double** B_4 = matmul(B_3, B);
    double** C_4 = matmul(C_3, C);

    double** A = sum(sum(B_4, C_4), D);

    print_matrix(B_4);
    print_matrix(C_4);


    return 0;
}