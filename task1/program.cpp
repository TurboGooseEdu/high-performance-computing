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


double** generate_random_square_matrix(unsigned int seed) {
    std::srand(seed);
    double** matrix = new double*[N];
    for (int i = 0; i < N; ++i) {
        matrix[i] = new double[N];
        for (int j = 0; j < N; ++j) {
            matrix[i][j] = std::rand() % 100;
        }
    }
    return matrix;
}

double** get_ones_matrix() {
    double** matrix = new double*[N];
    for (int i = 0; i < N; ++i) {
        matrix[i] = new double[N];
        for (int j = 0; j < N; ++j) {
            matrix[i][j] = 1;
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
}


double** parallel_matmul(double** a, double** b) 
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

// A = B^4 + C^4 + Tr(B^3 + C^3)E

double** deg(double** matrix, int deg) {
    double** mulcopy = copy_matrix(matrix);
    for (int i = 0; i < deg - 1; i++) {
        mulcopy = parallel_matmul(mulcopy, matrix);
        print_matrix(mulcopy);
        std::cout << std::endl;
    }
    return mulcopy;
}

int main()
{

    omp_set_num_threads(omp_get_num_procs());

    double** B = generate_random_square_matrix(1);
    double** C = generate_random_square_matrix(2);
    double** E = get_ones_matrix();

    print_matrix(B);
    std::cout << std::endl;


    double** B_3 = deg(B, 3);




    // parallel_matmul(B_mul, B, B_mul); // B^3
    // print_matrix(B_mul);

    // double** C_mul = copy_matrix(C);
    // parallel_matmul(C_mul, C, C_mul); // C^2
    // parallel_matmul(C_mul, C, C_mul); // C^3

    // parallel_matmul(C_mul, C, C_mul); // C^4


    return 0;
}