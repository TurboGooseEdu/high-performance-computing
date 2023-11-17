#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

#define N 1000


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
    return copy;
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

double** get_matrix_of_numbers(double num) {
    double** matrix = new double*[N];
    for (int i = 0; i < N; ++i) {
        matrix[i] = new double[N];
        for (int j = 0; j < N; j++) {
            matrix[i][j] = num;
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
            matrix[i][j] = std::rand() % 10;
        }
    }
    return matrix;
}


double** matmul(double** a, double** b) 
{
    double** res = init_matrix();
    int i, j, k;
    #pragma omp parallel for private(i,j,k) shared(a,b,res)
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            for (k = 0; k < N; ++k) {
                res[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return res;
} 


double** pow(double** matrix, int deg) {
    double** mulcopy = copy_matrix(matrix);
    for (int i = 0; i < deg - 1; i++) {
        mulcopy = matmul(mulcopy, matrix);
    }
    return mulcopy;
}

double** sum(double** a, double** b) {
    double** res = init_matrix();
    int i, j;
    #pragma omp parallel for private(i,j) shared(a,b,res)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            res[i][j] = a[i][j] + b[i][j];
        }
    }
    return res;
}

double trace_of_sum(double** a, double** b) {
    double trace = 0;
    #pragma omp parallel for reduction (+:trace)
    for (int i = 0; i < N; i++) {
        trace += a[i][i] + b[i][i];
    }
    return trace;
}

double calculate(double** B, double** C) {
    double start = omp_get_wtime();

    double** B_3 = pow(B, 3);
    double** C_3 = pow(C, 3);

    double** D = get_matrix_of_numbers(trace_of_sum(B_3, C_3));

    double** B_4 = matmul(B_3, B);
    double** C_4 = matmul(C_3, C);


    double** A = sum(sum(B_4, C_4), D);

    double end = omp_get_wtime();
    return end - start;
}

// A = B^4 + C^4 + Tr(B^3 + C^3)E

int main()
{
    int threads = 1;

    double prev_result = 0;
    double cur_result = 999999999;

    double** B = generate_random_square_matrix(1);
    double** C = generate_random_square_matrix(2);
    
    do {
        omp_set_num_threads(threads);
        prev_result = cur_result;
        cur_result = calculate(B, C);
        printf("Threads: %d; Elapsed time: %f\n", threads, cur_result);
        threads++;
    } while (cur_result < prev_result);
    return 0;
}