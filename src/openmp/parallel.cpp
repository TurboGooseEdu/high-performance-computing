#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

#define RUNS 10
int N = 10;


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


double** generate_random_matrix(unsigned int seed) {
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


double** matmul(double** a, double** b) {
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

double** calculate(double** B, double** C) {
    // A = B^4 + C^4 + Tr(B^3 + C^3)E

    double** B_3 = pow(B, 3);
    double** C_3 = pow(C, 3);

    double** D = get_matrix_of_numbers(trace_of_sum(B_3, C_3));

    double** B_4 = matmul(B_3, B);
    double** C_4 = matmul(C_3, C);

    double** A = sum(sum(B_4, C_4), D);
    return A;
}

double calculate_with_time_measuring(double** B, double** C) {
    double start = omp_get_wtime();
    calculate(B, C);
    double end = omp_get_wtime();
    return end - start;
}

double calculate_mean_elapsed(double** B, double** C) {
    double sum_times = 0;
    for (int i = 0; i < RUNS; i++) {
        sum_times += calculate_with_time_measuring(B, C);
    }
    return sum_times / RUNS;
}

void run_experiment() {
    int threads = 1;

    double prev_result = 0;
    double cur_result = 999999999;

    double** B = generate_random_matrix(1);
    double** C = generate_random_matrix(2);
    
    do {
        omp_set_num_threads(threads);
        prev_result = cur_result;
        cur_result = calculate_mean_elapsed(B, C);
        printf("%d, %d, %f\n", N, threads, cur_result);
        threads++;
    } while (cur_result < prev_result);
}

int main(int argc, char* argv[]) {

    if (argc == 2) {
        N = std::stoi(argv[1]);
    }

    run_experiment();

    return 0;
}