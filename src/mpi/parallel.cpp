#include <iostream>
#include <cstdlib>
#include <chrono>
#include "mpi.h"

#define RUNS 10

using namespace std;

int comm_size;
int my_rank;
int N = 3;

double* init_matrix() {
    return new double[N * N];
}

void delete_matrix(double* matrix) {
    delete[] matrix;
}


double* copy_matrix(double* matrix) {
    double* copy = init_matrix();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            copy[i * N + j] = matrix[i * N + j];
        }
    }
    return copy;
}

void print_matrix(double* matrix) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << matrix[i * N + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

double* get_matrix_of_numbers(double num) {
    double* matrix = new double[N * N];
    fill_n(matrix, N * N, num);
    return matrix;
}


double* generate_random_matrix() {
    double* matrix = new double[N * N];
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            matrix[i * N + j] = rand() % 10;
        }
    }
    return matrix;
}


double* matmul(double* a, double* b) {
    double* res = init_matrix();
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                res[i * N + j] += a[i * N + k] * b[k * N + j];
            }
        }
    }
    return res;
} 


double* pow(double* matrix, int deg) {
    double* mulcopy = copy_matrix(matrix);
    for (int i = 0; i < deg - 1; i++) {
        mulcopy = matmul(mulcopy, matrix);
    }
    return mulcopy;
}

double* sum(double* a, double* b) {
    double* res = init_matrix();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            res[i * N + j] = a[i * N + j] + b[i * N + j];
        }
    }
    return res;
}

double trace_of_sum(double* a, double* b) {
    double trace = 0;
    for (int i = 0; i < N; i++) {
        trace += a[i * N + i] + b[i * N + i];
    }
    return trace;
}

double* calculate(double* B, double* C) {
    // A = B^4 + C^4 + Tr(B^3 + C^3)E

    double* B_3 = pow(B, 3);
    double* C_3 = pow(C, 3);

    double* D = get_matrix_of_numbers(trace_of_sum(B_3, C_3));

    double* B_4 = matmul(B_3, B);
    double* C_4 = matmul(C_3, C);

    double* A = sum(sum(B_4, C_4), D);

    delete_matrix(B_3);
    delete_matrix(C_3);
    delete_matrix(D);
    delete_matrix(B_4);
    delete_matrix(C_4);

    return A;
}

double calculate_with_time_measuring(double* B, double* C) {
    auto start = chrono::high_resolution_clock::now();
    double* result = calculate(B, C);
    auto end = chrono::high_resolution_clock::now();
    auto elapsed_secs = chrono::duration_cast<chrono::duration<double>>(end - start);
    delete_matrix(result);
    return elapsed_secs.count();
}

double calculate_mean_elapsed(double* B, double* C) {
    double sum_times = 0;
    for (int i = 0; i < RUNS; i++) {
        sum_times += calculate_with_time_measuring(B, C);
    }
    return sum_times / RUNS;
}

void run_experiment() {
    double* B = generate_random_matrix();
    double* C = generate_random_matrix();
    
    int result = calculate_mean_elapsed(B, C);
    printf("%d, %d, %f\n", N, comm_size, result);
}

int main(int argc, char* argv[]) {

    srand(1);
    if (argc == 2) {
        N = stoi(argv[1]);
    }

    MPI_Init(argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    run_experiment();

    MPI_Finalize();

    return 0;
}