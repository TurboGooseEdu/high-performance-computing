#include <iostream>
#include <cstdlib>
#include <fstream>
#include "mpi.h"

#define RUNS 3

using namespace std;

char* output_file = "results.txt";

int comm_size;
int my_rank;

int N = 100;

void print_if_root(char* message) {
    if (my_rank == 0) {
        cout << message << endl;
    }
}

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


void fill_matrix_with_random_nums(double* matrix) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            matrix[i * N + j] = rand() % 10;
        }
    }
}


double* matmul(double* a, double* b) {
    int rows_per_proc = (int) (N / comm_size);
    int elems_per_proc = rows_per_proc * N;
    int offset = my_rank * elems_per_proc;
    double* res = init_matrix();

    MPI_Bcast(b, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatter(a, elems_per_proc, MPI_DOUBLE, my_rank ? a + offset: MPI_IN_PLACE, elems_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = my_rank * rows_per_proc; i < (my_rank + 1) * rows_per_proc; i++) {
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                res[i * N + j] += a[i * N + k] * b[k * N + j];
    }

    MPI_Gather(my_rank ? res + offset: MPI_IN_PLACE, elems_per_proc, MPI_DOUBLE, res, elems_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

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
    int rows_per_proc = (int) (N / comm_size);
    int elems_per_proc = rows_per_proc * N;
    int offset = my_rank * elems_per_proc;
    double* res = init_matrix();

    MPI_Scatter(a, elems_per_proc, MPI_DOUBLE, my_rank ? a + offset: MPI_IN_PLACE, elems_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatter(b, elems_per_proc, MPI_DOUBLE, my_rank ? b + offset: MPI_IN_PLACE, elems_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = my_rank * rows_per_proc; i < (my_rank + 1) * rows_per_proc; i++) {
        for (int j = 0; j < N; j++) {
            res[i * N + j] = a[i * N + j] + b[i * N + j];
        }
    }

    MPI_Gather(my_rank ? res + offset: MPI_IN_PLACE, elems_per_proc, MPI_DOUBLE, res, elems_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

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

    print_if_root("\n");
    double* B_3 = pow(B, 3);
    print_if_root("B^3 calculated");
    double* C_3 = pow(C, 3);
    print_if_root("C^3 calculated");

    double* D = get_matrix_of_numbers(trace_of_sum(B_3, C_3));
    print_if_root("Tr(B^3 + C^3)E calculated");

    double* B_4 = matmul(B_3, B);
    print_if_root("B^4 calculated");
    double* C_4 = matmul(C_3, C);
    print_if_root("C^4 calculated");

    double* A = sum(sum(B_4, C_4), D);
    print_if_root("Sum calculated");
    print_if_root("----");

    delete_matrix(B_3);
    delete_matrix(C_3);
    delete_matrix(D);
    delete_matrix(B_4);
    delete_matrix(C_4);

    return A;
}

double calculate_with_time_measuring(double* B, double* C) {    
    double start = MPI_Wtime(); 
    double* result = calculate(B, C);
    double end = MPI_Wtime();

    delete_matrix(result);
    return end - start;
}

void writeToFile(double time_elapsed) {
    ofstream file(output_file, ios::app);
    if (!file.is_open()) {
        printf("Cannot open file %s", output_file);
        exit(1);
    }
    file << N << ", " << comm_size << ", " << time_elapsed << endl;
    file.close();
}

void run_experiment() {
    double* B = init_matrix();
    double* C = init_matrix();
    if (my_rank == 0) {
        fill_matrix_with_random_nums(B);
        fill_matrix_with_random_nums(C);

        double total_time = 0;
        for (int i = 0; i < RUNS; i++) {
            total_time += calculate_with_time_measuring(B, C);
        }
        double avg_elapsed = total_time / RUNS;
        cout << "Average time elapsed (secs): " << avg_elapsed << endl;
        
        writeToFile(avg_elapsed);
    } else {
        for (int i = 0; i < RUNS; i++) {
            calculate(B, C);
        }
    }
}


int main(int argc, char* argv[]) {
    srand(1);
    if (argc > 1) {
        N = stoi(argv[1]);
        if (argc == 3) {
            output_file = argv[2];
        }
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    cout << "Process " << my_rank << " running" << endl;

    run_experiment();

    MPI_Finalize();

    return 0;
}