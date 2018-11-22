/**
 * MIPT | DIHT | 3 course | Parallel computing
 * Task 1. Sum of the array.
 *
 * @author Mary Feofanova
 */

#include <stdio.h>
#include <mpi.h>
#include <string.h>

const int N = 50000;

int main(int argc, char **argv) {
    int a[N];
    int my_rank, num_procs, offset;
    long local_sum;
    offset = 0;
    local_sum = 0;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int chunk = N / num_procs;

    printf("I am process number %d of %d\n", my_rank, num_procs);
    if (my_rank == 0) {
        //Initialize array
        for (int i = 0; i < N; i++) {
            a[i] = i;
        }

        // Send array chunks to others
        for (int other_rank = 1; other_rank < num_procs; other_rank++) {
            MPI_Send(a + chunk * (other_rank - 1), chunk, MPI_INT, other_rank, 0, MPI_COMM_WORLD);
        }
        offset = chunk * (num_procs - 1);

        // Compute local sum
        for (int i = 0; i < N; i++) {
            local_sum += a[i];
        }
    } else {
        MPI_Recv(a, chunk, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    }

    // Compute array chunk locally
    long my_sum = 0;
    int end = my_rank == 0 ? N : offset + chunk;
    for (int i = offset; i < end; i++) {
        my_sum += a[i];
    }

    // Send back if necessary
    if (my_rank != 0) {
        MPI_Send(&my_sum, 1, MPI_LONG, 0, 0, MPI_COMM_WORLD);
    }

    // First process collects all information and represents it
    if (my_rank == 0) {
        long proc_sum = 0;
        long sum = my_sum;

        printf("Sum of process %d is: %ld\n", 0, my_sum);
        for (int other_rank = 1; other_rank < num_procs; other_rank++) {
            MPI_Recv(&proc_sum, 1, MPI_LONG, other_rank, 0, MPI_COMM_WORLD, &status);
            sum += proc_sum;
            printf("Sum of process %d is: %ld\n", other_rank, proc_sum);
        }

        printf("Total locally computed sum: %ld\n", local_sum);
        printf("Total parallel sum: %ld\n", sum);
        printf("Difference between computations: %ld\n", local_sum - sum);
        char* result = local_sum - sum == 0 ? "OK" : "FAILED";
        printf("===================%s===================\n", result);
    }

    MPI_Finalize();
    return 0;
}
