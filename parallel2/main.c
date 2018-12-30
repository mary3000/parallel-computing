/**
 * MIPT | DIHT | 3 course | Parallel computing
 * Task 2. Heat equation.
 *
 * @author Mary Feofanova
 */

#include <mpi.h>
#include <math.h>
#include <malloc.h>
#include <string.h>
#include <time.h>

#define U_LEFT 80;
#define U_RIGHT 30;
#define U_0 5;

#define RO 8960;
#define C 380;
#define LAMBDA 401;

void ComputeHeat(int num_procs, int my_rank, double **heat, int height, int size_num, int time_num,
                 int remainder, double time_step, double area_step) {
    double lambda = LAMBDA;
    double ro = RO;
    double c = C;
    double k = lambda / (ro * c);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < size_num; j++) {
            if (j == 0) {
                heat[i][j] = U_LEFT;
            } else if (j == size_num - 1) {
                heat[i][j] = U_RIGHT;
            } else if (i == 0 && my_rank == 1) {
                heat[i][j] = 0;
            } else if (i == height - 1 && (remainder == 0 && my_rank == num_procs - 1 || my_rank == 0)) {
                heat[i][j] = 100;
            } else {
                heat[i][j] = U_0;
            }
        }
    }

    double** v = (double **) malloc(height * sizeof(double *));
    for (int i = 0; i < height; i++) {
        v[i] = (double *) malloc(size_num * sizeof(double));
    }
    double* v0 = (double *) malloc(size_num * sizeof(double *));
    double* v1 = (double *) malloc(size_num * sizeof(double *));

    for (int t = 0; t < time_num; t++) {
        for (int i = 0; i < height; i++) {
            for (int j = 1; j < size_num - 1; j++) {
                v[i][j] = heat[i][j] + k * time_step / (pow(area_step, 2)) * (heat[i][j + 1] - 2 * heat[i][j] + heat[i][j - 1]);
            }
        }
        int offset = my_rank == 1 ? 1 : 0;
        int end_offset = remainder == 0 && my_rank == num_procs - 1 || my_rank == 0 ? 1 : 0;
        if ((my_rank != 0 && remainder != 0) || (remainder == 0 && my_rank != num_procs - 1)) {
            if (my_rank != 0 && my_rank % 2 == 0 || my_rank == 0 && my_rank % 2 == 0) {
                MPI_Send(v[height - 1], size_num, MPI_DOUBLE, (my_rank + 1) % num_procs, 0, MPI_COMM_WORLD);
                MPI_Recv(v1, size_num, MPI_DOUBLE, (my_rank + 1) % num_procs, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(v1, size_num, MPI_DOUBLE, (my_rank + 1) % num_procs, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(v[height - 1], size_num, MPI_DOUBLE, (my_rank + 1) % num_procs, 0, MPI_COMM_WORLD);\
            }
        } else {
            v1 = v[height - 1];
        }
        if (my_rank != 1) {
            if (my_rank != 0 && my_rank % 2 == 0 || my_rank == 0 && my_rank % 2 == 0) {
                MPI_Send(v[0], size_num, MPI_DOUBLE, (num_procs + my_rank - 1) % num_procs, 0, MPI_COMM_WORLD);
                MPI_Recv(v0, size_num, MPI_DOUBLE, (num_procs + my_rank - 1) % num_procs, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(v0, size_num, MPI_DOUBLE, (num_procs + my_rank - 1) % num_procs, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                MPI_Send(v[0], size_num, MPI_DOUBLE, (num_procs + my_rank - 1) % num_procs, 0, MPI_COMM_WORLD);
            }
        } else {
            v0 = v[0];
        }
        for (int i = offset; i < height - end_offset; i++) {
            for (int j = 1; j < size_num - 1; j++) {
                double* v_prev = v0;
                if (i >= 1) {
                    v_prev = v[i - 1];
                }
                double* v_next = v1;
                if (i + 1 < height) {
                    v_next = v[i + 1];
                }
                heat[i][j] = v[i][j] + k * time_step / (pow(area_step, 2)) * (v_next[j] - 2 * v[i][j] + v_prev[j]);
            }
        }
    }
    free(v0);
    free(v1);
    free(v);
}

void GatherResult(int num_procs, int my_rank, double** heat, int size_num, int chunk_height, int height, int remainder, clock_t begin) {
    double * res = NULL;
    if (my_rank == 0) {
        res = (double*) malloc(chunk_height * num_procs * size_num * sizeof(double) * 10);
    }
    double * ptrArray = (double*) malloc(chunk_height * size_num * sizeof(double));
    if (my_rank != 0 || remainder != 0) {
        for (int i = 0; i < height; i++) {
            memcpy(ptrArray + i * size_num, heat[i], size_num * sizeof(double));
        }
    }
    MPI_Gather(ptrArray, chunk_height * size_num, MPI_DOUBLE, res, chunk_height * size_num, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (my_rank == 0) {
        FILE *myfile;
        myfile = fopen("res11.csv", "w+");
        for (int i = chunk_height; i < chunk_height * num_procs; i++) {
            for (int j = 0; j < size_num; j++) {
                fprintf(myfile, "%lf", res[i * size_num + j]);
                if (j != size_num - 1) {
                    fputc(',', myfile);
                }
            }
            fputc('\n', myfile);
        }
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < size_num; j++) {
                fprintf(myfile, "%lf", res[i * size_num + j]);
                if (j != size_num - 1) {
                    fputc(',', myfile);
                }
            }
            fputc('\n', myfile);
        }

        fclose(myfile);
        clock_t end = clock();
        double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        myfile = fopen("time11.csv", "a");
        fprintf(myfile, "%d %lf\n", num_procs, time_spent);
        fclose(myfile);
    }
    if (my_rank == 0) {
        free(res);
    }
    free(ptrArray);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    clock_t begin = clock();
    int my_rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double** heat = NULL;
    double time = 60;
    double time_step = 0.002;
    double area = 0.5;
    double area_step = 0.005;

    int time_num = round(time / time_step);
    int size_num = round(area / area_step);
    int chunk_height = size_num / (num_procs - 1);
    int remainder = size_num % (num_procs - 1);
    int height = chunk_height;
    if (my_rank == 0) {
        height = remainder;
    }
    if (height != 0) {
        heat = (double **) malloc(height * sizeof(double *));
        for (int i = 0; i < height; i++) {
            heat[i] = (double *) malloc(size_num * sizeof(double));
        }
        ComputeHeat(num_procs, my_rank, heat, height, size_num, time_num, remainder, time_step, area_step);
    }
    GatherResult(num_procs, my_rank, heat, size_num, chunk_height, height, remainder, begin);
    if (height != 0) {
        free(heat);
    }
    MPI_Finalize();
    return 0;
}//YjDkpK8ZueE2vA4f