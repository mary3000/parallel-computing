/**
 * MIPT | DIHT | 3 course | Parallel computing
 * Task 3. Integrate (OpenMP)
 *
 * @author Mary Feofanova
 */

#include <iostream>
#include <omp.h>
#include <math.h>
#include <iomanip>

double IntegrateWithCriticalSection(const size_t num_threads, const double a, const double b,
        double (&func)(double),const size_t steps_number) {
    int i = 0;
    double step = (b - a) / steps_number;
    omp_set_num_threads(num_threads);
    double sum = (func(a) + func(b)) / 2 * step;
    double local_sum = 0;
    double begin, end, total;
    begin = omp_get_wtime();
#pragma omp parallel shared(sum) private(i, local_sum)
    {
        local_sum = 0;
#pragma omp for
        for (i = 0; i < steps_number; i++) {
            local_sum += func(a + (i + 1) * step) * step;
        }
#pragma omp critical
        {
            sum += local_sum;
        }
    }
    end = omp_get_wtime();
    total = end - begin;
    FILE *myfile;
    myfile = fopen("time_cs.csv", "a");
    fprintf(myfile, "%zu %lf %lf\n", num_threads, total, sum);
    return sum;
}

double f(double x) {
    return 4 / (1 + std::pow(x, 2));
}

double IntegrateWithReduction(const size_t num_threads, const double a, const double b,
        double (&func)(double), const size_t steps_number) {
    int i = 0;
    double step = (b - a) / steps_number;
    omp_set_num_threads(num_threads);
    double sum = (func(a) + func(b)) / 2 * step;
    double begin, end, total;
    begin = omp_get_wtime();
#pragma omp parallel for reduction(+:sum)
    for (i = 0; i < steps_number; i++) {
        sum += func(a + (i + 1) * step) * step;
    }
    end = omp_get_wtime();
    total = end - begin;
    FILE *myfile;
    myfile = fopen("time_red.csv", "a");
    fprintf(myfile, "%zu %lf %lf\n", num_threads, total, sum);
    return sum;
}

int main() {
#ifdef _OPENMP
    printf("OpenMP is supported! %d \n", _OPENMP);
#endif
    for (size_t i = 1; i <= 12; i++) {
        double sum1 = IntegrateWithCriticalSection(i, 0, 1, f, 1.0e8);
        double sum2 = IntegrateWithReduction(i, 0, 1, f, 1.0e8);
        std::cout << i << " " << std::setprecision(9) << "cs: " << sum1 << std::endl;
        std::cout << i << " " << "red: " << sum2 << std::endl;
    }
    return 0;
}//YjDkpK8ZueE2vA4f