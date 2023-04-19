#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <cmath>

#ifndef NUMBER
#   define NUMBER 1e3
#endif // NUMBER

#ifndef PRECISION
#   define PRECISION 6
#endif // PRECISION

inline double f(double x) { return (4.0 / (1 + x * x)); }
double trapezoids_method2(int begin, int end, int N) {
    double res = 0.0;
    double dx = 1.0 / N;
    int steps = end - begin;
    for (int i = 0; i < steps; i++)
        res += (f((begin + i) * dx) + f((begin + i + 1) * dx)) / (2 * N);
    return res;
}

int main(int argc, char* argv[]) {
    int size, rank;
    MPI_Status Status;

    double begin, end;
    double tbegin, tend;

    int N = NUMBER;
    int prec = PRECISION;

    int range[] = {0, 0};

    double consistently = 0.0;
    double parallel = 0.0;
    double intermediate_result = 0.0;

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        begin = MPI_Wtime();
        consistently = trapezoids_method2(0, N, N);
        end = MPI_Wtime();
    }
    MPI_Barrier(MPI_COMM_WORLD);

    tbegin = MPI_Wtime();

    if (rank == 0) {
        int number = static_cast<int>(trunc(N / size));
        int diff = N % size;

        range[1] = 0;
        for (int i = 1; i < size; i++) {
            range[0] = range[1];
            range[1] += number;
            if (diff != 0) {
                range[1]++;
                diff--;
            }
            MPI_Send(&range, 2, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        range[0] = range[1];
        range[1] = N;
    } else {
        MPI_Recv(&range, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, &Status);
    }

    intermediate_result = trapezoids_method2(range[0], range[1], N);

    if (rank == 0) {
        parallel += intermediate_result;
        std::cout << rank << ": " << intermediate_result << std::endl;

        double received_result;
        for (int i = 1; i < size; i++) {
            MPI_Recv(&received_result, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &Status);
            std::cout << i << ": " << received_result << std::endl;
            parallel += received_result;
        }
        tend = MPI_Wtime();
    } else {
        MPI_Bsend(&intermediate_result, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        double cons_duration = end - begin;
        std::cout << std::setprecision(prec) << std::fixed
                  << "Time: " << cons_duration << " "
                  << std::setprecision(prec) << std::fixed
                  << "Consistent result: " << consistently << std::endl;

        double par_duration = tend - tbegin;
        std::cout << std::setprecision(prec) << std::fixed
                  << "Time: " << par_duration << " "
                  << std::setprecision(prec) << std::fixed
                  << "Parallel result:   " << parallel << std::endl;

        double acceleration = cons_duration / par_duration;
        std::cout << std::setprecision(prec) << std::fixed
                  << "Acceleration: " << acceleration << std::endl;
    }

    MPI_Finalize();
    return 0;
}
