#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <mpi.h>

#define N_TEMP_POINTS 11

#ifndef NUMBER
#   define NUMBER 2000
#endif

#ifndef MODE
#   define MODE 0
#endif

#ifndef PRECISION
#   define PRECISION 6
#endif

#ifndef MAX_TEMP
#   define MAX_TEMP 1e-4
#endif

double exact_solution_point(double x, double t) {
    double temp = 0;
    for (int m = 0; m <= 50000; m++)
        temp += (4 / M_PI) * exp(-pow(M_PI, 2) * t * pow(2 * m + 1, 2))
            / (2 * m + 1) * sin(M_PI * x * (2 * m + 1));
    return temp;
}
std::vector<double> exact_solution(int N, double T) {
    std::vector<double> temperatures(N_TEMP_POINTS);
    int number = static_cast<int>(trunc(N / 10));
    int rem = N - number * 10;
    int step = 0;

    temperatures[0] = 0;
    for (int i = 1; i < N_TEMP_POINTS - 1; i++) {
        step += number;
        if (rem != 0) {
            rem--;
            step++;
        }
        temperatures[i] = exact_solution_point(1.0 / N * step, T);
    }
    temperatures[N_TEMP_POINTS - 1] = 0;

    return temperatures;
}

double next(double left, double center, double right, double coef) {
    return center + coef * (left - 2 * center + right);
};
void calc_next(double range[2], std::vector<double>& steps, double c) {
    int sz = (int) steps.size();
    double tmp1;
    double tmp2 = steps[0];

    steps[0] = next(range[0], steps[0], steps[1], c);
    for (int i = 1; i < sz - 1; i++) {
        tmp1 = steps[i];
        steps[i] = next(tmp2, steps[i], steps[i + 1], c);
        tmp2 = tmp1;
    }
    steps[sz - 1] = next(tmp2, steps[sz - 1], range[1], c);
}

void slow(int iterations, double range[2], std::vector<double>& steps, int size, double c,
          MPI_Status& Status, int rank) {
    if (rank == 0)
        std::cout << "========== SLOW MODE ==========" << std::endl;

    int proc_sz = (int) steps.size();
    for (int i = 0; i < iterations; i++) {
        calc_next(range, steps, c);

        if (rank > 0)
            MPI_Send(&steps[0], 1, MPI_DOUBLE, rank - 1, 2, MPI_COMM_WORLD);
        if (rank < size - 1)
            MPI_Recv(&range[1], 1, MPI_DOUBLE, rank + 1, 2, MPI_COMM_WORLD, &Status);

        if (rank < size - 1)
            MPI_Send(&steps[proc_sz - 1], 1, MPI_DOUBLE, rank + 1, 3, MPI_COMM_WORLD);
        if (rank > 0)
            MPI_Recv(&range[0], 1, MPI_DOUBLE, rank - 1, 3, MPI_COMM_WORLD, &Status);
    }
}
void fast(int iterations, double range[2], std::vector<double>& steps, int size, double c,
          MPI_Status& Status, int rank) {
    if (rank == 0)
        std::cout << "========== FAST MODE ==========" << std::endl;

    int proc_sz = (int) steps.size();
    for (int i = 0; i < iterations; i++) {
        calc_next(range, steps, c);

        if (rank % 2 == 0) {
            if (rank > 0)
                MPI_Send(&steps[0], 1, MPI_DOUBLE, rank - 1, rank - 1, MPI_COMM_WORLD);
            if (rank > 0)
                MPI_Recv(&range[0],1,MPI_DOUBLE,rank - 1,rank,MPI_COMM_WORLD,&Status);
            if (rank < size - 1)
                MPI_Recv(&range[1],1,MPI_DOUBLE,rank + 1,rank,MPI_COMM_WORLD,&Status);
            if (rank < size - 1)
                MPI_Send(&steps[proc_sz - 1],1,MPI_DOUBLE,rank + 1,rank + 1,MPI_COMM_WORLD);
        } else {
            if (rank < size - 1)
                MPI_Recv(&range[1],1,MPI_DOUBLE,rank + 1,rank,MPI_COMM_WORLD,&Status);
            if (rank < size - 1)
                MPI_Send(&steps[proc_sz - 1],1,MPI_DOUBLE,rank + 1,rank + 1,MPI_COMM_WORLD);
            if (rank > 0)
                MPI_Send(&steps[0], 1, MPI_DOUBLE, rank - 1, rank - 1, MPI_COMM_WORLD);
            if (rank > 0)
                MPI_Recv(&range[0],1,MPI_DOUBLE,rank - 1,rank,MPI_COMM_WORLD,&Status);
        }
    }
}

int distribute_ranges(double range[2], int rank, MPI_Status& Status, std::vector<int>& proc_szs) {
    int proc_sz = 0;
    int size = (int) proc_szs.size();
    if (rank == 0) {
        int number = static_cast<int>(trunc((NUMBER - 1) / size));
        int rem = (NUMBER - 1) % size;
        for (int i = size - 1; i > 0; i--) {
            proc_sz = number;
            if (rem != 0) {
                proc_sz++;
                rem--;
            }
            proc_szs[i] = proc_sz;
            MPI_Send(&proc_sz, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
        }

        proc_sz = number;
        proc_szs[0] = proc_sz;
        range[0] = 0;
        range[1] = 1;
        if (size == 1)
            range[1] = 0;
    } else {
        MPI_Recv(&proc_sz, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &Status);
        range[0] = range[1] = 1;
        if (rank == size - 1)
            range[1] = 0;
    }
    return proc_sz;
}

void calc_full_temperatures(std::vector<double>& full_temperatures, MPI_Status& Status,
                            std::vector<int>& proc_szs, std::vector<double>& steps) {
    int size = (int) proc_szs.size();
    for (int j = 0; j < proc_szs[0]; j++)
        full_temperatures.push_back(steps[j]);

    for (int i = 1; i < size; i++) {
        std::vector<double> process_answer(proc_szs[i]);
        MPI_Recv(&process_answer[0], proc_szs[i], MPI_DOUBLE, i, 6, MPI_COMM_WORLD, &Status);
        for (int j = 0; j < proc_szs[i]; j++)
            full_temperatures.push_back(process_answer[j]);
    }
}

void calc_temperatures(std::vector<double>& temperatures, std::vector<double>& full_temperatures) {
    temperatures[0] = 0;
    int number = static_cast<int>(trunc(NUMBER / (N_TEMP_POINTS - 1)));
    int rem = NUMBER % (N_TEMP_POINTS - 1);
    int step = -1;

    for (int i = 1; i < (N_TEMP_POINTS - 1); i++) {
        step += number;
        if (rem != 0) {
            step++;
            rem--;
        }
        temperatures[i] = full_temperatures[step];
    }
    temperatures[N_TEMP_POINTS - 1] = 0;
}

void print_results(double begin, double end, int prec, std::vector<double>& temperatures,
                   std::vector<double>& ex_solution) {
    std::cout << "Exact solution:" << std::endl;
    for (int i = 0; i < N_TEMP_POINTS; i++)
        std::cout << std::setprecision(prec) << std::fixed << ex_solution[i] << " ";
    std::cout << std::endl;

    std::cout << "Approximation solution:" << std::endl;
    for (int i = 0; i < N_TEMP_POINTS; i++)
        std::cout << std::setprecision(prec) << std::fixed << temperatures[i] << " ";
    std::cout << std::endl;

    std::cout << std::endl;

    double error = 0.0;
    std::cout << std::setw(20) << std::left << "Mean squared error: ";
    for (int i = 0; i < N_TEMP_POINTS; i++)
        error += pow(temperatures[i] - ex_solution[i], 2);
    std::cout << std::setprecision(prec) << std::fixed << std::setw(4 + prec) << std::right
              << pow(error, 0.5) << std::endl;

    std::cout << std::setw(20) << std::left << "Time: "
              << std::setprecision(prec) << std::fixed << std::setw(4 + prec) << std::right
              << (end - begin) << std::endl;
}

int main(int argc, char* argv[]) {
    int rank, size;
    int N = NUMBER;
    double T = MAX_TEMP;
    double h = 1.0 / N;
    double dt = h * h / 2;
    double c = dt / (h * h);
    double range[2];
    double begin, end;
    int prec = PRECISION;
    MPI_Status Status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    begin = MPI_Wtime();

    std::vector<int> proc_szs(size, 0);
    int proc_sz = distribute_ranges(range, rank, Status, proc_szs);
    std::vector<double> steps(proc_sz, 1);

    int iterations = static_cast<int>(floor(T / dt));
    if (MODE)
        fast(iterations, range, steps, size, c, Status, rank);
    else
        slow(iterations, range, steps, size, c, Status, rank);

    if (rank == 0) {
        std::vector<double> full_temperatures;
        calc_full_temperatures(full_temperatures, Status, proc_szs, steps);

        std::vector<double> temperatures(N_TEMP_POINTS);
        calc_temperatures(temperatures, full_temperatures);

        end = MPI_Wtime();

        std::vector<double> ex_solution = exact_solution(N, T);
        print_results(begin, end, prec, temperatures, ex_solution);
    } else {
        MPI_Send(&steps[0], proc_sz, MPI_DOUBLE, 0, 6, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
