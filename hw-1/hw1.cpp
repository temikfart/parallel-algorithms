#include <iostream>
#include <iomanip>
#include <mpi.h>

inline double f(double x) { return (4.0 / (1 + x * x)); }
double trapezoids_method(double begin, double end, double dx) {
    double res = 0.0;
    int iterations = static_cast<int>((end - begin) / dx);
    for (int i = 0; i < iterations; i++) {
        double x_i = begin + dx * i;
        double x_ip1 = x_i + dx;
        res += (f(x_i) + f(x_ip1)) / 2 * (x_ip1 - x_i);
    }
    return res;
}

struct Result {
    Result() = default;
    Result(double lhs, double rhs, double res) : result(res) {
        range[0] = lhs;
        range[1] = rhs;
    }
    double range[2] = {0.0, 0.0};
    double result = 0;
};
MPI_Datatype create_result_type() {
    // Define the block lengths and displacements for the struct
    int block_lengths[2] = {2, 1};
    MPI_Aint displacements[2] = {0, 2 * sizeof(double)};

    // Define the types of the struct members
    MPI_Datatype types[2] = {MPI_DOUBLE, MPI_DOUBLE};

    // Create the new struct type
    MPI_Datatype ResultT;
    MPI_Type_create_struct(2, block_lengths, displacements, types, &ResultT);

    // Commit the new type to MPI
    MPI_Type_commit(&ResultT);

    return ResultT;
}

void print_answer(const std::string& type, double answer, int precision, double time) {
    std::cout << std::setw(15) << std::left << type;
    std::cout << std::fixed << std::setprecision(precision) << answer << '\t';
    std::cout << std::setw(7) << std::left << "Time";
    std::cout << std::fixed << std::setprecision(6) << time << std::endl;
}

int main(int argc, char* argv[]) {
    // MPI variables
    int size, rank;
    MPI_Status Status;

    // Time tracking
    double begin, end;
    double tbegin, tend;

    // Parameters
    int N = 1e3;
    double dx = 1.0 / N;
    int precision = 6;

    // Calculations variable
    double answer_consistently;
    double range[] = {0.0, 0.0};

    // MPI initialization
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Datatype ResultT = create_result_type();

    // Calculate integral consistently
    if (rank == 0) {
        begin = MPI_Wtime();
        answer_consistently = trapezoids_method(0, 1, dx);
        end = MPI_Wtime();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    tbegin = MPI_Wtime();

    // Divide range into pieces for each process
    if (rank == 0) {
        double step = 1.0 / size;
        for (int i = 1; i < size; i++) {
            range[0] = step * i;
            range[1] = range[0] + step;
            MPI_Send(&range, 2, ResultT, i, 0, MPI_COMM_WORLD);
        }
        range[0] = 0;
        range[1] = step;
    } else {
        MPI_Recv(&range, 2, ResultT, 0, 0, MPI_COMM_WORLD, &Status);
    }

    Result Res = {range[0], range[1], trapezoids_method(range[0], range[1], dx)};

    if (rank == 0) {
        std::cout << "i\tx_i\tx_{i+1}\tI" << std::endl;
        std::cout << std::fixed << std::setprecision(precision);
        std::cout << rank << '\t' << range[0] << '\t' << range[1] << '\t' << Res.result << std::endl;
        for (int i = 1; i < size; i++){
            Result Proc_result;
            MPI_Recv(&Proc_result, 1, ResultT, i, 0, MPI_COMM_WORLD, &Status);
            std::cout << i << '\t' << Proc_result.range[0] << '\t' << Proc_result.range[1] << '\t' << Proc_result.result << std::endl;
            Res.result += Proc_result.result;
        }
        tend = MPI_Wtime();
        print_answer("Parallel", Res.result, precision, tend - tbegin);
        print_answer("Consistently", answer_consistently, precision, end - begin);
    } else {
        MPI_Bsend(&Res, 1, ResultT, 0, 0, MPI_COMM_WORLD);
    }

    // Free the new type when done using it
    MPI_Type_free(&ResultT);
    MPI_Finalize();

    return 0;
}
