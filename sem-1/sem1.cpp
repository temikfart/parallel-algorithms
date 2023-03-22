#include <iostream>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int i;
    int array[10];
    int myrank, size;
    MPI_Status Status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    std::cout << "I am " << myrank << " of " << size << std::endl;

    double begin, end, total;
    MPI_Barrier(MPI_COMM_WORLD);
    begin = MPI_Wtime();

    if (myrank == 0) {
        for (i = 0; i < 10; i++) {
            array[i] = i;
        }
        MPI_Send(&array[5], 5, MPI_INT, 1, 1, MPI_COMM_WORLD);
    }
    if (myrank == 1) {
        MPI_Recv(array, 5, MPI_INT, 0, 1 , MPI_COMM_WORLD, &Status);

        for (i = 0; i < 5; i++) {
            std::cout << array[i] << " ";
        }
        std::cout << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    total = end - begin;

    std::cout << std::fixed;
    std::cout << "(TIME) " << myrank << ": " << total << std::endl;

    MPI_Finalize();

    return 0;
}
