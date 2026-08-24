#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// flatten the 3d grid into 1d array to keep memory contiguous. R is the halo offset  
#define IDX(i, j, k) (((i) + R) * (ny + 2 * R) * (nz + 2 * R) + ((j) + R) * (nz + 2 * R) + ((k) + R))

// handles all the non-blocking MPI comms so the grid doesn't deadlock 
void do_halo_exchange(double **data, int F, int nx, int ny, int nz, int R, 
                      int rank_left, int rank_right, int rank_bottom, int rank_top, int rank_back, int rank_front, 
                      MPI_Comm comm, MPI_Datatype type_x, MPI_Datatype type_y, MPI_Datatype type_z) {
    
    // allocate max possible requests: F fields * 6 faces * 2 (send+recv) = 12
    MPI_Request *reqs = malloc((F * 12) * sizeof(MPI_Request));
    int req_cnt = 0;

    for (int f = 0; f < F; f++) {
        // x direction (left/right)
        if (rank_left != -1) {
            MPI_Irecv(&data[f][IDX(-R, 0, 0)], 1, type_x, rank_left, f*6 + 0, comm, &reqs[req_cnt++]);
            MPI_Isend(&data[f][IDX(0, 0, 0)], 1, type_x, rank_left, f*6 + 1, comm, &reqs[req_cnt++]);
        }
        if (rank_right != -1) {
            MPI_Irecv(&data[f][IDX(nx, 0, 0)], 1, type_x, rank_right, f*6 + 1, comm, &reqs[req_cnt++]);
            MPI_Isend(&data[f][IDX(nx-R, 0, 0)], 1, type_x, rank_right, f*6 + 0, comm, &reqs[req_cnt++]);
        }
        
        // y direction (bottom/top)
        if (rank_bottom != -1) {
            MPI_Irecv(&data[f][IDX(0, -R, 0)], 1, type_y, rank_bottom, f*6 + 2, comm, &reqs[req_cnt++]);
            MPI_Isend(&data[f][IDX(0, 0, 0)], 1, type_y, rank_bottom, f*6 + 3, comm, &reqs[req_cnt++]);
        }
        if (rank_top != -1) {
            MPI_Irecv(&data[f][IDX(0, ny, 0)], 1, type_y, rank_top, f*6 + 3, comm, &reqs[req_cnt++]);
            MPI_Isend(&data[f][IDX(0, ny-R, 0)], 1, type_y, rank_top, f*6 + 2, comm, &reqs[req_cnt++]);
        }
        
        // z direction (back/front)
        if (rank_back != -1) {
            MPI_Irecv(&data[f][IDX(0, 0, -R)], 1, type_z, rank_back, f*6 + 4, comm, &reqs[req_cnt++]);
            MPI_Isend(&data[f][IDX(0, 0, 0)], 1, type_z, rank_back, f*6 + 5, comm, &reqs[req_cnt++]);
        }
        if (rank_front != -1) {
            MPI_Irecv(&data[f][IDX(0, 0, nz)], 1, type_z, rank_front, f*6 + 5, comm, &reqs[req_cnt++]);
            MPI_Isend(&data[f][IDX(0, 0, nz-R)], 1, type_z, rank_front, f*6 + 4, comm, &reqs[req_cnt++]);
        }
    }

    if (req_cnt > 0) MPI_Waitall(req_cnt, reqs, MPI_STATUSES_IGNORE);
    free(reqs);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 13) {
        if (rank == 0) printf("Usage: %s d ppn px py pz nx ny nz T seed F isovalue\n", argv[0]);
        MPI_Finalize(); 
        return 0;
    }

    // parse cmd args
    int d = atoi(argv[1]);
    int ppn = atoi(argv[2]); 
    (void)ppn; // ppn is not used in this implementation but parsed for consistency with the spec
    int px = atoi(argv[3]), py = atoi(argv[4]), pz = atoi(argv[5]);
    int nx = atoi(argv[6]), ny = atoi(argv[7]), nz = atoi(argv[8]);
    int T = atoi(argv[9]), seed = atoi(argv[10]), F = atoi(argv[11]);
    double isovalue = atof(argv[12]);
    
    int R = (d - 1) / 6; 

    // finding my coordinates in the 3D grid manually
    int cx = rank / (py * pz);
    int cy = (rank / pz) % py;
    int cz = rank % pz;

    // figure out who my neighbors are
    int rank_left   = (cx > 0)      ? rank - (py * pz) : -1;
    int rank_right  = (cx < px - 1) ? rank + (py * pz) : -1;
    int rank_bottom = (cy > 0)      ? rank - pz        : -1;
    int rank_top    = (cy < py - 1) ? rank + pz        : -1;
    int rank_back   = (cz > 0)      ? rank - 1         : -1;
    int rank_front  = (cz < pz - 1) ? rank + 1         : -1;

    // memory allocation
    long long local_grid_size = (long long)(nx + 2 * R) * (ny + 2 * R) * (nz + 2 * R);
    double **current_data = (double **)malloc(F * sizeof(double *));
    double **next_data = (double **)malloc(F * sizeof(double *));
    
    for (int f = 0; f < F; f++) {
        current_data[f] = (double *)calloc(local_grid_size, sizeof(double));
        next_data[f] = (double *)calloc(local_grid_size, sizeof(double));
    }

    // fill the array using the exact random formula from the assignment pdf
    int arrSize = nx * ny * nz;
    double *init_buf = (double *)malloc(arrSize * sizeof(double));
    srand(seed);

    for (int f = 0; f < F; f++) {
        for (int j = 0; j < arrSize; j++) {
            init_buf[j] = (double)rand() * (rank + 1) / (110426.0 + f + j);
        }
        int p = 0;
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < nz; k++) {
                    current_data[f][IDX(i, j, k)] = init_buf[p++];
                }
            }
        }
    }
    free(init_buf);

    // creating custom datatypes for Y and Z faces so we dont have to pack/unpack buffers manually
    // this made the -O3 optimization way faster
    MPI_Datatype type_face_x, type_face_y, type_face_z;
    
    int count_x = R * ny;
    int *blen_x = malloc(count_x * sizeof(int)), *disp_x = malloc(count_x * sizeof(int));
    for(int i=0; i<R; i++) {
        for(int j=0; j<ny; j++) {
            blen_x[i*ny + j] = nz;
            disp_x[i*ny + j] = i * (ny+2*R)*(nz+2*R) + j * (nz+2*R);
        }
    }
    MPI_Type_indexed(count_x, blen_x, disp_x, MPI_DOUBLE, &type_face_x);
    MPI_Type_commit(&type_face_x);
    free(blen_x); free(disp_x);

    int count_y = nx * R;
    int *blen_y = malloc(count_y * sizeof(int)), *disp_y = malloc(count_y * sizeof(int));
    for(int i=0; i<nx; i++) {
        for(int j=0; j<R; j++) {
            blen_y[i*R + j] = nz;
            disp_y[i*R + j] = i * (ny+2*R)*(nz+2*R) + j * (nz+2*R);
        }
    }
    MPI_Type_indexed(count_y, blen_y, disp_y, MPI_DOUBLE, &type_face_y);
    MPI_Type_commit(&type_face_y);
    free(blen_y); free(disp_y);

    int count_z = nx * ny;
    int *blen_z = malloc(count_z * sizeof(int)), *disp_z = malloc(count_z * sizeof(int));
    for(int i=0; i<nx; i++) {
        for(int j=0; j<ny; j++) {
            blen_z[i*ny + j] = R;
            disp_z[i*ny + j] = i * (ny+2*R)*(nz+2*R) + j * (nz+2*R);
        }
    }
    MPI_Type_indexed(count_z, blen_z, disp_z, MPI_DOUBLE, &type_face_z);
    MPI_Type_commit(&type_face_z);
    free(blen_z); free(disp_z);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // initial fill of the halos
    do_halo_exchange(current_data, F, nx, ny, nz, R, rank_left, rank_right, rank_bottom, rank_top, rank_back, rank_front, MPI_COMM_WORLD, type_face_x, type_face_y, type_face_z);

    // main simulation loop
    for (int t = 1; t <= T; t++) {
        
        // 1. calculate stencil averages
        for (int f = 0; f < F; f++) {
            for (int i = 0; i < nx; i++) {
                for (int j = 0; j < ny; j++) {
                    for (int k = 0; k < nz; k++) {
                        double sum = current_data[f][IDX(i, j, k)];
                        int count = 1;

                        // check if we are on a global boundary edge. if so, don't divide by 7
                        // FAQ says divide only by the actual number of neighbours we have 
                        for (int r = 1; r <= R; r++) {
                            if (cx * nx + i - r >= 0) { sum += current_data[f][IDX(i - r, j, k)]; count++; }
                            if (cx * nx + i + r < px * nx) { sum += current_data[f][IDX(i + r, j, k)]; count++; }
                            if (cy * ny + j - r >= 0) { sum += current_data[f][IDX(i, j - r, k)]; count++; }
                            if (cy * ny + j + r < py * ny) { sum += current_data[f][IDX(i, j + r, k)]; count++; }
                            if (cz * nz + k - r >= 0) { sum += current_data[f][IDX(i, j, k - r)]; count++; }
                            if (cz * nz + k + r < pz * nz) { sum += current_data[f][IDX(i, j, k + r)]; count++; }
                        }
                        next_data[f][IDX(i, j, k)] = sum / count;
                    }
                }
            }
        }

        // 2. swap for next iter
        double **temp = current_data;
        current_data = next_data;
        next_data = temp;

        // 3. refresh halos for the next step / isovalue counting
        do_halo_exchange(current_data, F, nx, ny, nz, R, rank_left, rank_right, rank_bottom, rank_top, rank_back, rank_front, MPI_COMM_WORLD, type_face_x, type_face_y, type_face_z);

        // 4. count isovalue intersections using marching squares approach
        long long *local_counts = (long long *)calloc(F, sizeof(long long));
        for (int f = 0; f < F; f++) {
            long long isocnt = 0;
            for (int i = 0; i < nx; i++) {
                for (int j = 0; j < ny; j++) {
                    for (int k = 0; k < nz; k++) {
                        double v = current_data[f][IDX(i, j, k)];
                        // checking if the edges cross the isovalue (marching squares from lecture)
                        // check edges in x, y, and z directions
                         // FORWARD CHECKS
                        if (cx * nx + i < px * nx - 1) {
                            double vx = current_data[f][IDX(i + 1, j, k)];
                            if ((v <= isovalue && vx > isovalue) || (v >= isovalue && vx < isovalue)) isocnt++;
                        }
                        if (cy * ny + j < py * ny - 1) {
                            double vy = current_data[f][IDX(i, j + 1, k)];
                            if ((v <= isovalue && vy > isovalue) || (v >= isovalue && vy < isovalue)) isocnt++;
                        }
                        if (cz * nz + k < pz * nz - 1) {
                            double vz = current_data[f][IDX(i, j, k + 1)];
                            if ((v <= isovalue && vz > isovalue) || (v >= isovalue && vz < isovalue)) isocnt++;
                        }
                        // BACKWARD CHECKS
                        // If we are at the start of our local chunk, but NOT the absolute edge of the global grid, 
                        // look backwards into the halo to double-count the boundary shared with the left/bottom/back neighbor.
                        if (i == 0 && cx > 0) {
                            double vx_back = current_data[f][IDX(-1, j, k)];
                            if ((v <= isovalue && vx_back > isovalue) || (v >= isovalue && vx_back < isovalue)) isocnt++;
                        }
                        if (j == 0 && cy > 0) {
                            double vy_back = current_data[f][IDX(i, -1, k)];
                            if ((v <= isovalue && vy_back > isovalue) || (v >= isovalue && vy_back < isovalue)) isocnt++;
                        }
                        if (k == 0 && cz > 0) {
                            double vz_back = current_data[f][IDX(i, j, -1)];
                            if ((v <= isovalue && vz_back > isovalue) || (v >= isovalue && vz_back < isovalue)) isocnt++;
                        }

                    }
                }
            }
            local_counts[f] = isocnt;
        }
        // gather to root
        long long *global_counts = NULL;
        if (rank == 0) global_counts = (long long *)calloc(F, sizeof(long long));
        
        MPI_Reduce(local_counts, global_counts, F, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            for (int f = 0; f < F; f++) {
                printf("%lld", global_counts[f]);
                if (f < F - 1) printf(" ");
            }
            printf("\n");
            fflush(stdout); 
            free(global_counts);
        }
        free(local_counts);
    }

    double local_time = MPI_Wtime() - start_time, max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("%lf\n", max_time);
        fflush(stdout);
    }

   // clean up memory before exit
    for (int f = 0; f < F; f++) { 
        free(current_data[f]); 
        free(next_data[f]); 
    }
    free(current_data); 
    free(next_data);
    MPI_Type_free(&type_face_x); 
    MPI_Type_free(&type_face_y); 
    MPI_Type_free(&type_face_z);
    
    MPI_Finalize();
    return 0;
}