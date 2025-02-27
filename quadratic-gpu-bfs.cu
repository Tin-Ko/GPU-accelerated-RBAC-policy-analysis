//#include <__clang_cuda_builtin_vars.h>
//#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>

#include <driver_types.h>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define NUM_NODES 5
#define NUM_EDGES 5
__global__ void CUDA_BFS_KERNEL(int *d_vertices, int *d_edges, bool *d_frontier, bool *d_visited, int *d_distances, bool *d_done) {
    int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    printf("threadID : %d, d_frontier[%d] : %d, d_visited[%d] : %d\n", threadID, threadID, d_frontier[threadID], threadID);
    if (threadID > NUM_NODES) {
        *d_done = false;
    }

    if (d_frontier[threadID] == true && d_visited[threadID] == false) {
        d_frontier[threadID] = false;
        d_visited[threadID] = true;
        __syncthreads();

        int start = d_vertices[threadID];
        int end = start + d_vertices[threadID + 1] - d_vertices[threadID];
        for (int i = start; i < end; i++) {
            int neighborID = d_edges[i];
            if (d_visited[neighborID] == false) {
                printf("threadID : %d, neighborID : %d\n", threadID, neighborID);
                d_distances[neighborID] = d_distances[threadID] + 1;
                d_frontier[neighborID] = true;
                *d_done = false;
            }
        }
    }
}


int main() {
    int h_vertices[NUM_NODES];
    int h_edges[NUM_EDGES];

    h_vertices[0] = 0;
    h_vertices[1] = 2;
    h_vertices[2] = 3;
    h_vertices[3] = 4;
    h_vertices[4] = 4;

    h_edges[0] = 1;
    h_edges[1] = 2;
    h_edges[2] = 4;
    h_edges[3] = 3;
    h_edges[4] = 4;

    bool h_frontier[NUM_NODES] = { false };
    bool h_visited[NUM_NODES] = { false };
    int h_distances[NUM_NODES] = { 0 };

    int source = 0;
    h_frontier[source] = true;

    int* d_vertices;
    cudaMalloc((void**)&d_vertices, sizeof(int)*NUM_NODES);
    cudaMemcpy(d_vertices, h_vertices, sizeof(int)*NUM_NODES, cudaMemcpyHostToDevice);

    int* d_edges;
    cudaMalloc((void**)&d_edges, sizeof(int)*NUM_EDGES);
    cudaMemcpy(d_edges, h_edges, sizeof(int)*NUM_EDGES, cudaMemcpyHostToDevice);

    bool* d_frontier;
    cudaMalloc((void**)&d_frontier, sizeof(bool)*NUM_NODES);
    cudaMemcpy(d_frontier, h_frontier, sizeof(bool)*NUM_NODES, cudaMemcpyHostToDevice);

    bool* d_visited;
    cudaMalloc((void**)&d_visited, sizeof(bool)*NUM_NODES);
    cudaMemcpy(d_visited, h_visited, sizeof(bool)*NUM_NODES, cudaMemcpyHostToDevice);

    int* d_distances;
    cudaMalloc((void**)&d_distances, sizeof(int)*NUM_NODES);
    cudaMemcpy(d_distances, h_distances, sizeof(int)*NUM_NODES, cudaMemcpyHostToDevice);


    int num_blocks = 1;
    int threads = 5;

    bool h_done = false;
    bool *d_done;
    cudaMalloc((void**)&d_done, sizeof(bool));
    int count = 0;

    while (!h_done) {
        h_done = true;
        cudaMemcpy(d_done, &h_done, sizeof(bool), cudaMemcpyHostToDevice);
        CUDA_BFS_KERNEL <<<num_blocks, threads>>>(d_vertices, d_edges, d_frontier, d_visited, d_distances, d_done);
        cudaMemcpy(&h_done, d_done, sizeof(bool), cudaMemcpyDeviceToHost);
    }


    cudaMemcpy(h_distances, d_distances, sizeof(int)*NUM_NODES, cudaMemcpyDeviceToHost);

    printf("\nDistance: ");

    for (int i = 0; i < NUM_NODES; i++) {
        printf("%d, ", h_distances[i]);
    }

    printf("\n");
    std::cin.get();

}
