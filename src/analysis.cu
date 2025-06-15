#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "closure.cuh"
#include "globals.h"
#include "utils.h"

using namespace std;



// __global__ void analysis_kernel(
//     bool* states,   // current existing states
//     bool* worksetIn, bool* worksetOut, int* worksetIdx, int* worksetSize,
//     bool* s, bool* relCanAssign, bool* relPos, bool* relNeg,
//     int numUsers, int numRules, int numRoles
// ) {
//     int user = threadIdx.x + 1;
//     int rule = threadIdx.y;
//     int stateIndex = blockIdx.x;    // This thread deals with worksetIn[stateIndex]
//
//     if (user >= numUsers || rule >= numRules) {
//         return;
//     }
//
//     __shared__ bool
//
//     __shared__ int cond2Flag[NUM_USERS * NUM_CA_RULES];
//     __shared__ int cond3Flag[NUM_USERS * NUM_CA_RULES];
//     __shared__ int cond4Flag[NUM_USERS * NUM_CA_RULES];
//
//     if (threadIdx.z == 0) cond2Flag[user * numRules + rule] = 0;
//     if (threadIdx.z == 0) cond3Flag[user * numRules + rule] = 0;
//     if (threadIdx.z == 0) cond4Flag[user * numRules + rule] = 0;
//
//     __syncthreads();
//
//     int role = threadIdx.z;
//
//     bool cond1 = false;
//     bool cond2 = false;
//     bool cond3 = false;
//     bool cond4 = false;
//
//     // Cond1 and Cond2
//     if (role < numRoles) {
//         cond1 = (relCanAssign[numRoles * 4 * rule + numRoles * 3 + role] && (relPos[role] && relNeg[role]));
//     }
//
//     // Cond2
//     if (role < numRoles && (!relCanAssign[numRoles * 4 * rule + numRoles + role] || s[user * numRoles + role]) == 0) {
//         atomicExch(&cond2Flag[user * numRules + rule], 1);
//     }
//
//
//     // Cond3
//     if (role < numRoles && (s[user * numRoles + role] && relCanAssign[numRoles * 4 * rule + numRoles * 2 + role])) {
//         atomicExch(&cond3Flag[user * numRules + rule], 1);
//     }
//
//
//     // Cond4
//     if (role < numRoles && s[role] && relCanAssign[numRoles * 4 * rule + role]) {
//         atomicExch(&cond4Flag[user * numRules + rule], 1);
//     }
//
//     __syncthreads();
//
//     if (cond2Flag[user * numRules + rule] == 1) cond2 = false;
//     if (cond3Flag[user * numRules + rule] == 1) cond3 = false;
//     if (cond4Flag[user * numRules + rule] == 1) cond4 = true;
//
//
//     bool allCond = cond1 && cond2 && cond3 && cond4;
//
//     if (allCond) {
//         // Check if state is reached
//         // Add closure(state) to the worksetOut
//     }
//
//     __syncthreads();
//
// }


int main() {
    bool CA[NUM_CA_RULES][4][NUM_ROLES] = {
        {
            {1, 0, 0, 0, 0, 0, 0, 0},
            {0, 1, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 1, 0, 0, 0, 0, 0}
        },
        {
            {0, 0, 0, 0, 0, 1, 0, 0},
            {0, 0, 1, 1, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 1, 0, 0, 0}
        },
        {
            {1, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 1, 0, 0},
            {0, 0, 1, 0, 0, 0, 0, 0},
            {0, 0, 0, 1, 0, 0, 0, 0}
        },
        {
            {0, 1, 0, 0, 0, 0, 0, 0},
            {1, 0, 0, 0, 0, 0, 0, 1},
            {0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 1, 0, 0}
        },
        {
            {0, 1, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 1, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 1, 0}
        }
    };

    bool s[NUM_USERS][NUM_ROLES] = {
        {1, 1, 1, 0, 0, 1, 0, 1},
        {1, 0, 1, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 1},
        {0, 1, 0, 0, 0, 0, 0, 1},
        {0, 0, 0, 0, 0, 1, 0, 0}
    };

    bool relPos[NUM_ROLES] = {1, 1, 1, 1, 1, 1, 0, 1};
    bool relNeg[NUM_ROLES] = {0, 0, 1, 0, 0, 0, 0, 0};

    bool worksetIn[MAX_STATES_WORKSET][NUM_USERS][NUM_ROLES] = {};
    bool worksetOut[MAX_STATES_WORKSET][NUM_USERS][NUM_ROLES] = {};

    //loadWorkset(worksetIn, s);
    loadWorkset(&worksetIn[0][0][0], &s[0][0]);

    vector<bool*> States;

    bool* d_relPos;
    bool* d_relNeg;
    bool* d_s;
    bool* d_CA;
    bool* d_worksetIn;
    bool* d_worksetOut;

    cudaMalloc(&d_relPos, NUM_ROLES * sizeof(bool));
    cudaMalloc(&d_relNeg, NUM_ROLES * sizeof(bool));

    cudaMalloc(&d_s, NUM_USERS * NUM_ROLES * sizeof(bool));
    cudaMalloc((void**)&d_CA, NUM_CA_RULES * 4 * NUM_ROLES * sizeof(bool));

    cudaMemcpy(d_relPos, relPos, NUM_ROLES * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_relNeg, relNeg, NUM_ROLES * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, s, NUM_USERS * NUM_ROLES * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_CA, CA, NUM_CA_RULES * 4 * NUM_ROLES * sizeof(bool), cudaMemcpyHostToDevice);

    dim3 gridDim(1);
    dim3 blockDim(NUM_USERS, NUM_CA_RULES, NUM_ROLES);

    // closure_kernel<<<gridDim, blockDim>>>(d_s, d_CA, d_relPos, d_relNeg, NUM_USERS, NUM_CA_RULES, NUM_ROLES);

    /*
    while (!workSet.empty()) {
        analysis_kernel<<<gridDim, blockDim>>>(d_s, d_CA, d_relPos, d_relNeg, NUM_USERS, NUM_CA_RULES, NUM_ROLES);
        
    }
    */
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cout << "CUDA kernel launch failed: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    cudaDeviceSynchronize();

    cudaMemcpy(s, d_s, NUM_USERS * NUM_ROLES * sizeof(bool), cudaMemcpyDeviceToHost);

    for (int i = 0; i < NUM_USERS; i++) {
        for (int j = 0; j < NUM_ROLES; j++) {
            cout << s[i][j] << " ";
        }
        cout << endl;
    }


    cudaFree(d_relPos);
    cudaFree(d_relNeg);
    cudaFree(d_s);
    cudaFree(d_CA);

    return 0;
}



