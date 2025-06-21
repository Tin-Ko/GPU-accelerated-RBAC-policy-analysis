#include <cuda_runtime.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "analysis.cuh"
#include "closure.cuh"
#include "globals.h"

using namespace std;

__global__ void analysis_kernel(bool *worksetIn, bool *worksetOut, int worksetOutIndex,
                                bool *relCanAssign, bool *relPos, bool *relNeg,
                                int numUsers, int numRules, int numRoles) {
    int user = threadIdx.x + 1;
    int rule = threadIdx.y;
    int stateIndex = blockIdx.x; // This thread deals with worksetIn[stateIndex]
    int blockSize = blockDim.x * blockDim.y * blockDim.z;

    int blockStateCounter[blockSize]; // need to be locked

    int stateSize = numUsers * numRoles;

    // should be defined in main function

    if (user >= numUsers || rule >= numRules) {
        return;
    }

    __shared__ int cond2Flag[NUM_USERS * NUM_CA_RULES];
    __shared__ int cond3Flag[NUM_USERS * NUM_CA_RULES];
    __shared__ int cond4Flag[NUM_USERS * NUM_CA_RULES];

    if (threadIdx.z == 0)
        cond2Flag[user * numRules + rule] = 0;
    if (threadIdx.z == 0)
        cond3Flag[user * numRules + rule] = 0;
    if (threadIdx.z == 0)
        cond4Flag[user * numRules + rule] = 0;

    __syncthreads();

    int role = threadIdx.z;

    bool cond1 = false;
    bool cond2 = false;
    bool cond3 = false;
    bool cond4 = false;

    // Cond1 and Cond2
    if (role < numRoles) {
        cond1 = (relCanAssign[numRoles * 4 * rule + numRoles * 3 + role] && (relPos[role] && relNeg[role]));
    }

    // Cond2
    if (role < numRoles && (!relCanAssign[numRoles * 4 * rule + numRoles + role] || worksetIn[stateIndex * stateSize + user * numRoles + role]) == 0) {
        atomicExch(&cond2Flag[user * numRules + rule], 1);
    }

    // Cond3
    if (role < numRoles && (worksetIn[stateIndex * stateSize + user * numRoles + role] && relCanAssign[numRoles * 4 * rule + numRoles * 2 + role])) {
        atomicExch(&cond3Flag[user * numRules + rule], 1);
    }

    // Cond4
    if (role < numRoles && worksetIn[stateIndex * stateSize + role] && relCanAssign[numRoles * 4 * rule + role]) {
        atomicExch(&cond4Flag[user * numRules + rule], 1);
    }

    __syncthreads();

    if (cond2Flag[user * numRules + rule] == 1)
        cond2 = false;
    if (cond3Flag[user * numRules + rule] == 1)
        cond3 = false;
    if (cond4Flag[user * numRules + rule] == 1)
        cond4 = true;

    bool allCond = cond1 && cond2 && cond3 && cond4;

    __syncthreads();

    if (allCond) {
        bool newState[numUsers * numRoles] = worksetIn[stateIndex * stateSize + user * numRoles + role];
        worksetOut[worksetOutIndex * stateSize + user * numRoles + role] = worksetIn[stateIndex * stateSize + user * numRoles + role] || allCond;
        worksetOut[worksetOutIndex * stateSize + role] = worksetIn[stateIndex * stateSize + role] || allCond;
        // int offset = (5 * blockDim.x + blockStateCounter[blockDim.x] * numUsers * numRoles);
        // blockStateCounter[blockDim.x]++;
    }

    __syncthreads();
}