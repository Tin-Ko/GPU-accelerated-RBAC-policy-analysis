#include <iostream>
#include <cuda_runtime.h>


#define NUM_USERS 5
#define NUM_ROLES 8
#define NUM_CA_RULES 5


__global__ void closure_kernel(bool* s, bool* relCanAssign, bool* relPos, bool* relNeg, int numUsers, int numRules, int numRoles) {
    // int tid = threadIdx.x;
    int user = threadIdx.x + 1;   // Starts from user 1
    int rule = threadIdx.y;

    if (user >= numUsers || rule >= numRules) {
        return;
    }

    __shared__ int cond2Flag[NUM_USERS * NUM_CA_RULES];
    __shared__ int cond3Flag[NUM_USERS * NUM_CA_RULES];
    __shared__ int cond4Flag[NUM_USERS * NUM_CA_RULES];

    if (threadIdx.z == 0) cond2Flag[user * numRules + rule] = 0;
    if (threadIdx.z == 0) cond3Flag[user * numRules + rule] = 0;
    if (threadIdx.z == 0) cond4Flag[user * numRules + rule] = 0;

    __syncthreads();

    int role = threadIdx.z;

    bool cond1 = false;
    bool cond2 = true;
    bool cond3 = true;
    bool cond4 = false;

    // Cond1 and Cond2
    if (role < numRoles) {
        cond1 = (relCanAssign[numRoles * 4 * rule + numRoles * 3  + role] && (relPos[role] && !relNeg[role]));
    }
    
    // Cond2
    if (role < numRoles && (!relCanAssign[numRoles * 4 * rule + numRoles + role] || s[user * numRoles + role]) == 0) {
        atomicExch(&cond2Flag[user * numRules + rule], 1);
    }


    // Cond3
    if (role < numRoles && (s[user * numRoles + role] && relCanAssign[numRoles * 4 * rule + numRoles * 2 + role])) {
        atomicExch(&cond3Flag[user * numRules + rule], 1);
    }

    
    // Cond4
    if (role < numRoles && s[role] && relCanAssign[numRoles * 4 * rule + role]) {
        atomicExch(&cond4Flag[user * numRules + rule], 1);
    }

    __syncthreads();

    if (cond2Flag[user * numRules + rule] == 1) cond2 = false;
    if (cond3Flag[user * numRules + rule] == 1) cond3 = false;
    if (cond4Flag[user * numRules + rule] == 1) cond4 = true;


    bool allCond = cond1 && cond2 && cond3 && cond4;

    if (allCond) {
        s[user * numRoles + role] = s[user * numRoles + role] || allCond;
        s[role] = s[role] || allCond;
    }
    
    __syncthreads();

}
