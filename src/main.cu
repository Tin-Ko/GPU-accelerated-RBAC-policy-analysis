#include <cuda_runtime.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "analysis.cuh"
#include "closure.cuh"
#include "globals.h"
#include "utils.h"

using namespace std;

struct State {
    bool s[NUM_USERS * NUM_ROLES];
};

int main() {
    bool CA[NUM_CA_RULES * 4 * NUM_ROLES] = {
        // rule 0
        1, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0,
        // rule 1
        0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0,
        // rule 2
        1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0,
        // rule 3
        0, 1, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0,
        // rule 4
        0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0};

    bool s[NUM_USERS * NUM_ROLES] = {1, 1, 1, 0, 0, 1, 0, 1,
                                     1, 0, 1, 0, 0, 0, 0, 0,
                                     0, 1, 0, 0, 0, 0, 0, 1,
                                     0, 1, 0, 0, 0, 0, 0, 1,
                                     0, 0, 0, 0, 0, 1, 0, 0};

    bool relPos[NUM_ROLES] = {1, 1, 1, 1, 1, 1, 0, 1};
    bool relNeg[NUM_ROLES] = {0, 0, 1, 0, 0, 0, 0, 0};

    bool worksetIn[MAX_STATES_WORKSET * NUM_USERS * NUM_ROLES] = {};
    // bool worksetOut[MAX_STATES_WORKSET * 5][NUM_USERS][NUM_ROLES] = {};
    bool worksetOut[MAX_STATES_WORKSET * 5 * NUM_USERS * NUM_ROLES];
    int worksetOutIndex = -1;

    int goalUser = 4;
    int goalRole = 5;

    int goalReached = 1;

    loadWorkset(worksetIn, s, 0);

    vector<bool *> States;

    bool *d_relPos;
    bool *d_relNeg;
    bool *d_s;
    bool *d_CA;
    bool *d_worksetIn;
    bool *d_worksetOut;

    int *d_worksetOutIndex;

    int *d_goalReached;

    cudaMalloc(&d_relPos, NUM_ROLES * sizeof(bool));
    cudaMalloc(&d_relNeg, NUM_ROLES * sizeof(bool));

    cudaMalloc(&d_s, NUM_USERS * NUM_ROLES * sizeof(bool));
    cudaMalloc(&d_CA, NUM_CA_RULES * 4 * NUM_ROLES * sizeof(bool));

    cudaMalloc(&d_worksetIn, MAX_STATES_WORKSET * NUM_USERS * NUM_ROLES * sizeof(bool));
    cudaMalloc(&d_worksetOut, MAX_STATES_WORKSET * 5 * NUM_USERS * NUM_ROLES * sizeof(bool));
    cudaMalloc(&d_worksetOutIndex, sizeof(int));

    cudaMalloc(&d_goalReached, sizeof(bool));

    cudaMemcpy(d_relPos, relPos, NUM_ROLES * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_relNeg, relNeg, NUM_ROLES * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, s, NUM_USERS * NUM_ROLES * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_CA, CA, NUM_CA_RULES * 4 * NUM_ROLES * sizeof(bool), cudaMemcpyHostToDevice);

    cudaMemcpy(d_worksetOutIndex, &worksetOutIndex, sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_goalReached, &goalReached, sizeof(int), cudaMemcpyHostToDevice);

    dim3 gridDim(MAX_STATES_WORKSET);
    dim3 blockDim(NUM_USERS, NUM_CA_RULES, NUM_ROLES);

    map<string, int> stateIdMap;
    vector<State> pendingStates;
    int idCounter = 0;

    // for (int n = 0; n < MAX_STATES_WORKSET; ++n) {
    //     for (int i = 0; i < NUM_USERS; i++) {
    //         for (int j = 0; j < NUM_ROLES; j++) {
    //             cout << worksetIn[n * NUM_USERS * NUM_ROLES + i * NUM_ROLES + j] << " ";
    //         }
    //         cout << endl;
    //     }
    // }

    while (!pendingStates.empty()) {
        cudaMemcpy(d_worksetOut, worksetOut, MAX_STATES_WORKSET * 5 * NUM_USERS * NUM_ROLES * sizeof(bool), cudaMemcpyHostToDevice);
        closure_kernel<<<gridDim, blockDim>>>(d_worksetOut, d_CA, d_relPos, d_relNeg, NUM_USERS, NUM_CA_RULES, NUM_ROLES, goalUser, goalRole, d_goalReached);
        cudaMemcpy(worksetOut, d_worksetOut, MAX_STATES_WORKSET * 5 * NUM_USERS * NUM_ROLES * sizeof(bool), cudaMemcpyDeviceToHost);

        for (int i = 0; i < 5 * MAX_STATES_WORKSET; i++) {
            // if ascii not in map then add <ascii, id> into map
            State currentState;
            for (int j = 0; j < NUM_USERS; j++) {
                for (int k = 0; k < NUM_ROLES; k++) {
                    currentState.s[j * NUM_ROLES + k] = worksetOut[i * NUM_USERS * NUM_ROLES + j * NUM_ROLES + k];
                }
            }
            string asciiState = getStateAscii(currentState.s);
            if (stateIdMap.find(asciiState) == stateIdMap.end()) {
                stateIdMap.insert({asciiState, idCounter++});
                pendingStates.push_back(currentState);
            }
        }
        for (int i = 0; i < MAX_STATES_WORKSET; i++) {
            // Assign new worksetIn
            for (int j = 0; j < NUM_USERS * NUM_ROLES; ++j) {
                worksetIn[i * NUM_USERS * NUM_ROLES + j] = pendingStates[i].s[j];
            }
            pendingStates.erase(pendingStates.begin());
        }
        cudaMemcpy(d_worksetIn, worksetIn, MAX_STATES_WORKSET * NUM_USERS * NUM_ROLES * sizeof(bool), cudaMemcpyHostToDevice);
        analysis_kernel<<<gridDim, blockDim>>>(d_worksetIn, d_worksetOut, d_worksetOutIndex, d_CA, d_relPos, d_relNeg, NUM_USERS, NUM_CA_RULES, NUM_ROLES, goalUser, goalRole, d_goalReached);
        cudaMemcpy(worksetOut, d_worksetOut, MAX_STATES_WORKSET * 5 * NUM_USERS * NUM_ROLES * sizeof(bool), cudaMemcpyDeviceToHost);
    }

    // closure_kernel<<<gridDim, blockDim>>>(d_s, d_CA, d_relPos, d_relNeg, NUM_USERS, NUM_CA_RULES, NUM_ROLES);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cout << "CUDA kernel launch failed: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    cudaDeviceSynchronize();

    cudaMemcpy(&goalReached, d_goalReached, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Goal Reached: %d", goalReached);

    cudaMemcpy(s, d_s, NUM_USERS * NUM_ROLES * sizeof(bool), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < NUM_USERS; i++) {
    //     for (int j = 0; j < NUM_ROLES; j++) {
    //         cout << s[i * NUM_ROLES + j] << " ";
    //     }
    //     cout << endl;
    // }

    cudaFree(d_relPos);
    cudaFree(d_relNeg);
    cudaFree(d_s);
    cudaFree(d_CA);
    cudaFree(d_worksetIn);
    cudaFree(d_worksetOut);
    cudaFree(d_worksetOutIndex);

    return 0;
}
