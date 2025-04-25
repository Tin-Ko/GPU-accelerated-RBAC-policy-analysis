#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#define NUM_USERS 5
#define NUM_ROLES 8
#define NUM_CA_RULES 5

__global__ void closure_kernel(bool* s, bool* relCanAssign, bool* relPos, bool* relNeg, int numUsers, int numRules, int numRoles) {

    int idx = threadIdx.x * blockDim.y + threadIdx.y;

    // for all rules
    for (int i = 0; i < numRules; i++) {
        // for all users
        for (int j = 1; j <= numUsers; j++) {
            
            __shared__ bool* cond1Result;
            cond1Result= new bool[numRoles];

            __shared__ bool* cond2Result;
            cond2Result = new bool[numRoles];

            __shared__ bool* cond3Result;
            cond3Result = new bool[numRoles];

            __shared__ bool* cond4Result;
            cond4Result = new bool[numRoles];

            __shared__ bool* allConditions;
            allConditions = new bool[numRoles];

            __shared__ int ctl_1;
            __shared__ int ctl_2;
            __shared__ int ctl_3;

            int currentRole = idx % 8;

            // Cond1
            if (threadIdx.x == 3) {
                cond1Result[currentRole] = 0;
            }
            __syncthreads();


            if (threadIdx.x == 3) {
                cond1Result[currentRole] = (relCanAssign[idx + 32 * i] && (relPos[currentRole] && !relNeg[currentRole]));
            }
            __syncthreads();

            // Cond2
            if (threadIdx.x == 1) {
                cond2Result[currentRole] = 0;
                ctl_1 = 0;
            }
            __syncthreads();

            if (threadIdx.x == 1) {
                if((!relCanAssign[idx + 32 * i] || s[j * 8 + currentRole]) == 0){
                    atomicExch(&ctl_1, 1);
                }
            }
            __syncthreads();

            if (threadIdx.x == 1) {
                if (ctl_1) {
                    cond2Result[currentRole] = 0;
                }
                else {
                    cond2Result[currentRole] = 1;
                }
            }
            __syncthreads();

            // Cond3
            if (threadIdx.x == 2) {
                cond3Result[currentRole] = 0;
                ctl_2 = 1;
            }
            __syncthreads();

            if (threadIdx.x == 2) {
                if ((s[j * 8 + currentRole] && relCanAssign[idx + 32 * i]) == 1) {
                    atomicExch(&ctl_2, 0);
                }
            }
            __syncthreads();

            if (threadIdx.x == 2) {
                if(ctl_2 == 0){
                    cond3Result[currentRole] = 0;
                }
                else{
                    cond3Result[currentRole] = 1;
                }
            }
            __syncthreads();

            // Cond4
            if (threadIdx.x == 0) {
                cond4Result[currentRole] = 0;
                ctl_3 = 0;
            }
            __syncthreads();

            if (threadIdx.x == 0) {
                if((s[currentRole] && relCanAssign[idx + 32 * i]) == 1){
                    atomicExch(&ctl_3, 1);
                }
            }
            __syncthreads();

            if (threadIdx.x == 0) {
                if(ctl_3){
                    cond4Result[currentRole] = 1;
                }
                else{
                    cond4Result[currentRole] = 0;
                }
            }
            __syncthreads();

            if (threadIdx.x == 0) {
                allConditions[currentRole] = 0;
            }
            __syncthreads();

            if (threadIdx.x == 0 && i == 2 && j == 4) {
            }

            // All condition combined
            if (threadIdx.x == 0) {
                allConditions[currentRole] = cond1Result[currentRole] && cond2Result[currentRole] && cond3Result[currentRole] && cond4Result[currentRole];
                s[currentRole + j * 8] = s[currentRole + j * 8] || allConditions[currentRole];
                s[currentRole] = s[currentRole] || allConditions[currentRole];

            }
            __syncthreads();

        }
    }
}

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

    bool* d_relPos;
    bool* d_relNeg;
    bool* d_s;
    bool* d_CA;

    cudaMalloc(&d_relPos, NUM_ROLES * sizeof(bool));
    cudaMalloc(&d_relNeg, NUM_ROLES * sizeof(bool));
    cudaMalloc(&d_s, NUM_USERS * NUM_ROLES * sizeof(bool));
    cudaMalloc((void**)&d_CA, NUM_CA_RULES * 4 * NUM_ROLES * sizeof(bool));

    cudaMemcpy(d_relPos, relPos, NUM_ROLES * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_relNeg, relNeg, NUM_ROLES * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, s, NUM_USERS * NUM_ROLES * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_CA, CA, NUM_CA_RULES * 4 * NUM_ROLES * sizeof(bool), cudaMemcpyHostToDevice);

    dim3 gridDim(1, 1);
    dim3 blockDim(4, 8);

    closure_kernel<<<gridDim, blockDim>>>(d_s, d_CA, d_relPos, d_relNeg, NUM_USERS, NUM_CA_RULES, NUM_ROLES);
    
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
