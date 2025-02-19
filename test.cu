#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#define NUM_USERS 8
#define NUM_ROLES 5
#define NUM_CA_RULES 5

__global__ void closure_kernel(bool* s, bool* relCanAssign, bool* relPos, bool* relNeg, int numUsers, int numRules, int numRoles) {

    int idx = threadIdx.x * blockDim.y + threadIdx.y;

    // for all rules
    for(int i = 0; i < numRules; i++){
        // index = 32 * i + idx
        // printf("CA[%d]: %d\n", 32*i + idx, relCanAssign[32*i + idx]);
        // for all users
        for(int j = 1; j <= numUsers; j++){
            
            bool* cond1Result = new bool[numRoles];
            bool* cond2Result = new bool[numRoles];
            bool* cond3Result = new bool[numRoles];
            bool* cond4Result = new bool[numRoles];
            bool* allConditions = new bool[numRoles];

            // Cond1
            if(threadIdx.x == 3){
                cond1Result[idx%8] = (relCanAssign[idx + 32*i] && (relPos[idx%8] && !relNeg[idx%8]));
            }

            // Cond2
            if(threadIdx.x == 1){
                int ctl_1 = 0;
                if((!relCanAssign[idx + 32*i] || s[j*8 + idx%8]) == 0){
                    ctl_1 = 1;
                }
                if(ctl_1){
                    cond2Result[idx%8] = 0;
                }
                else{
                    cond2Result[idx%8] = 1;
                }
            }

            // Cond3
            if(threadIdx.x == 2){
                int ctl_2 = 1;
                if((s[j*8 + idx%8] && relCanAssign[idx + 32*i]) == 1){
                    ctl_2 = 0;
                }
                if(ctl_2 == 0){
                    cond3Result[idx%8] = 0;
                }
                else{
                    cond3Result[idx%8] = 1;
                }
            }

            // Cond4
            if(threadIdx.x == 0){
                int ctl_3 = 0;
                if((s[idx%8] && relCanAssign[idx + 32*i]) == 1){
                    ctl_3 = 1;
                }
                if(ctl_3){
                    cond4Result[idx%8] = 1;
                }
                else{
                    cond4Result[idx%8] = 0;
                }
            }

            __syncthreads();

            // All condition combined
            if(threadIdx.x == 0){
                allConditions[idx%8] = cond1Result[idx%8] && cond2Result[idx%8] && cond3Result[idx%8] && cond4Result[idx%8];
                // printf("All Conditions[%d]: %d\n", idx%8, allConditions[idx%8]);
                s[idx%8 + j*8] = s[idx%8 + j*8] || allConditions[idx%8];
                s[idx%8] = s[idx%8] || allConditions[idx%8];

            }

            delete[] cond1Result;
            delete[] cond2Result;
            delete[] cond3Result;
            delete[] cond4Result;
            delete[] allConditions;
        }
    }
}

int main() {
    bool CA[NUM_CA_RULES][4][NUM_USERS] = {
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

    bool s[NUM_CA_RULES][NUM_USERS] = {
        {1, 1, 1, 0, 0, 1, 0, 1},
        {1, 0, 1, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 1},
        {0, 1, 0, 0, 0, 0, 0, 1},
        {0, 0, 0, 0, 0, 1, 0, 0}
    };

    bool relPos[NUM_USERS] = {1, 1, 1, 1, 1, 1, 0, 1};
    bool relNeg[NUM_USERS] = {0, 0, 1, 0, 0, 0, 0, 0};

    bool* d_relPos;
    bool* d_relNeg;
    bool* d_s;
    bool* d_CA;

    cudaMalloc(&d_relPos, NUM_USERS * sizeof(bool));
    cudaMalloc(&d_relNeg, NUM_USERS * sizeof(bool));
    cudaMalloc(&d_s, NUM_CA_RULES * NUM_USERS * sizeof(bool));
    cudaMalloc((void**)&d_CA, NUM_CA_RULES * 4 * NUM_USERS * sizeof(bool));

    cudaMemcpy(d_relPos, relPos, NUM_USERS * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_relNeg, relNeg, NUM_USERS * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, s, NUM_CA_RULES * NUM_USERS * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_CA, CA, NUM_CA_RULES * 4 * NUM_USERS * sizeof(bool), cudaMemcpyHostToDevice);

    dim3 gridDim(1, 1);
    dim3 blockDim(4, 8);

    closure_kernel<<<gridDim, blockDim>>>(d_s, d_CA, d_relPos, d_relNeg, NUM_USERS, NUM_CA_RULES, NUM_ROLES);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cout << "CUDA kernel launch failed: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    cudaDeviceSynchronize();

    cudaMemcpy(s, d_s, NUM_CA_RULES * NUM_USERS * sizeof(bool), cudaMemcpyDeviceToHost);

    for (int i = 0; i < NUM_CA_RULES; i++) {
        for (int j = 0; j < NUM_USERS; j++) {
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
