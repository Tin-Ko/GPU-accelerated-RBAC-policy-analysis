#pragma once

__global__ void analysis_kernel(bool *worksetIn, bool *worksetOut,
                                bool *relCanAssign, bool *relPos, bool *relNeg,
                                int numUsers, int numRules, int numRoles);