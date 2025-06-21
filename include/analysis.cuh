#pragma once

__global__ void analysis_kernel(bool *worksetIn, bool *worksetOut, int *worksetOutIndex,
                                bool *relCanAssign, bool *relPos, bool *relNeg,
                                int numUsers, int numRules, int numRoles,
                                int goalUser, int goalRole, int *goalReached);
