#pragma once

__global__ void closure_kernel(bool *workset,
                               bool *relCanAssign, bool *relPos, bool *relNeg,
                               int numUsers, int numRules, int numRoles,
                               int goalUser, int goalRole, int *goalReached);