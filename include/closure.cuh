#pragma once

__global__ void closure_kernel(bool* s, bool* relCanAssign, bool *relPos, bool* relNeg, int numUsers, int numRules, int numRoles);
