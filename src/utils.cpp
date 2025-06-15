#include <stdio.h>
#include "globals.h"

void loadWorkset(bool* workset, const bool* state){
    for (unsigned int i = 0; i < NUM_USERS; ++i){
        for (unsigned int j = 0; j < NUM_ROLES; ++j){
            workset[0 * NUM_USERS * NUM_ROLES + i * NUM_ROLES + j] = state[i * NUM_ROLES + j];
        }
    }
}
