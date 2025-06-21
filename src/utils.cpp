#include "utils.h"
#include "globals.h"
#include <stdio.h>
#include <string>

void loadWorkset(bool *workset, bool *state, int worksetIndex) {
    for (unsigned int i = 0; i < NUM_USERS; ++i) {
        for (unsigned int j = 0; j < NUM_ROLES; ++j) {
            workset[worksetIndex * NUM_USERS * NUM_ROLES + i * NUM_ROLES + j] = state[i * NUM_ROLES + j];
        }
    }
}

std::string getStateAscii(bool *state) {
    int length = NUM_USERS * NUM_ROLES;

    unsigned int fullByte = length / 8;
    std::string stateAscii = "";
    for (unsigned int byte = 0; byte < fullByte; ++byte) {
        char ch = 0;
        for (unsigned int bit = 0; bit < 8; ++bit) {
            ch |= state[byte * 8 + bit] << (7 - bit);
        }
        stateAscii += ch;
    }

    return stateAscii;
}
