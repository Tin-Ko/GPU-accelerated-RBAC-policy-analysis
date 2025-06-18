#include "utils.h"
#include "globals.h"
#include <stdio.h>
#include <string>

void loadWorkset(bool *workset, const bool *state) {
    for (unsigned int i = 0; i < NUM_USERS; ++i) {
        for (unsigned int j = 0; j < NUM_ROLES; ++j) {
            workset[0 * NUM_USERS * NUM_ROLES + i * NUM_ROLES + j] = state[i * NUM_ROLES + j];
        }
    }
}

std::string getStateAscii(const bool *state) {
    int length = NUM_USERS * NUM_ROLES;

    // Flatten state
    bool flattenedState[length];
    for (unsigned int i = 0; i < NUM_USERS; ++i) {
        for (unsigned int j = 0; j < NUM_ROLES; ++j) {
            flattenedState[i * NUM_ROLES + j] = state[i * NUM_ROLES + j];
        }
    }

    // Convert to Ascii
    unsigned int fullByte = length / 8;
    std::string stateAscii = "";
    for (unsigned int byte = 0; byte < fullByte; ++byte) {
        char ch = 0;
        for (unsigned int bit = 0; bit < 8; ++bit) {
            ch |= flattenedState[byte * 8 + bit] << (7 - bit);
        }
        stateAscii += ch;
    }

    return stateAscii;
}

int getStateHex(const bool *state) {
}
