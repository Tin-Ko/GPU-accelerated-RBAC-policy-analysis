import numpy as np

r1 = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
    ],
    dtype=bool,
)
r2 = np.array(
    [
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
    ],
    dtype=bool,
)
r3 = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
    ],
    dtype=bool,
)
r4 = np.array(
    [
        [0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
    ],
    dtype=bool,
)
r5 = np.array(
    [
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ],
    dtype=bool,
)
r6 = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]], dtype=bool)
r7 = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0]], dtype=bool)
r8 = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0]], dtype=bool)
assignRules = np.array([r1, r2, r3, r4, r5])
revokeRules = np.array([r6, r7, r8])
goal = np.array([0, 0, 0, 0, 1, 0, 0, 0], dtype=bool)
relPos = w = seen = np.copy(goal)
relNeg = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
relRules = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
offset = len(assignRules)

while not np.all(w == False):
    # can_assign
    for i, rule in enumerate(assignRules):
        if not np.all(np.bitwise_and(rule[3], w) == False):
            index = np.argmax(rule[3] != 0)
            w[index] = False
            w = np.copy(w)
            seen[index] = True
            seen = np.copy(seen)
            relRules[i] = True
            relRules = np.copy(relRules)
            relNeg = np.copy(np.bitwise_or(rule[2], relNeg))
            relPos = np.copy(np.bitwise_or(rule[1], relPos))
            relPos = np.copy(np.bitwise_or(rule[0], relPos))

    # can_revoke
    for i, rule in enumerate(revokeRules):
        if not np.all(np.bitwise_and(rule[1], relNeg) == False):
            relRules[offset + i] = True
            relRules = np.copy(relRules)
            relPos = np.copy(np.bitwise_or(rule[0], relPos))

    # update w
    if np.all(w == np.bitwise_or(w, np.bitwise_and(relPos, np.bitwise_not(seen)))):
        break
    w = np.copy(np.bitwise_or(w, np.bitwise_and(relPos, np.bitwise_not(seen))))

print("rel+:     ", relPos)
print("rel-:     ", relNeg)
print("relRules: ", relRules)
