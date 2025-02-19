import numpy as np
from numpy.core.multiarray import dtype

CA1 = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
    ],
    dtype=bool,
)
CA2 = np.array(
    [
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
    ],
    dtype=bool,
)
CA3 = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
    ],
    dtype=bool,
)
CA4 = np.array(
    [
        [0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
    ],
    dtype=bool,
)

s = np.array(
    [
        [1, 1, 0, 0, 0],
        [1, 0, 1, 1, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [1, 0, 1, 1, 0],
    ],
    dtype=bool,
)
relCanAssign = np.stack((CA1, CA2, CA3, CA4))
relPos = np.array([1, 1, 1, 1, 1, 1, 0, 1], dtype=bool)
relNeg = np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=bool)


def closure(s, relCanAssign, relPos, relNeg):
    for rule in range(relCanAssign.shape[0]):
        # Every relevant canAssign rule
        CA = relCanAssign[rule, :, :]
        CA = np.copy(CA)
        for user in range(1, s.shape[1]):
            # Every user
            # Cond1 CA[3] and (rel+ and not rel-) = CA[3] or [0 ... 0]
            print(f"CA{rule}, user{user}")
            cond1Result = np.logical_and(
                CA[3], np.logical_and(relPos, np.logical_not(relNeg))
            )
            print(f"cond1Result : \n{cond1Result}")

            # Cond2 not CA[1] or S[user] = [0 ... 0] if 0 in result else [1 ... 1]
            cond2Result = (
                np.zeros(s.shape[0], dtype=bool)
                if np.min(np.logical_or(np.logical_not(CA[1]), s[:, user])) == 0
                else np.ones(s.shape[0], dtype=bool)
            )
            print(f"cond2Result : \n{cond2Result}")

            # Cond3 s[:, user] and CA[2] = [1 ... 1] if result == [0 ... 0] else [0 ... 0]
            cond3Result = (
                np.ones(s.shape[0], dtype=bool)
                if np.max(np.logical_and(s[:, user], CA[2])) == 0
                else np.zeros(s.shape[0], dtype=bool)
            )
            print(f"cond3Result : \n{cond3Result}")

            # Cond4 s[all] and CA[0] = [1 ... 1] if result != [0 ... 0] else [0 ... 0]
            cond4Result = (
                np.ones(s.shape[0], dtype=bool)
                if np.max(np.logical_and(s[:, 0], CA[0])) == 1
                else np.zeros(s.shape[0], dtype=bool)
            )
            print(f"cond4Result : \n{cond4Result}")

            allConditions = np.logical_and(
                np.logical_and(np.logical_and(cond1Result, cond2Result), cond3Result),
                cond4Result,
            )
            print(f"all conditions : \n{allConditions}\n")
            s[:, user] = np.logical_or(s[:, user], allConditions)
            s = np.copy(s)
            s[:, 0] = np.logical_or(s[:, 0], allConditions)
            s = np.copy(s)
    return s


print("state before compute closure")
for i in range(s.shape[0]):
    print(s[i, :])
print("\n")
sClosure = closure(s, relCanAssign, relPos, relNeg)
print(f"sClosure.shape = {sClosure.shape}\n")
print("state after compute closure")
for i in range(sClosure.shape[0]):
    print(sClosure[i, :])
