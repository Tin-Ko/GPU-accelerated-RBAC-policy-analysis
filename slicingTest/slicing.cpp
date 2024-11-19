#include <ios>
#include <iostream>
#include <map>
#include <string>
#include <set>
#include <vector>

using namespace std;

struct canAssign {
    string rAdmin;
    vector<string> rPos;
    vector<string> rNeg;
    string rTarget;
    bool operator<(const canAssign& other) const {
        if (rAdmin != other.rAdmin) {
            return rAdmin < other.rAdmin;
        }
        return rTarget < other.rTarget;

    }
};

struct canRevoke {
    string rAdmin;
    string rTarget;
    bool operator<(const canRevoke& other) const {
        if (rAdmin != other.rAdmin) {
            return rAdmin < other.rAdmin;
        }
        return rTarget < other.rTarget;
    }
};

void printAssignRule(canAssign);
void printRevokeRule(canRevoke);

int main() {
    vector<canAssign> canAssignRules;
    vector<canRevoke> canRevokeRules;

    vector<string> goals;
    vector<string> W;
    vector<string> relP;
    vector<string> relN;
    set<canAssign> relAssignRules;
    set<canRevoke> relRevokeRules;
    map<string, bool> seen;

    // Assign rules
    canAssignRules.push_back({"r1", {"r2"}, {}, "r3"});
    canAssignRules.push_back({"r6", {"r4", "r3"}, {}, "r5"});
    canAssignRules.push_back({"r1", {"r6"}, {"r3"}, "r4"});
    canAssignRules.push_back({"r2", {"r8", "r1"}, {}, "r6"});
    canAssignRules.push_back({"r2", {"r6"}, {}, "r7"});

    canRevokeRules.push_back({"r1", "r6"});
    canRevokeRules.push_back({"r1", "r3"});
    canRevokeRules.push_back({"r1", "r4"});

    // Assign goals
    goals.push_back("r5");
    W.push_back("r5");
    relP.push_back("r5");


    
    while (!W.empty()) {
        string currentR = W.back();
        W.pop_back();
        // cout << "currentR : " << currentR << endl;
        seen[currentR] = true;

        for (auto r : canAssignRules) {
            if (r.rTarget != currentR) {
                 continue;
            }
            // Add positive role and admin role from canAssign rules to relP and W
            for (string s : r.rPos) {
                if (seen[s] == true) {
                    cout << "seen" << endl;
                    continue;
                }
                relP.push_back(s);
                W.push_back(s);
            }

            if (seen[r.rAdmin] == false) {
                relP.push_back(r.rAdmin);
                W.push_back(r.rAdmin);
            }

            // Add negative role to relN
            for (string s :r.rNeg) {
                relN.push_back(s);
            }

            // Add assign rule to relAssignRules
            relAssignRules.insert(r);

        }

        for (string rN : relN) {
            for (auto r : canRevokeRules) {
                if (r.rTarget != rN) {
                    continue;
                }
                if (seen[r.rAdmin] == false) {
                    relP.push_back(r.rAdmin);

                    // Add admin role from canRevoke rules to W
                    W.push_back(r.rAdmin);

                    // Add revoke rule to relRevokeRules
                }
                // Add admin role from canRevoke rules to relP
                relRevokeRules.insert(r);
            }

        }
        
    }

    cout << "relPos : " << endl;
    for (string s : relP) {
        cout << s << " , ";
    }
    cout << endl;
    cout << "relNeg : " << endl;
    for (string s : relN) {
        cout << s << " , ";
    }
    cout << endl;
    cout << "relAssignRules : " << endl;
    for (auto r : relAssignRules) {
        printAssignRule(r);
    }
    cout << "relRevokeRules : " << endl;
    for (auto r : relRevokeRules) {
        printRevokeRule(r);
    }

}

void printAssignRule(canAssign rule) {
    cout << "rAdmin : " << rule.rAdmin << " , rPos : ";
    for (string s : rule.rPos) {
        cout << s ;
    }
    cout << " , rNeg : ";
    for (string s : rule.rNeg) {
        cout << s;
    }
    cout << " , rTarget : " << rule.rTarget << endl;
}

void printRevokeRule(canRevoke rule) {
    cout << "rAdmin : " << rule.rAdmin << " , rTarget : " << rule.rTarget << endl;
}
