#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <sstream>
using namespace std;

struct can_assign{
    map<string, vector<int>> admins;
    map<string, vector<int>> posRoles;
    map<string, vector<int>> negRoles;
    map<string, vector<int>> targetRoles;
};

struct can_revoke{
    map<string, vector<int>> admins;
    map<string, vector<int>> targetRoles;
};

struct returnStruct{
    vector<string> relPosRoles;
    vector<string> relNegRoles;
    vector<int> relAssignRules;
    vector<int> relRevokeRules;
};

bool isNotSeen(string target, vector<string> vec);
bool isNotIn(int target, vector<int> vec);
vector<string> splitString(string str, char delimiter);
returnStruct slicing(vector<string> goal, can_assign assign, can_revoke revoke, map<int, vector<string>> assignRules, map<int, vector<string>> revokeRules);
void printReturnStruct(const returnStruct& rs);

int main(){
    can_assign canAssign;
    can_revoke canRevoke;
    canAssign.admins["r1"] = {0, 2, 3};
    canAssign.admins["r3"] = {1};
    canAssign.posRoles["r2"] = {0};
    canAssign.posRoles["r4"] = {1, 3};
    canAssign.posRoles["r3"] = {2, 3};
    canAssign.negRoles["r3"] = {0};
    canAssign.targetRoles["r4"] = {0};
    canAssign.targetRoles["r5"] = {1};
    canAssign.targetRoles["r6"] = {2};
    canAssign.targetRoles["r2"] = {3};
    canRevoke.admins["r6"] = {0};
    canRevoke.admins["r4"] = {1};
    canRevoke.targetRoles["r3"] = {0};
    canRevoke.targetRoles["r2"] = {1};
    vector<string> goal = {"r4", "r6"};
    map<int, vector<string>> assignRules;
    map<int, vector<string>> revokeRules;
    assignRules[0] = {"r1", "r2", "r3", "r4"};
    assignRules[1] = {"r3", "r4", "NULL", "r5"};
    assignRules[2] = {"r1", "r3", "NULL", "r6"};
    assignRules[3] = {"r1", "r3", "NULL", "r2"};
    revokeRules[0] = {"r6", "r3"};
    revokeRules[1] = {"r4", "r2"};

    returnStruct returnVal = slicing(goal, canAssign, canRevoke, assignRules, revokeRules);

    printReturnStruct(returnVal);

    return 0;
}

returnStruct slicing(vector<string> goal, can_assign assign, can_revoke revoke, map<int, vector<string>> assignRules, map<int, vector<string>> revokeRules){
    // goal = {"r4", "r6"}
    vector<string> relPosRoles = goal;
    vector<string> relNegRoles;
    vector<int> relAssignRules;
    vector<int> relRevokeRules;
    vector<string> workSet = goal;
    vector<string> seen;
    while (workSet.size() != 0){
        string currentProcessingRole = workSet.back();
        workSet.pop_back();
        seen.push_back(currentProcessingRole);
        // make sure that r exists
        if(assign.targetRoles.count(currentProcessingRole) > 0){
            vector<int> currentRelRules = assign.targetRoles[currentProcessingRole];
            // for all can_assign rules that target is r
            for (int i = 0; i < currentRelRules.size(); i++){
                // add current rule into relRules
                if (isNotIn(currentRelRules[i], relAssignRules)){
                    relAssignRules.push_back(currentRelRules[i]);
                }
                // if the admin is not seen
                if (isNotSeen(assignRules[currentRelRules[i]][0], seen) && isNotSeen(assignRules[currentRelRules[i]][0], relPosRoles)){
                    workSet.push_back(assignRules[currentRelRules[i]][0]);
                    relPosRoles.push_back(assignRules[currentRelRules[i]][0]);
                }
                // if the positive roles are not seen(may have more than 1 postive roles)
                vector<string> tmp = splitString(assignRules[currentRelRules[i]][1], '&');
                for(int j = 0; j < tmp.size(); j++){
                    if(isNotSeen(tmp[j], seen) && isNotSeen(tmp[j], relPosRoles)){
                        workSet.push_back(tmp[j]);
                        relPosRoles.push_back(tmp[j]);
                    }
                }
                // if new negative roles
                if(assignRules[currentRelRules[i]][2] != "NULL"){
                    tmp = splitString(assignRules[currentRelRules[i]][2], '&');
                    for(int j = 0; j < tmp.size(); j++){
                        if(isNotSeen(tmp[j], relNegRoles)){
                            relNegRoles.push_back(tmp[j]);
                        }
                    }
                }
            }
        }
        // see if can revoke a negative role
        for (int i = 0; i < relNegRoles.size(); i++){
            string currentRelNegRoles = relNegRoles[i];
            // for all can_revoke rules that target is negative role
            if(revoke.targetRoles.count(currentRelNegRoles) > 0){
                vector<int> currentNegRelRules = revoke.targetRoles[currentRelNegRoles];
                for(int j = 0; j < currentNegRelRules.size(); j++){
                    if(isNotIn(currentNegRelRules[j], relRevokeRules)){
                        relRevokeRules.push_back(currentNegRelRules[j]);
                    }
                    if(isNotSeen(revokeRules[j][0], seen)){
                        relPosRoles.push_back(revokeRules[j][0]);
                        workSet.push_back(revokeRules[j][0]);
                    }
                }
                // for (int j = 0; j < currentRelNegRoles.size(); j++){
                //     if (isNotSeen(revokeRules[j][0], seen)){
                //         relPosRoles.push_back(revokeRules[j][0]);
                //         workSet.push_back(revokeRules[j][0]);
                //     }
                // }
            }
        }
    }
    returnStruct returnVal;
    returnVal.relAssignRules = relAssignRules;
    returnVal.relRevokeRules = relRevokeRules;
    returnVal.relPosRoles = relPosRoles;
    returnVal.relNegRoles = relNegRoles;
    return returnVal;
}

bool isNotSeen(string target, vector<string> vec){
    if(find(vec.begin(), vec.end(), target) != vec.end()){
        return false;
    }
    else{
        return true;
    }
}

bool isNotIn(int target, vector<int> vec){
    if(find(vec.begin(), vec.end(), target) != vec.end()){
        return false;
    }
    else{
        return true;
    }
}

vector<string> splitString(string str, char delimiter){
    stringstream ss(str);
    string token;
    vector<string> tokens;

    // Split the string using the delimiter
    while (getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

void printReturnStruct(const returnStruct& rs) {
    cout << "relPosRoles: ";
    for (const auto& role : rs.relPosRoles) {
        cout << role << " ";
    }
    cout << endl;

    cout << "relNegRoles: ";
    for (const auto& role : rs.relNegRoles) {
        cout << role << " ";
    }
    cout << endl;

    cout << "relAssignRules: ";
    for (const auto& rule : rs.relAssignRules) {
        cout << rule << " ";
    }
    cout << endl;

    cout << "relRevokeRules: ";
    for (const auto& rule : rs.relRevokeRules) {
        cout << rule << " ";
    }
    cout << endl;
}