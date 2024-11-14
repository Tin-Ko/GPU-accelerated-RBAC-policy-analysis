#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <sstream>
#include <fstream>
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

struct relRulesAndRoles{
    vector<int> relAssignRules;
    vector<int> relRevokeRules;
    vector<string> relPosRoles;
    vector<string> relNegRoles;
};

struct Rules{
    map<int, vector<string>> assignRules;
    map<int, vector<string>> revokeRules;
};

// functions
bool isNotSeen(string target, vector<string> vec);
bool isNotIn(int target, vector<int> vec);
vector<string> splitString(string str, char delimiter);
relRulesAndRoles slicing(vector<string> goal, string fileName);
void printSlicingResult(relRulesAndRoles rs);
Rules readRules(string fileName);
can_assign parseAssignRules(map<int, vector<string>> assignRules);
can_revoke parseRevokeRules(map<int, vector<string>> revokeRules);

int main(){
    vector<string> goal = {"r4", "r6"};

    // apply slicing
    relRulesAndRoles rel = slicing(goal, "input.txt");

    printSlicingResult(rel);

    return 0;
}

relRulesAndRoles slicing(vector<string> goal, string fileName){
    
    Rules retRules = readRules(fileName);
    map<int, vector<string>> assignRules = retRules.assignRules;
    map<int, vector<string>> revokeRules = retRules.revokeRules;
    
    can_assign canAssign = parseAssignRules(assignRules);
    can_revoke canRevoke = parseRevokeRules(revokeRules);
    
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
        if(canAssign.targetRoles.count(currentProcessingRole) > 0){
            vector<int> currentRelRules = canAssign.targetRoles[currentProcessingRole];
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
            if(canRevoke.targetRoles.count(currentRelNegRoles) > 0){
                vector<int> currentNegRelRules = canRevoke.targetRoles[currentRelNegRoles];
                for(int j = 0; j < currentNegRelRules.size(); j++){
                    if(isNotIn(currentNegRelRules[j], relRevokeRules)){
                        relRevokeRules.push_back(currentNegRelRules[j]);
                    }
                    if(isNotSeen(revokeRules[j][0], seen)){
                        relPosRoles.push_back(revokeRules[j][0]);
                        workSet.push_back(revokeRules[j][0]);
                    }
                }
            }
        }
    }
    relRulesAndRoles returnVal = {relAssignRules, relRevokeRules, relPosRoles, relNegRoles};
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

void printSlicingResult(relRulesAndRoles rs){
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

Rules readRules(string fileName){
    ifstream file(fileName);
    string line;
    map<int, vector<string>> assignRules;
    map<int, vector<string>> revokeRules;
    bool isAssign = false, isRevoke = false;
    int assignIndex = 0, revokeIndex = 0;

    while (getline(file, line)) {
        if (line == "can_assign") {
            isAssign = true;
            isRevoke = false;
            continue;
        } else if (line == "can_revoke") {
            isAssign = false;
            isRevoke = true;
            continue;
        }

        vector<string> rule;
        stringstream ss(line);
        string item;

        while (getline(ss, item, ',')) {
            item.erase(0, item.find_first_not_of(" \t\n\r"));
            item.erase(item.find_last_not_of(" \t\n\r") + 1);
            rule.push_back(item);
        }

        if (isAssign) {
            assignRules[assignIndex++] = rule;
        } else if (isRevoke) {
            revokeRules[revokeIndex++] = rule;
        }
    }

    file.close();
    Rules retRules = {assignRules, revokeRules};
    return retRules;
}

can_assign parseAssignRules(map<int, vector<string>> assignRules){
    can_assign result;
    for (const auto& rule : assignRules) {
        const auto& role = rule.second;

        string admin = role[0];
        string posRole = role[1];
        string negRole = role[2];
        string targetRole = role[3];

        result.admins[admin].push_back(rule.first);
        result.posRoles[posRole].push_back(rule.first);
        result.negRoles[negRole].push_back(rule.first);
        result.targetRoles[targetRole].push_back(rule.first);
    }

    return result;
}

can_revoke parseRevokeRules(map<int, vector<string>> revokeRules){
    can_revoke result;
    for (const auto& rule : revokeRules) {
        const auto& role = rule.second;

        string admin = role[0];
        string targetRole = role[1];

        result.admins[admin].push_back(rule.first);
        result.targetRoles[targetRole].push_back(rule.first);
    }

    return result;
}