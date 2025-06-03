//+----------------------------------------------------------------------------+
// list all the can assign rules first, then all the can revoke rules
// and then all the user-role assignments.
// The input file format should be as follows:
// can_assign(auth, cond, targetRole)
// can_revoke(auth, targetRole)
// UA(user, role)
//+----------------------------------------------------------------------------+

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <map>
using namespace std;

struct AssignRule {
    int auth;
    vector<int> cond;
    int targetRole;
};

struct RevokeRule {
    int auth;
    int targetRole;
};

string trim(const string& s);

int main() {

    vector<AssignRule> canAssign;    // can asign rules with AssignRule struct
    vector<RevokeRule> canRevoke;    // can revoke rules with RevokeRule struct

    map<string, int> roleMap;
    int roleCount = 0;
    map<string, int> userMap;
    int userCount = 0;
    map<int, vector<int>> usrRoleMap;
    int usrRoleCount = 0;


    ifstream infile("rules.txt");
    if (!infile) {
        cerr << "Unable to open file.\n";
        return 1;
    }
    string line;
    while (getline(infile, line)) {
        if(line[0] == 'R' || line[0] == 'P'){
            continue;   
        }
        else if(line[4] == 'a'){
            // can_assign rule
            AssignRule canAssignRule;
            size_t start = line.find('(');
            size_t end = line.find(')');

            std::string content = line.substr(start + 1, end - start - 1);

            // trim rule content
            string auth = "";
            string cond = "";
            string targetRole = "";
            stringstream ss(content);
            if (getline(ss, auth, ',') &&
                getline(ss, cond, ',') &&
                getline(ss, targetRole, ',')) {
                auth = auth;
                cond = trim(cond);
                targetRole = targetRole;
            }
            // if auth is not in map
            if(roleMap.find(auth) == roleMap.end()){
                roleMap[auth] = roleCount;
                roleCount++;
            }
            // if targetRole is not in map
            if(roleMap.find(targetRole) == roleMap.end()){
                roleMap[targetRole] = roleCount;
                roleCount++;
            }
            canAssignRule.auth = roleMap[auth];
            canAssignRule.targetRole = roleMap[targetRole];

            // trim condition
            stringstream condStream(cond);
            string condPart;
            vector<string> token;
            while(condStream >> condPart) {
                if(condPart[0] == '-'){
                    condPart = condPart.substr(1);
                    if(roleMap.find(condPart) == roleMap.end()){
                        roleMap[condPart] = roleCount;
                        roleCount++;
                    }
                    canAssignRule.cond.push_back(-roleMap[condPart]);
                }
                else{
                    if(roleMap.find(condPart) == roleMap.end()){
                        roleMap[condPart] = roleCount;
                        roleCount++;
                    }
                    canAssignRule.cond.push_back(roleMap[condPart]);
                }
            }
            canAssign.push_back(canAssignRule);
        }
        else if(line[4] == 'r'){
            // can_revoke rule
            RevokeRule canRevokeRule;
            size_t start = line.find('(');
            size_t end = line.find(')');
            std::string content = line.substr(start + 1, end - start - 1);
            string auth = "";
            string targetRole = "";
            stringstream ss(content);
            if (getline(ss, auth, ',') &&
                getline(ss, targetRole, ',')) {
                auth = auth;
                targetRole = targetRole;
            }

            if(roleMap.find(auth) == roleMap.end()){
                roleMap[auth] = roleCount;
                roleCount++;
            }
            if(roleMap.find(targetRole) == roleMap.end()){
                roleMap[targetRole] = roleCount;
                roleCount++;
            }
            canRevokeRule.auth = roleMap[auth];
            canRevokeRule.targetRole = roleMap[targetRole];
            canRevoke.push_back(canRevokeRule);
        }
        else if(line[0] == 'U'){
            // user-role assignment
            size_t start = line.find('(');
            size_t end = line.find(')');
            std::string content = line.substr(start + 1, end - start - 1);
            string user = "";
            string role = "";
            stringstream ss(content);
            if (getline(ss, user, ',') &&
                getline(ss, role, ',')) {
                user = trim(user);
                role = trim(role);
            }
            // add user to user map
            if(userMap.find(user) == userMap.end()){
                userMap[user] = userCount;
                userCount++;
            }
            // make user the role exists
            if(roleMap.find(role) == roleMap.end()){
                roleMap[role] = roleCount;
                roleCount++;
            }

            // add user if it's not in usrRoleMap
            if(usrRoleMap.find(userMap[user])== usrRoleMap.end()){
                usrRoleMap[userMap[user]] = vector<int>();
                usrRoleMap[userMap[user]].push_back(roleMap[role]);
                usrRoleCount++;
            }
            else{
                // if user already exists, add the role to the user's role list
                usrRoleMap[userMap[user]].push_back(roleMap[role]);
            }
        }
    }
    infile.close();
    // // print the can_assign rules
    // cout << "Can Assign Rules:\n";
    // for (const auto& rule : canAssign) {
    //     cout << "can_assign(" << rule.auth << ", ";
    //     for (size_t i = 0; i < rule.cond.size(); ++i) {
    //         cout << rule.cond[i];
    //         if (i < rule.cond.size() - 1) {
    //             cout << " ";
    //         }
    //     }
    //     cout << ", " << rule.targetRole << ")\n";
    // }
    // // print the can_revoke rules
    // cout << "Can Revoke Rules:\n";
    // for (const auto& rule : canRevoke) {
    //     cout << "can_revoke(" << rule.auth << ", " << rule.targetRole << ")\n";
    // }
    // // print the role map
    // cout << "Role Map:\n";
    // for (const auto& role : roleMap) {
    //     cout << role.first << " -> " << role.second << "\n";
    // }
    // // print the user map
    // cout << "User Map:\n";
    // for (const auto& user : userMap) {
    //     cout << user.first << " -> " << user.second << "\n";
    // }
    // // print the user-role map
    // cout << "User-Role Map:\n";
    // for (const auto& usrRole : usrRoleMap) {
    //     cout << usrRole.first << " -> ";
    //     for (const auto& role : usrRole.second) {
    //         cout << role << " ";
    //     }
    //     cout << "\n";
    // }
    return 0;
}

string trim(const string& s){
    size_t start = s.find_first_not_of(" \t\n\r");
    size_t end = s.find_last_not_of(" \t\n\r");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}