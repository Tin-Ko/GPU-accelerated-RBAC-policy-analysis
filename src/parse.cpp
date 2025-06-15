// #include <iostream>
// #include <fstream>
// #include <string>
// #include <sstream>
// #include <map>
// #include <vector>
// using namespace std;
//
// struct AssignRule {
//     int auth;
//     vector<int> cond;
//     int targetRole;
// };
//
// struct RevokeRule {
//     int auth;
//     int targetRole;
// };
//
// string trim(const string& s);
//
// int main() {
//     map<string, int> roleMap;
//     vector<AssignRule> canAssign;
//     vector<RevokeRule> canRevoke;
//     int roleCount = 0;
//     ifstream infile("rules.txt");
//     if (!infile) {
//         cerr << "Unable to open file.\n";
//         return 1;
//     }
//     string line;
//     while (getline(infile, line)) {
//         if(line[0] == 'R' || line[0] == 'P' || line[0] == 'U'){
//             continue;   
//         }
//         else if(line[4] == 'a'){
//             // can_assign rule
//             AssignRule canAssignRule;
//             size_t start = line.find('(');
//             size_t end = line.find(')');
//
//             std::string content = line.substr(start + 1, end - start - 1);
//
//             // trim rule content
//             string auth = "";
//             string cond = "";
//             string targetRole = "";
//             stringstream ss(content);
//             if (getline(ss, auth, ',') &&
//                 getline(ss, cond, ',') &&
//                 getline(ss, targetRole, ',')) {
//                 auth = auth;
//                 cond = trim(cond);
//                 targetRole = targetRole;
//             }
//
//             if(roleMap.find(auth) != roleMap.end()){
//                 roleMap[auth] = roleCount;
//                 roleCount++;
//             }
//             if(roleMap.find(targetRole) != roleMap.end()){
//                 roleMap[targetRole] = roleCount;
//                 roleCount++;
//             }
//             canAssignRule.auth = roleMap[auth];
//             canAssignRule.targetRole = roleMap[targetRole];
//
//             // trim condition
//             stringstream condStream(cond);
//             string condPart;
//             vector<string> token;
//             while(condStream >> condPart) {
//                 if(condPart[0] == '-'){
//                     condPart = condPart.substr(1);
//                     if(roleMap.find(condPart) != roleMap.end()){
//                         roleMap[condPart] = roleCount;
//                         roleCount++;
//                     }
//                     canAssignRule.cond.push_back(-roleMap[condPart]);
//                 }
//                 else{
//                     if(roleMap.find(condPart) == roleMap.end()){
//                         roleMap[condPart] = roleCount;
//                         roleCount++;
//                     }
//                     canAssignRule.cond.push_back(roleMap[condPart]);
//                 }
//             }
//             canAssign.push_back(canAssignRule);
//         }
//         else if(line[4] == 'r'){
//             // can_revoke rule
//             RevokeRule canRevokeRule;
//             size_t start = line.find('(');
//             size_t end = line.find(')');
//             std::string content = line.substr(start + 1, end - start - 1);
//             string auth = "";
//             string targetRole = "";
//             stringstream ss(content);
//             if (getline(ss, auth, ',') &&
//                 getline(ss, targetRole, ',')) {
//                 auth = auth;
//                 targetRole = targetRole;
//             }
//
//             if(roleMap.find(auth) != roleMap.end()){
//                 roleMap[auth] = roleCount;
//                 roleCount++;
//             }
//             if(roleMap.find(targetRole) != roleMap.end()){
//                 roleMap[targetRole] = roleCount;
//                 roleCount++;
//             }
//             canRevokeRule.auth = roleMap[auth];
//             canRevokeRule.targetRole = roleMap[targetRole];
//             canRevoke.push_back(canRevokeRule);
//         }
//     }
//     infile.close();
//     return 0;
// }
//
// string trim(const string& s){
//     size_t start = s.find_first_not_of(" \t\n\r");
//     size_t end = s.find_last_not_of(" \t\n\r");
//     return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
// }
