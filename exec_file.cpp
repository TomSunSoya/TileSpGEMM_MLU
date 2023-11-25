#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <algorithm>
using namespace std;

int main() {
    ifstream fin("wang3.mtx");
    string line;

    ofstream fout("wang.mtx");

    int i = 0;
    while (getline(fin, line)) {
        if (i++ > 14) {
            int index = line.find('.');
            if (line[index-1] == '-') --index;
            double val = stod(line.substr(index));
            val *= 10000;
            string t = to_string(val);
            line.replace(index, t.size(), t);
        }
        fout << line << endl;
    }
    return 0;
}