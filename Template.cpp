#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <deque>
#include <queue>
#include <stack>
#include <set>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <functional>
#include <utility>
#include <bitset>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <cassert>
#include <iterator>
using namespace std;
using ull = unsigned long long;
using ll = long long;
using vi = vector<int>;
using vl = vector<long>;
using vll = vector<long long>;
using vvi = vector<vi>;
using vvl = vector<vl>;
using vvll = vector<vll>;
using vs = vector<string>;
using pii = pair<int, int>;
#define OVERLOAD_REP(_1, _2, _3, name, ...) name
#define REP1(i, n) for (auto i = std::decay_t<decltype(n)>{}; (i) != (n); ++(i))
#define REP2(i, l, r) for (auto i = (l); (i) != (r); ++(i))
#define rep(...) OVERLOAD_REP(__VA_ARGS__, REP2, REP1)(__VA_ARGS__)
#define dbg(foo) cout << #foo << " : " << foo << endl
template< typename T1, typename T2 >
ostream &operator<<(ostream &os, const pair< T1, T2 >& p) {
    os << p.first << " " << p.second;
    return os;
}

template< typename T1, typename T2 >
istream &operator>>(istream &is, pair< T1, T2 > &p) {
    is >> p.first >> p.second;
    return is;
}

template< typename T >
ostream &operator<<(ostream &os, const vector< T > &v) {
    for(int i = 0; i < (int) v.size(); i++) {
        os << v[i] << (i + 1 != v.size() ? " " : "");
    }
    return os;
}

template< typename T >
ostream &operator<<(ostream &os, const vector< vector< T > > &v) {
    for(int i = 0; i < (int) v.size(); i++) {
        os << v[i] << (i + 1 != v.size() ? "\n" : "");
    }
    return os;
}

template< typename T >
istream &operator>>(istream &is, vector< T > &v) {
    for(T &in : v) is >> in;
    return is;
}

int main() {
    cin.tie(nullptr);
    std::ios_base::sync_with_stdio(false);

    return 0;
}
