#include <algorithm>
#include <chrono>
#include <vector>

class timer {
public:
    std::vector<double> arr;
    bool sort_flag = false;
    std::chrono::time_point<std::chrono::steady_clock> s;
    void start(){
        s = std::chrono::steady_clock::now();
    }
    double end(){
        auto t = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - s).count();
        double l = t * 1e-6f;
        arr.push_back(l);
        sort_flag = false;
        return l;
    }

    double mean(){
        double sum=0;
        for(auto it : arr)
            sum += it;
        return sum/arr.size();
    }

    void sort(){
        if(sort_flag) return;
        std::sort(arr.begin(), arr.end());
        sort_flag = true;
    }

    
    double pile(float p){
        sort();
        int idx = (arr.size() - 1) * p;
        return arr[idx];
    }

    double max(){
        sort();
        return arr[arr.size() - 1];
    }
    double min(){
        sort();
        return arr[0];
    }

    timer(/* args */){}
    ~timer(){}
};

