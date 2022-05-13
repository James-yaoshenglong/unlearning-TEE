#ifndef PURCHASE_ARCH_HPP
#define PURCHASE_ARCH_HPP

#include <assert.h>

#include <chrono>
#include <numeric>
#include <vector>
#include <unordered_map>

#include "example_utils.hpp"
#include "Enclave.h"
#include "data_structure.hpp"

using namespace dnnl;

using namespace std;

class MLP{
    private:
        int network[3];
        float alpha;
        engine eng;
        stream s;
        memory::dim batch;
        vector<primitive> net_fwd;
        vector<primitive> net_bwd;
        vector<std::unordered_map<int, memory>> net_fwd_args;
        vector<std::unordered_map<int, memory>> net_bwd_args;
        vector<float> user_src;
        memory user_src_memory;
        vector<float> user_dst;
        memory user_dst_memory;
        vector<float> fc1_weights;
        memory fc1_user_weights_memory;
        vector<float> fc1_bias;
        memory fc1_user_bias_memory;
        vector<float> fc2_weights;
        memory fc2_user_weights_memory;
        vector<float> fc2_bias;
        memory fc2_user_bias_memory;
        vector<float> net_diff_dst;
        memory fc2_user_diff_dst_memory;
        vector<float> fc1_user_diff_weights_buffer;
        memory fc1_user_diff_weights_memory;
        vector<float> fc1_diff_bias_buffer;
        memory fc1_diff_bias_memory;
        vector<float> fc2_user_diff_weights_buffer;
        memory fc2_user_diff_weights_memory;
        vector<float> fc2_diff_bias_buffer;
        memory fc2_diff_bias_memory;
    public:
        MLP(int arch[3], float a, int b);
        void forward(const vector<float>& input);
        void backward(const vector<float>& target);
        void train(float* data, float* label, int epoch, int size);
        void setModel(Model* model);
        void saveModel(Model* model);
        vector<float> inference(vector<float>& input);
        static memory::dim product(const memory::dims &dims) {
            return std::accumulate(dims.begin(), dims.end(), (memory::dim)1,
                    std::multiplies<memory::dim>());
        }
};

#endif