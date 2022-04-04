/*
 * Copyright (C) 2011-2021 Intel Corporation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the
 *     distribution.
 *   * Neither the name of Intel Corporation nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

/// @example cnn_inference_f32.cpp
/// @copybrief cnn_inference_f32_cpp
/// > Annotated version: @ref cnn_inference_f32_cpp

/// @page cnn_inference_f32_cpp CNN f32 inference example
/// This C++ API example demonstrates how to build an AlexNet neural
/// network topology for forward-pass inference.
///
/// > Example code: @ref cnn_inference_f32.cpp
///
/// Some key take-aways include:
///
/// * How tensors are implemented and submitted to primitives.
/// * How primitives are created.
/// * How primitives are sequentially submitted to the network, where the output
///   from primitives is passed as input to the next primitive. The latter
///   specifies a dependency between the primitive input and output data.
/// * Specific 'inference-only' configurations.
/// * Limiting the number of reorders performed that are detrimental
///   to performance.
///
/// The example implements the AlexNet layers
/// as numbered primitives (for example, fc21, fc21, fc22).

#include <assert.h>

#include <chrono>
#include <numeric>
#include <vector>
#include <unordered_map>

#include "example_utils.hpp"
#include "Enclave.h"

using namespace dnnl;

using namespace std;

static memory::dim product(const memory::dims &dims) {
    return std::accumulate(dims.begin(), dims.end(), (memory::dim)1,
            std::multiplies<memory::dim>());
}

static void simple_net(engine::kind engine_kind, int times = 100, int feature_num = 6) {
    using tag = memory::format_tag;
    using dt = memory::data_type;

    /// Initialize an engine and stream. The last parameter in the call represents
    /// the index of the engine.
    /// @snippet cnn_inference_f32.cpp Initialize engine and stream
    //[Initialize engine and stream]
    engine eng(engine_kind, 0);
    stream s(eng);
    //[Initialize engine and stream]

    /// Create a vector for the primitives and a vector to hold memory
    /// that will be used as arguments.
    /// @snippet cnn_inference_f32.cpp Create network
    //[Create network]
    std::vector<primitive> net_fwd, net_bwd;
    std::vector<std::unordered_map<int, memory>> net_fwd_args, net_bwd_args;
    //[Create network]

    const memory::dim batch = 1;

    // fc1 inner product {batch, feature_num} (x) {1000, feature_num}-> {batch, 1000}
    memory::dims fc1_src_tz = {batch, feature_num};
    memory::dims fc1_weights_tz = {10, feature_num};
    memory::dims fc1_bias_tz = {10};
    memory::dims fc1_dst_tz = {batch, 10};

    /// Allocate buffers for input and output data, weights, and bias.
    /// @snippet cnn_inference_f32.cpp Allocate buffers
    //[Allocate buffers]
    std::vector<float> user_src(batch * feature_num); //batch * feature_num
    std::vector<float> user_dst(batch * 1);
    //[Allocate buffers]

    // initializing non-zero values for user_src, latter should get from out
    for (size_t i = 0; i < user_src.size(); ++i)
        user_src[i] = sinf((float)i);

    auto user_src_memory = memory({{fc1_src_tz}, dt::f32, tag::nc}, eng);
    write_to_dnnl_memory(user_src.data(), user_src_memory);

    std::vector<float> fc1_weights(product(fc1_weights_tz));
    std::vector<float> fc1_bias(product(fc1_bias_tz));

    // initializing non-zero values for weights and bias
    for (size_t i = 0; i < fc1_weights.size(); ++i)
        fc1_weights[i] = sinf((float)i);
    for (size_t i = 0; i < fc1_bias.size(); ++i)
        fc1_bias[i] = sinf((float)i);

    // create memory for user data
    auto fc1_user_weights_memory
            = memory({{fc1_weights_tz}, dt::f32, tag::nc}, eng);
    write_to_dnnl_memory(fc1_weights.data(), fc1_user_weights_memory);

    auto fc1_user_bias_memory = memory({{fc1_bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(fc1_bias.data(), fc1_user_bias_memory);

    // create memory descriptors for fc2olution data w/ no specified format
    auto fc1_src_md = memory::desc({fc1_src_tz}, dt::f32, tag::any);
    auto fc1_bias_md = memory::desc({fc1_bias_tz}, dt::f32, tag::any);
    auto fc1_weights_md = memory::desc({fc1_weights_tz}, dt::f32, tag::any);
    auto fc1_dst_md = memory::desc({fc1_dst_tz}, dt::f32, tag::any);

               
    // create a inner_product
    auto fc1_desc = inner_product_forward::desc(prop_kind::forward_inference,
            fc1_src_md, fc1_weights_md, fc1_bias_md, fc1_dst_md);
    auto fc1_prim_desc = inner_product_forward::primitive_desc(fc1_desc, eng);

    auto fc1_src_memory = user_src_memory;
    if (fc1_prim_desc.src_desc() != user_src_memory.get_desc()) {
        fc1_src_memory = memory(fc1_prim_desc.src_desc(), eng);
        net_fwd.push_back(reorder(user_src_memory, fc1_src_memory));
        net_fwd_args.push_back({{DNNL_ARG_FROM, user_src_memory},
                {DNNL_ARG_TO, fc1_src_memory}});
    }
        
    auto fc1_weights_memory = fc1_user_weights_memory;
    if (fc1_prim_desc.weights_desc() != fc1_user_weights_memory.get_desc()) {
        fc1_weights_memory = memory(fc1_prim_desc.weights_desc(), eng);
        reorder(fc1_user_weights_memory, fc1_weights_memory)
                .execute(s, fc1_user_weights_memory, fc1_weights_memory);
    }

    auto fc1_dst_memory = memory(fc1_prim_desc.dst_desc(), eng);

    // create fc2olution primitive and add it to net
    

    net_fwd.push_back(inner_product_forward(fc1_prim_desc)); 
    //here in the inner product api has problem, solved the problem is in config.xml
    net_fwd_args.push_back({{DNNL_ARG_SRC, fc1_src_memory},
            {DNNL_ARG_WEIGHTS, fc1_weights_memory},
            {DNNL_ARG_BIAS, fc1_user_bias_memory},
            {DNNL_ARG_DST, fc1_dst_memory}});
                      
    //relu1
    // {batch, 1000} -> {batch, 1000}
    const float negative1_slope = 1.0f;

    // create relu primitive and add it to net
    auto relu1_desc = eltwise_forward::desc(prop_kind::forward_inference,
            algorithm::eltwise_relu, fc1_dst_memory.get_desc(),
            negative1_slope);
    auto relu1_prim_desc = eltwise_forward::primitive_desc(relu1_desc, eng);

    net_fwd.push_back(eltwise_forward(relu1_prim_desc));
    net_fwd_args.push_back({{DNNL_ARG_SRC, fc1_dst_memory},
            {DNNL_ARG_DST, fc1_dst_memory}});


    // fc2 inner product {batch, 1000} (x) {1, 1000}-> {batch, 1}
    memory::dims fc2_weights_tz = {1, 10};
    memory::dims fc2_bias_tz = {1};
    memory::dims fc2_dst_tz = {batch, 1};

    std::vector<float> fc2_weights(product(fc2_weights_tz));
    std::vector<float> fc2_bias(product(fc2_bias_tz));

    // initializing non-zero values for weights and bias
    for (size_t i = 0; i < fc2_weights.size(); ++i)
        fc2_weights[i] = sinf((float)i);
    for (size_t i = 0; i < fc2_bias.size(); ++i)
        fc2_bias[i] = sinf((float)i);

    // create memory for user data
    auto fc2_user_weights_memory
            = memory({{fc2_weights_tz}, dt::f32, tag::nc}, eng);
    write_to_dnnl_memory(fc2_weights.data(), fc2_user_weights_memory);
    auto fc2_user_bias_memory = memory({{fc2_bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(fc2_bias.data(), fc2_user_bias_memory);
    auto user_dst_memory = memory({{fc2_dst_tz}, dt::f32, tag::nc}, eng);
    write_to_dnnl_memory(user_dst.data(), user_dst_memory);

    // create memory descriptors for fc2olution data w/ no specified format
    auto fc2_bias_md = memory::desc({fc2_bias_tz}, dt::f32, tag::any);
    auto fc2_weights_md = memory::desc({fc2_weights_tz}, dt::f32, tag::any);
    auto fc2_dst_md = memory::desc({fc2_dst_tz}, dt::f32, tag::any);

    // create a inner_product
    auto fc2_desc = inner_product_forward::desc(prop_kind::forward_inference,
            fc1_dst_memory.get_desc(), fc2_weights_md, fc2_bias_md, fc2_dst_md);
    auto fc2_prim_desc = inner_product_forward::primitive_desc(fc2_desc, eng);

    auto fc2_weights_memory = fc2_user_weights_memory;
    if (fc2_prim_desc.weights_desc() != fc2_user_weights_memory.get_desc()) {
        fc2_weights_memory = memory(fc2_prim_desc.weights_desc(), eng);
        reorder(fc2_user_weights_memory, fc2_weights_memory)
                .execute(s, fc2_user_weights_memory, fc2_weights_memory);
    }

    auto fc2_dst_memory = memory(fc2_prim_desc.dst_desc(), eng);

    // create fc2olution primitive and add it to net
    net_fwd.push_back(inner_product_forward(fc2_prim_desc));
    net_fwd_args.push_back({{DNNL_ARG_SRC, fc1_dst_memory},
            {DNNL_ARG_WEIGHTS, fc2_weights_memory},
            {DNNL_ARG_BIAS, fc2_user_bias_memory},
            {DNNL_ARG_DST, fc2_dst_memory}});

    // create reorder between internal and user data if it is needed and
    // add it to net after fc2ing
    if (fc2_dst_memory != user_dst_memory) {
        net_fwd.push_back(reorder(fc2_dst_memory, user_dst_memory));
        net_fwd_args.push_back({{DNNL_ARG_FROM, fc2_dst_memory},
                {DNNL_ARG_TO, user_dst_memory}});
    }

    //-----------------------------------------------------------------------
    //----------------- Backward Stream -------------------------------------
    // ... user diff_data ...
    std::vector<float> net_diff_dst(batch * 1);
    // for (size_t i = 0; i < net_diff_dst.size(); ++i)
    //     net_diff_dst[i] = sinf((float)i);
    net_diff_dst[0] = 0.1;
    //here should initialized by self computing

    auto fc2_user_diff_dst_memory
            = memory({{fc2_dst_tz}, dt::f32, tag::nc}, eng);
    write_to_dnnl_memory(net_diff_dst.data(), fc2_user_diff_dst_memory);
    
    // Backward inner_product with respect to weights
    // create user format diff weights and diff bias memory
    std::vector<float> fc2_user_diff_weights_buffer(product(fc2_weights_tz));
    std::vector<float> fc2_diff_bias_buffer(product(fc2_bias_tz));

    auto fc2_user_diff_weights_memory
            = memory({{fc2_weights_tz}, dt::f32, tag::nc}, eng);
    write_to_dnnl_memory(fc2_user_diff_weights_buffer.data(),
            fc2_user_diff_weights_memory);
    auto fc2_diff_bias_memory = memory({{fc2_bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(fc2_diff_bias_buffer.data(), fc2_diff_bias_memory);

    // create memory descriptors
    auto fc2_bwd_src_md = memory::desc({fc1_dst_tz}, dt::f32, tag::any);
    auto fc2_diff_bias_md = memory::desc({fc2_bias_tz}, dt::f32, tag::any);
    auto fc2_diff_weights_md
            = memory::desc({fc2_weights_tz}, dt::f32, tag::any);
    auto fc2_diff_dst_md = memory::desc({fc2_dst_tz}, dt::f32, tag::any);

    // create backward fc2olution primitive descriptor
    auto fc2_bwd_weights_desc
            = inner_product_backward_weights::desc(
                    fc2_bwd_src_md, fc2_diff_weights_md, fc2_diff_bias_md,
                    fc2_diff_dst_md);
    auto fc2_bwd_weights_pd = inner_product_backward_weights::primitive_desc(
            fc2_bwd_weights_desc, eng, fc2_prim_desc);

    // for best performance fc2olution backward might chose
    // different memory format for src and diff_dst
    // than the memory formats preferred by forward fc2olution
    // for src and dst respectively
    // create reorder primitives for src from forward fc2olution to the
    // format chosen by backward fc2olution
    auto fc2_bwd_src_memory = fc1_dst_memory;
    if (fc2_bwd_weights_pd.src_desc() != fc1_dst_memory.get_desc()) {
        fc2_bwd_src_memory = memory(fc2_bwd_weights_pd.src_desc(), eng);
        net_bwd.push_back(reorder(fc1_dst_memory, fc2_bwd_src_memory));
        net_bwd_args.push_back({{DNNL_ARG_FROM, fc1_dst_memory},
                {DNNL_ARG_TO, fc2_bwd_src_memory}});
    }

    // create reorder primitives for diff_dst between diff_src from relu_bwd
    // and format preferred by fc2_diff_weights
    auto fc2_diff_dst_memory = fc2_user_diff_dst_memory;
    if (fc2_dst_memory.get_desc() != fc2_user_diff_dst_memory.get_desc()) {
        fc2_diff_dst_memory = memory(fc2_dst_memory.get_desc(), eng);
        net_bwd.push_back(
                reorder(fc2_user_diff_dst_memory, fc2_diff_dst_memory));
        net_bwd_args.push_back({{DNNL_ARG_FROM, fc2_user_diff_dst_memory},
                {DNNL_ARG_TO, fc2_diff_dst_memory}});
    }

    // create backward fc2olution primitive
    net_bwd.push_back(inner_product_backward_weights(fc2_bwd_weights_pd));
    net_bwd_args.push_back({{DNNL_ARG_SRC, fc2_bwd_src_memory},
            {DNNL_ARG_DIFF_DST, fc2_diff_dst_memory},
            // delay putting DIFF_WEIGHTS until reorder (if needed)
            {DNNL_ARG_DIFF_BIAS, fc2_diff_bias_memory}});

    // create reorder primitives between fc2 diff weights and user diff weights
    // if needed
    auto fc2_diff_weights_memory = fc2_user_diff_weights_memory;
    if (fc2_bwd_weights_pd.diff_weights_desc()
            != fc2_user_diff_weights_memory.get_desc()) {
        fc2_diff_weights_memory
                = memory(fc2_bwd_weights_pd.diff_weights_desc(), eng);
        net_bwd_args.back().insert(
                {DNNL_ARG_DIFF_WEIGHTS, fc2_diff_weights_memory});

        net_bwd.push_back(reorder(
                fc2_diff_weights_memory, fc2_user_diff_weights_memory));
        net_bwd_args.push_back({{DNNL_ARG_FROM, fc2_diff_weights_memory},
                {DNNL_ARG_TO, fc2_user_diff_weights_memory}});
    } else {
        net_bwd_args.back().insert(
                {DNNL_ARG_DIFF_WEIGHTS, fc2_diff_weights_memory});
    }


    //Backward inner_product with respect to data
    // create memory descriptors for inner-product
    auto fc2_diff_src_md = memory::desc({fc1_dst_tz}, dt::f32, tag::any);
    // auto fc2_diff_dst_md = memory::desc({fc2_dst_tz}, dt::f32, tag::any);
    
    // create backward fc2ing descriptor
    auto fc2_bwd_desc = inner_product_backward_data::desc(
            fc2_diff_src_md, fc2_weights_md, fc2_diff_dst_md);
    // backward primitive descriptor needs to hint forward descriptor
    auto fc2_bwd_pd
            = inner_product_backward_data::primitive_desc(fc2_bwd_desc, eng, fc2_prim_desc);

    // create reorder primitive between user diff dst and fc2 diff dst
    // if required
    // auto fc2_diff_dst_memory = fc2_user_diff_dst_memory;
    // if (fc2_dst_memory.get_desc() != fc2_user_diff_dst_memory.get_desc()) {
    //     fc2_diff_dst_memory = memory(fc2_dst_memory.get_desc(), eng);
    //     net_bwd.push_back(
    //             reorder(fc2_user_diff_dst_memory, fc2_diff_dst_memory));
    //     net_bwd_args.push_back({{DNNL_ARG_FROM, fc2_user_diff_dst_memory},
    //             {DNNL_ARG_TO, fc2_diff_dst_memory}});
    // }

    // create memory for fc2 diff src
    auto fc2_diff_src_memory = memory(fc2_bwd_pd.diff_src_desc(), eng);

    // finally create backward fc2ing primitive
    net_bwd.push_back(inner_product_backward_data(fc2_bwd_pd));
    net_bwd_args.push_back({{DNNL_ARG_WEIGHTS, fc2_weights_memory},
            // {DNNL_ARG_BIAS, fc2_user_bias_memory},
            {DNNL_ARG_DIFF_DST, fc2_diff_dst_memory},
            {DNNL_ARG_DIFF_SRC, fc2_diff_src_memory}});

        // Backward relu
    // auto relu1_diff_dst_md = memory::desc({relu1_data_tz}, dt::f32, tag::any);
    auto relu1_diff_dst_md = fc1_prim_desc.dst_desc();
    auto relu1_src_md = fc1_prim_desc.dst_desc();

    // create backward relu primitive_descriptor
    auto relu1_bwd_desc = eltwise_backward::desc(algorithm::eltwise_relu,
            relu1_diff_dst_md, relu1_src_md, negative1_slope);
    auto relu1_bwd_pd
            = eltwise_backward::primitive_desc(relu1_bwd_desc, eng, relu1_prim_desc);

    // create reorder primitive between lrn diff src and relu diff dst
    // if required
    auto relu1_diff_dst_memory = fc2_diff_src_memory;
    if (relu1_diff_dst_memory.get_desc() != relu1_bwd_pd.diff_dst_desc()) {
        relu1_diff_dst_memory = memory(relu1_bwd_pd.diff_dst_desc(), eng);
        net_bwd.push_back(reorder(fc2_diff_src_memory, relu1_diff_dst_memory));
        net_bwd_args.push_back({{DNNL_ARG_FROM, fc2_diff_src_memory},
                {DNNL_ARG_TO, relu1_diff_dst_memory}});
    }

    // create memory for relu diff src
    auto relu1_diff_src_memory = memory(relu1_bwd_pd.diff_src_desc(), eng);

    // finally create a backward relu primitive
    net_bwd.push_back(eltwise_backward(relu1_bwd_pd));
    net_bwd_args.push_back({{DNNL_ARG_SRC, fc1_dst_memory},
            {DNNL_ARG_DIFF_DST, relu1_diff_dst_memory},
            {DNNL_ARG_DIFF_SRC, relu1_diff_src_memory}});

    // Backward inner_product with respect to weights
    // create user format diff weights and diff bias memory
    std::vector<float> fc1_user_diff_weights_buffer(product(fc1_weights_tz));
    std::vector<float> fc1_diff_bias_buffer(product(fc1_bias_tz));

    auto fc1_user_diff_weights_memory
            = memory({{fc1_weights_tz}, dt::f32, tag::nc}, eng);
    write_to_dnnl_memory(fc1_user_diff_weights_buffer.data(),
            fc1_user_diff_weights_memory);
    auto fc1_diff_bias_memory = memory({{fc1_bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(fc1_diff_bias_buffer.data(), fc1_diff_bias_memory);

    // create memory descriptors
    auto fc1_bwd_src_md = memory::desc({fc1_src_tz}, dt::f32, tag::any);
    auto fc1_diff_bias_md = memory::desc({fc1_bias_tz}, dt::f32, tag::any);
    auto fc1_diff_weights_md
            = memory::desc({fc1_weights_tz}, dt::f32, tag::any);
    auto fc1_diff_dst_md = memory::desc({fc1_dst_tz}, dt::f32, tag::any);

    // create backward fc2olution primitive descriptor
    auto fc1_bwd_weights_desc
            = inner_product_backward_weights::desc(
                    fc1_bwd_src_md, fc1_diff_weights_md, fc1_diff_bias_md,
                    fc1_diff_dst_md);
    auto fc1_bwd_weights_pd = inner_product_backward_weights::primitive_desc(
            fc1_bwd_weights_desc, eng, fc1_prim_desc);

    // for best performance fc2olution backward might chose
    // different memory format for src and diff_dst
    // than the memory formats preferred by forward fc2olution
    // for src and dst respectively
    // create reorder primitives for src from forward fc2olution to the
    // format chosen by backward fc2olution
    auto fc1_bwd_src_memory = fc1_src_memory;
    if (fc1_bwd_weights_pd.src_desc() != fc1_src_memory.get_desc()) {
        fc1_bwd_src_memory = memory(fc1_bwd_weights_pd.src_desc(), eng);
        net_bwd.push_back(reorder(fc1_src_memory, fc1_bwd_src_memory));
        net_bwd_args.push_back({{DNNL_ARG_FROM, fc1_src_memory},
                {DNNL_ARG_TO, fc1_bwd_src_memory}});
    }

    // create reorder primitives for diff_dst between diff_src from relu_bwd
    // and format preferred by fc2_diff_weights
    auto fc1_diff_dst_memory = relu1_diff_src_memory;
    if (fc1_dst_memory.get_desc() != relu1_diff_src_memory.get_desc()) {
        fc1_diff_dst_memory = memory(fc1_dst_memory.get_desc(), eng);
        net_bwd.push_back(
                reorder(relu1_diff_src_memory, fc1_diff_dst_memory));
        net_bwd_args.push_back({{DNNL_ARG_FROM, relu1_diff_src_memory},
                {DNNL_ARG_TO, fc1_diff_dst_memory}});
    }

    // create backward fc2olution primitive
    net_bwd.push_back(inner_product_backward_weights(fc1_bwd_weights_pd));
    net_bwd_args.push_back({{DNNL_ARG_SRC, fc1_bwd_src_memory},
            {DNNL_ARG_DIFF_DST, fc1_diff_dst_memory},
            // delay putting DIFF_WEIGHTS until reorder (if needed)
            {DNNL_ARG_DIFF_BIAS, fc1_diff_bias_memory}});

    // create reorder primitives between fc2 diff weights and user diff weights
    // if needed
    auto fc1_diff_weights_memory = fc1_user_diff_weights_memory;
    if (fc1_bwd_weights_pd.diff_weights_desc()
            != fc1_user_diff_weights_memory.get_desc()) {
        fc1_diff_weights_memory
                = memory(fc1_bwd_weights_pd.diff_weights_desc(), eng);
        net_bwd_args.back().insert(
                {DNNL_ARG_DIFF_WEIGHTS, fc1_diff_weights_memory});

        net_bwd.push_back(reorder(
                fc1_diff_weights_memory, fc1_user_diff_weights_memory));
        net_bwd_args.push_back({{DNNL_ARG_FROM, fc1_diff_weights_memory},
                {DNNL_ARG_TO, fc1_user_diff_weights_memory}});
    } else {
        net_bwd_args.back().insert(
                {DNNL_ARG_DIFF_WEIGHTS, fc1_diff_weights_memory});
    }



    // didn't we forget anything?
    assert(net_fwd.size() == net_fwd_args.size() && "something is missing");
    assert(net_bwd.size() == net_bwd_args.size() && "something is missing");

    int n_iter = 2; // number of iterations for training
    // execute
    while (n_iter) {
        // forward
        // for(int i=0; i<feature_num; i++){
        //     printf("%f ", user_src[i]);
        // }
        // printf("\n");
        // for(int i=0; i<10; i++){
        //     printf("%f ", fc2_weights[i]);
        // }
        // printf("\n");
        for (size_t i = 0; i < net_fwd.size(); ++i)
            net_fwd.at(i).execute(s, net_fwd_args.at(i));

        // for(int i=0; i<10; i++){
        //     printf("%f ",((float*)fc1_dst_memory.get_data_handle())[i]);
        // }
        // printf("\n");
        // printf("%f\n",((float*)fc2_dst_memory.get_data_handle())[0]);
        // update net_diff_dst
        // auto net_output = fc2_user_dst_memory.get_data_handle();
        // ..user updates net_diff_dst using net_output...
        // some user defined func update_diff_dst(net_diff_dst.data(),
        // net_output)

        for (size_t i = 0; i < net_bwd.size(); ++i)
            net_bwd.at(i).execute(s, net_bwd_args.at(i));

        // printf("%f\n", ((float*)fc2_diff_weights_memory.get_data_handle())[0]);
        // update weights and bias using diff weights and bias
        //
        // auto net_diff_weights
        //     = fc1_user_diff_weights_memory.get_data_handle();
        // auto net_diff_bias = fc1_diff_bias_memory.get_data_handle();
        //
        // ...user updates weights and bias using diff weights and bias...
        //
        // some user defined func update_weights(fc1_weights.data(),
        // fc1_bias.data(), net_diff_weights, net_diff_bias);

        --n_iter;
    }

    s.wait();
}

extern "C" int cnn_inference_f32_cpp() {
    try {
        int times = 2;   //SGX: change from 100 -> 1
        simple_net(parse_engine_kind(1, NULL), times, 6);
        printf("Intel(R) DNNL: cnn_inference_f32.cpp: passed\n");
    } catch (error &e) {
        // printf("%x\n", e);
        printf("Intel(R) DNNL: cnn_inference_f32.cpp: failed!!!\n");
    }
    return 0;
}