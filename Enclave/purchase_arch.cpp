#include <assert.h>

#include <chrono>
#include <numeric>
#include <vector>
#include <unordered_map>

#include "example_utils.hpp"
#include "Enclave.h"
#include "purchase_arch.hpp"

using namespace dnnl;

using namespace std;
using tag = memory::format_tag;
using dt = memory::data_type;

MLP::MLP(int arch[3], float a, int b){
    eng = engine(parse_engine_kind(1, NULL), 0);
    s = stream(eng);
    memcpy(network, arch, 3*sizeof(int));
    alpha = a;
    batch = b;
    // printf("alpha is %f\n", alpha);

    // const memory::dim batch = b;
    const memory::dim feature_num = network[0];
    const memory::dim fc1_node_num = network[1];

    // fc1 inner product {batch, feature_num} (x) {1000, feature_num}-> {batch, 1000}
    memory::dims fc1_src_tz = {batch, feature_num};
    memory::dims fc1_weights_tz = {fc1_node_num, feature_num};
    memory::dims fc1_bias_tz = {fc1_node_num};
    memory::dims fc1_dst_tz = {batch, fc1_node_num};

    /// Allocate buffers for input and output data, weights, and bias.
    /// @snippet cnn_inference_f32.cpp Allocate buffers
    //[Allocate buffers]
    user_src = vector<float>(batch * feature_num); //batch * feature_num
    user_dst = vector<float>(batch * network[2]);
    //[Allocate buffers]

    user_src_memory = memory({{fc1_src_tz}, dt::f32, tag::nc}, eng);

    fc1_weights = vector<float>(product(fc1_weights_tz));
    fc1_bias = vector<float>(product(fc1_bias_tz));

    fc1_user_weights_memory
            = memory({{fc1_weights_tz}, dt::f32, tag::nc}, eng);
    fc1_user_bias_memory = memory({{fc1_bias_tz}, dt::f32, tag::x}, eng);

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
    auto fc1_weights_memory = fc1_user_weights_memory;
    // if (fc1_prim_desc.weights_desc() != fc1_user_weights_memory.get_desc()) {
    //     fc1_weights_memory = memory(fc1_prim_desc.weights_desc(), eng);
    //     reorder(fc1_user_weights_memory, fc1_weights_memory)
    //             .execute(s, fc1_user_weights_memory, fc1_weights_memory);
    // }
    fc1_weights_memory = memory(fc1_prim_desc.weights_desc(), eng);
    net_fwd.push_back(reorder(fc1_user_weights_memory, fc1_weights_memory));
    net_fwd_args.push_back({{DNNL_ARG_FROM, fc1_user_weights_memory},
            {DNNL_ARG_TO,  fc1_weights_memory}});


    auto fc1_dst_memory = memory(fc1_prim_desc.dst_desc(), eng);

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
    // auto relu1_desc = eltwise_forward::desc(prop_kind::forward_inference,
    //         algorithm::eltwise_relu, fc1_dst_memory.get_desc(),
    //         negative1_slope);
    auto relu1_desc = eltwise_forward::desc(prop_kind::forward_inference,
        algorithm::eltwise_tanh, fc1_dst_memory.get_desc(),
        negative1_slope);
    auto relu1_prim_desc = eltwise_forward::primitive_desc(relu1_desc, eng);

    auto relu1_dst_memory = memory(fc1_dst_memory.get_desc(), eng);

    net_fwd.push_back(eltwise_forward(relu1_prim_desc));
    net_fwd_args.push_back({{DNNL_ARG_SRC, fc1_dst_memory},
            {DNNL_ARG_DST, relu1_dst_memory}});

    // fc2 inner product {batch, 1000} (x) {1, 1000}-> {batch, 1}
    memory::dims fc2_weights_tz = {network[2], fc1_node_num};
    memory::dims fc2_bias_tz = {network[2]};
    memory::dims fc2_dst_tz = {batch, network[2]};

    fc2_weights = vector<float>(product(fc2_weights_tz));
    fc2_bias = vector<float>(product(fc2_bias_tz));

    fc2_user_weights_memory
            = memory({{fc2_weights_tz}, dt::f32, tag::nc}, eng);
    fc2_user_bias_memory = memory({{fc2_bias_tz}, dt::f32, tag::x}, eng);
    user_dst_memory = memory({{fc2_dst_tz}, dt::f32, tag::nc}, eng);

        // create memory descriptors for fc2olution data w/ no specified format
    auto fc2_bias_md = memory::desc({fc2_bias_tz}, dt::f32, tag::any);
    auto fc2_weights_md = memory::desc({fc2_weights_tz}, dt::f32, tag::any);
    auto fc2_dst_md = memory::desc({fc2_dst_tz}, dt::f32, tag::any);

    // create a inner_product
    auto fc2_desc = inner_product_forward::desc(prop_kind::forward_inference,
            fc1_dst_memory.get_desc(), fc2_weights_md, fc2_bias_md, fc2_dst_md);
    auto fc2_prim_desc = inner_product_forward::primitive_desc(fc2_desc, eng);

    auto fc2_weights_memory = fc2_user_weights_memory;
    fc2_weights_memory = memory(fc2_prim_desc.weights_desc(), eng);
    net_fwd.push_back(reorder(fc2_user_weights_memory, fc2_weights_memory));
    net_fwd_args.push_back({{DNNL_ARG_FROM, fc2_user_weights_memory},
            {DNNL_ARG_TO,  fc2_weights_memory}});


    auto fc2_dst_memory = memory(fc2_prim_desc.dst_desc(), eng);
    // auto fc2_dst_memory = user_dst_memory;

    // create fc2olution primitive and add it to net
    net_fwd.push_back(inner_product_forward(fc2_prim_desc));
    net_fwd_args.push_back({{DNNL_ARG_SRC, relu1_dst_memory},
            {DNNL_ARG_WEIGHTS, fc2_weights_memory},
            {DNNL_ARG_BIAS, fc2_user_bias_memory},
            {DNNL_ARG_DST, fc2_dst_memory}});


    // relu2
    // {batch, 1000} -> {batch, 1000}
    // const float negative1_slope = 1.0f;

    // create relu primitive and add it to net
    // auto relu1_desc = eltwise_forward::desc(prop_kind::forward_inference,
    //         algorithm::eltwise_relu, fc1_dst_memory.get_desc(),
    //         negative1_slope);
    auto relu2_desc = eltwise_forward::desc(prop_kind::forward_inference,
        algorithm::eltwise_logistic, fc2_dst_memory.get_desc(),
        negative1_slope);
    auto relu2_prim_desc = eltwise_forward::primitive_desc(relu2_desc, eng);

    auto relu2_dst_memory = user_dst_memory;

    net_fwd.push_back(eltwise_forward(relu2_prim_desc));
    net_fwd_args.push_back({{DNNL_ARG_SRC, fc2_dst_memory},
            {DNNL_ARG_DST, relu2_dst_memory}});




    //-----------------------------------------------------------------------
    //----------------- Backward Stream -------------------------------------
    // ... user diff_data ...
    net_diff_dst = vector<float>(batch * network[2]);
    fc2_user_diff_dst_memory
        = memory({{fc2_dst_tz}, dt::f32, tag::nc}, eng);

    // Backward relu
    // auto relu1_diff_dst_md = memory::desc({relu1_data_tz}, dt::f32, tag::any);
    auto relu2_diff_dst_md = fc2_prim_desc.dst_desc();
    auto relu2_src_md = fc2_prim_desc.dst_desc();

    // create backward relu primitive_descriptor
    // auto relu1_bwd_desc = eltwise_backward::desc(algorithm::eltwise_relu,
    //         relu1_diff_dst_md, relu1_src_md, negative1_slope);
    auto relu2_bwd_desc = eltwise_backward::desc(algorithm::eltwise_logistic,
            relu2_diff_dst_md, relu2_src_md, negative1_slope);
    auto relu2_bwd_pd
            = eltwise_backward::primitive_desc(relu2_bwd_desc, eng, relu2_prim_desc);

    // create reorder primitive between lrn diff src and relu diff dst
    // if required
    auto relu2_diff_dst_memory = fc2_user_diff_dst_memory;
    relu2_diff_dst_memory = memory(relu2_bwd_pd.diff_dst_desc(), eng);
    net_bwd.push_back(reorder(fc2_user_diff_dst_memory, relu2_diff_dst_memory));
    net_bwd_args.push_back({{DNNL_ARG_FROM, fc2_user_diff_dst_memory},
            {DNNL_ARG_TO, relu2_diff_dst_memory}});

    // create memory for relu diff src
    auto relu2_diff_src_memory = memory(relu2_bwd_pd.diff_src_desc(), eng);

    // finally create a backward relu primitive
    net_bwd.push_back(eltwise_backward(relu2_bwd_pd));
    net_bwd_args.push_back({{DNNL_ARG_SRC, fc2_dst_memory},
            {DNNL_ARG_DIFF_DST, relu2_diff_dst_memory},
            {DNNL_ARG_DIFF_SRC, relu2_diff_src_memory}});



    // fc2_user_diff_dst_memory
    //         = memory({{fc2_dst_tz}, dt::f32, tag::nc}, eng);
    fc2_user_diff_weights_buffer = vector<float>(product(fc2_weights_tz));
    fc2_diff_bias_buffer = vector<float>(product(fc2_bias_tz));
    fc2_user_diff_weights_memory
            = memory({{fc2_weights_tz}, dt::f32, tag::nc}, eng);
    fc2_diff_bias_memory = memory({{fc2_bias_tz}, dt::f32, tag::x}, eng);


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
    
    auto fc2_bwd_src_memory = relu1_dst_memory;
    fc2_bwd_src_memory = memory(fc2_bwd_weights_pd.src_desc(), eng);
    net_bwd.push_back(reorder(relu1_dst_memory, fc2_bwd_src_memory));
    net_bwd_args.push_back({{DNNL_ARG_FROM, relu1_dst_memory},
            {DNNL_ARG_TO, fc2_bwd_src_memory}});
    // auto fc2_diff_dst_memory = fc2_user_diff_dst_memory;
    auto fc2_diff_dst_memory = relu2_diff_src_memory;
    fc2_diff_dst_memory = memory(fc2_dst_memory.get_desc(), eng);
    net_bwd.push_back(
            reorder(relu2_diff_src_memory, fc2_diff_dst_memory));
    net_bwd_args.push_back({{DNNL_ARG_FROM, relu2_diff_src_memory},
            {DNNL_ARG_TO, fc2_diff_dst_memory}});

    // create backward fc2olution primitive
    net_bwd.push_back(inner_product_backward_weights(fc2_bwd_weights_pd));
    net_bwd_args.push_back({{DNNL_ARG_SRC, fc2_bwd_src_memory},
            {DNNL_ARG_DIFF_DST, fc2_diff_dst_memory},
            // delay putting DIFF_WEIGHTS until reorder (if needed)
            {DNNL_ARG_DIFF_BIAS, fc2_diff_bias_memory}});
    auto fc2_diff_weights_memory = fc2_user_diff_weights_memory;
    fc2_diff_weights_memory
        = memory(fc2_bwd_weights_pd.diff_weights_desc(), eng);
    net_bwd_args.back().insert(
        {DNNL_ARG_DIFF_WEIGHTS, fc2_diff_weights_memory});
    net_bwd.push_back(reorder(
            fc2_diff_weights_memory, fc2_user_diff_weights_memory));
    net_bwd_args.push_back({{DNNL_ARG_FROM, fc2_diff_weights_memory},
            {DNNL_ARG_TO, fc2_user_diff_weights_memory}});


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
    // auto relu1_bwd_desc = eltwise_backward::desc(algorithm::eltwise_relu,
    //         relu1_diff_dst_md, relu1_src_md, negative1_slope);
    auto relu1_bwd_desc = eltwise_backward::desc(algorithm::eltwise_tanh,
            relu1_diff_dst_md, relu1_src_md, negative1_slope);
    auto relu1_bwd_pd
            = eltwise_backward::primitive_desc(relu1_bwd_desc, eng, relu1_prim_desc);

    // create reorder primitive between lrn diff src and relu diff dst
    // if required
    auto relu1_diff_dst_memory = fc2_diff_src_memory;
    relu1_diff_dst_memory = memory(relu1_bwd_pd.diff_dst_desc(), eng);
    net_bwd.push_back(reorder(fc2_diff_src_memory, relu1_diff_dst_memory));
    net_bwd_args.push_back({{DNNL_ARG_FROM, fc2_diff_src_memory},
            {DNNL_ARG_TO, relu1_diff_dst_memory}});

    // create memory for relu diff src
    auto relu1_diff_src_memory = memory(relu1_bwd_pd.diff_src_desc(), eng);

    // finally create a backward relu primitive
    net_bwd.push_back(eltwise_backward(relu1_bwd_pd));
    net_bwd_args.push_back({{DNNL_ARG_SRC, fc1_dst_memory},
            {DNNL_ARG_DIFF_DST, relu1_diff_dst_memory},
            {DNNL_ARG_DIFF_SRC, relu1_diff_src_memory}});

    // Backward inner_product with respect to weights
    // create user format diff weights and diff bias memory
    fc1_user_diff_weights_buffer = vector<float>(product(fc1_weights_tz));
    fc1_diff_bias_buffer = vector<float>(product(fc1_bias_tz));

    fc1_user_diff_weights_memory
            = memory({{fc1_weights_tz}, dt::f32, tag::nc}, eng);
    fc1_diff_bias_memory = memory({{fc1_bias_tz}, dt::f32, tag::x}, eng);

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

    auto fc1_bwd_src_memory = fc1_src_memory;
    fc1_bwd_src_memory = memory(fc1_bwd_weights_pd.src_desc(), eng);
    net_bwd.push_back(reorder(fc1_src_memory, fc1_bwd_src_memory));
    net_bwd_args.push_back({{DNNL_ARG_FROM, fc1_src_memory},
            {DNNL_ARG_TO, fc1_bwd_src_memory}});

    auto fc1_diff_dst_memory = relu1_diff_src_memory;
    fc1_diff_dst_memory = memory(fc1_dst_memory.get_desc(), eng);
    net_bwd.push_back(
            reorder(relu1_diff_src_memory, fc1_diff_dst_memory));
    net_bwd_args.push_back({{DNNL_ARG_FROM, relu1_diff_src_memory},
            {DNNL_ARG_TO, fc1_diff_dst_memory}});


    // create backward fc2olution primitive
    net_bwd.push_back(inner_product_backward_weights(fc1_bwd_weights_pd));
    net_bwd_args.push_back({{DNNL_ARG_SRC, fc1_bwd_src_memory},
            {DNNL_ARG_DIFF_DST, fc1_diff_dst_memory},
            // delay putting DIFF_WEIGHTS until reorder (if needed)
            {DNNL_ARG_DIFF_BIAS, fc1_diff_bias_memory}});
    auto fc1_diff_weights_memory = fc1_user_diff_weights_memory;
    fc1_diff_weights_memory
            = memory(fc1_bwd_weights_pd.diff_weights_desc(), eng);
    net_bwd_args.back().insert(
                {DNNL_ARG_DIFF_WEIGHTS, fc1_diff_weights_memory});
    net_bwd.push_back(reorder(
            fc1_diff_weights_memory, fc1_user_diff_weights_memory));
    net_bwd_args.push_back({{DNNL_ARG_FROM, fc1_diff_weights_memory},
            {DNNL_ARG_TO, fc1_user_diff_weights_memory}});
    
    // didn't we forget anything?
    assert(net_fwd.size() == net_fwd_args.size() && "something is missing");
    assert(net_bwd.size() == net_bwd_args.size() && "something is missing");
}

void MLP::forward(const vector<float>& input){
    user_src = vector<float>(input);
    write_to_dnnl_memory(user_src.data(), user_src_memory);
    write_to_dnnl_memory(fc1_weights.data(), fc1_user_weights_memory);
    write_to_dnnl_memory(fc1_bias.data(), fc1_user_bias_memory);
    write_to_dnnl_memory(fc2_weights.data(), fc2_user_weights_memory);
    write_to_dnnl_memory(fc2_bias.data(), fc2_user_bias_memory);

    for (size_t i = 0; i < net_fwd.size(); ++i)
        net_fwd.at(i).execute(s, net_fwd_args.at(i));
    read_from_dnnl_memory(user_dst.data(), user_dst_memory);
    // printf("sigmoid output is %f\n", ((float*)net_fwd_args.at(5)[DNNL_ARG_DST].get_data_handle())[0]);
    // printf("x is %f\n", ((float*)net_fwd_args.at(4)[DNNL_ARG_SRC].get_data_handle())[0]);

}

void MLP::backward(const vector<float>& label){
    for(int i=0; i<batch; i++){
        float y = label[i];
        // net_diff_dst[i] = label[i]-user_dst[i];
        // net_diff_dst[i] = 0.05f;
        net_diff_dst[i] = -y/user_dst[i]+(1-y)/(1-user_dst[i]);
        // net_diff_dst[i*2+1] = -y/user_dst[i*2+1]-(1-y)/(1-user_dst[i*2+1]);
        // net_diff_dst[i*2] = 100.0;
        // net_diff_dst[i*2+1] = 10.0;
    }
    // printf("batch is %d\n", batch);
    // printf("label is %f\n", label[100]);
    // printf("user dst is %f\n", user_dst[0]);
    // printf("diff is %f\n", net_diff_dst[0]);
    write_to_dnnl_memory(net_diff_dst.data(), fc2_user_diff_dst_memory);
    for (size_t i = 0; i < net_bwd.size(); ++i){
            net_bwd.at(i).execute(s, net_bwd_args.at(i));
    }
    // printf("sigmoid source diff is %.12f\n", ((float*)net_bwd_args.at(1)[DNNL_ARG_DIFF_SRC].get_data_handle())[0]);

    // printf("x in backward is %f\n", ((float*)net_bwd_args.at(4)[DNNL_ARG_SRC].get_data_handle())[0]);
    // printf("dst diff is %f\n", ((float*)net_bwd_args.at(4)[DNNL_ARG_DIFF_DST].get_data_handle())[0]);
    // printf("weight diff is %f\n", ((float*)net_bwd_args.at(4)[DNNL_ARG_DIFF_WEIGHTS].get_data_handle())[0]);
    // printf("bias diff is %f\n", ((float*)net_bwd_args.at(4)[DNNL_ARG_DIFF_BIAS].get_data_handle())[0]);

    // printf("fc2 src diff is %f\n", ((float*)net_bwd_args.at(6)[DNNL_ARG_DIFF_SRC].get_data_handle())[0]);

    read_from_dnnl_memory(fc1_user_diff_weights_buffer.data(),
            fc1_user_diff_weights_memory);
    read_from_dnnl_memory(fc1_diff_bias_buffer.data(), fc1_diff_bias_memory);
    read_from_dnnl_memory(fc2_user_diff_weights_buffer.data(),
            fc2_user_diff_weights_memory);
    read_from_dnnl_memory(fc2_diff_bias_buffer.data(), fc2_diff_bias_memory);

    for(int i=0; i<fc1_weights.size(); i++){
        fc1_weights[i]-=alpha*fc1_user_diff_weights_buffer[i]/batch;
    }
    for(int i=0; i<fc1_bias.size(); i++){
        fc1_bias[i]-=alpha*fc1_diff_bias_buffer[i]/batch;
    }
    for(int i=0; i<fc2_weights.size(); i++){
        fc2_weights[i]-=alpha*fc2_user_diff_weights_buffer[i]/batch;
    }
    for(int i=0; i<fc2_bias.size(); i++){
        fc2_bias[i]-=alpha*fc2_diff_bias_buffer[i]/batch;
    }
    // printf("%f\n", fc1_weights[0]);
}

void MLP::train(float* data, float* label, int epoch, int size, Model* model){
    //current no shuffle, shuffle use twice much space
    // printf("size is %d\n", size);
    int temp = batch;
    for(int i=0; i<epoch; i++){
        for(int j=0; j<size; j+=batch){
            int start = j;
            int end = j+batch<size?j+batch:size;
            batch = end-start;
            // printf("start is %d, end is %d\n", start, end);
            vector<float>input(data+start*network[0], data+end*network[0]);
            // printf("label is %f\n", *(label+end-1));
            vector<float>output(label+start, label+end);
			
			//deal with batch size different or directly discard
			if(batch != temp){
				int arch[3] = {600, 128, 1};
				MLP another = MLP(arch, alpha, batch);
				saveModel(model);
				another.setModel(model);
				another.forward(input);
                another.backward(output);
				another.saveModel(model);
				setModel(model);
				batch = temp;
				continue;
			}
            try {
                forward(input);
                backward(output);
                // printf("Intel(R) DNNL: cnn_inference_f32.cpp: passed\n");
            } catch (error &e) {
                // printf("%x\n", e);
                printf("Intel(R) DNNL: cnn_inference_f32.cpp: failed!!!\n");
            }
            batch = temp;
        }
    }
}

void MLP::setModel(Model* model){
    fc1_weights = vector<float>(model->fc1w, model->fc1b);
    fc1_bias = vector<float>(model->fc1b, model->fc2w);
    fc2_weights = vector<float>(model->fc2w, model->fc2b);
    fc2_bias = vector<float>(model->fc2b, model->fc2b+network[2]);
}
void MLP::saveModel(Model* model){
    memcpy(model->fc1w, fc1_weights.data(), fc1_weights.size()*sizeof(float));
    memcpy(model->fc1b, fc1_bias.data(), fc1_bias.size()*sizeof(float));
    memcpy(model->fc2w, fc2_weights.data(), fc2_weights.size()*sizeof(float));
    memcpy(model->fc2b, fc2_bias.data(), fc2_bias.size()*sizeof(float));
}

vector<float> MLP::inference(vector<float>& input){
    int temp = batch;
    batch = input.size()/network[0];
    forward(input);
    vector<float> result(batch);
    for(int i=0; i<batch; i++){
        result[i] = user_dst[i]>0.5f?1.0f:0.0f;
    }
    // printf("result is %f %f\n", user_dst[0], user_dst[1]);
    batch = temp;
    return result; 
}