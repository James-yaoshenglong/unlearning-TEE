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

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <vector>

# include <unistd.h>
# include <pwd.h>
# define MAX_PATH FILENAME_MAX

#include "sgx_urts.h"
#include "App.h"
#include "Enclave_u.h"
#include "data_structure.hpp"
#include "xxhash64.h"
// #include "merklecpp.h"
#include "merkletree.h"

/* Global EID shared by multiple threads */
sgx_enclave_id_t global_eid = 0;

/* intermedian data storage */
float * data;
float * label; 
int row;
int col;

typedef struct _sgx_errlist_t {
    sgx_status_t err;
    const char *msg;
    const char *sug; /* Suggestion */
} sgx_errlist_t;

/* Error code returned by sgx_create_enclave */
static sgx_errlist_t sgx_errlist[] = {
    {
        SGX_ERROR_UNEXPECTED,
        "Unexpected error occurred.",
        NULL
    },
    {
        SGX_ERROR_INVALID_PARAMETER,
        "Invalid parameter.",
        NULL
    },
    {
        SGX_ERROR_OUT_OF_MEMORY,
        "Out of memory.",
        NULL
    },
    {
        SGX_ERROR_ENCLAVE_LOST,
        "Power transition occurred.",
        "Please refer to the sample \"PowerTransition\" for details."
    },
    {
        SGX_ERROR_INVALID_ENCLAVE,
        "Invalid enclave image.",
        NULL
    },
    {
        SGX_ERROR_INVALID_ENCLAVE_ID,
        "Invalid enclave identification.",
        NULL
    },
    {
        SGX_ERROR_INVALID_SIGNATURE,
        "Invalid enclave signature.",
        NULL
    },
    {
        SGX_ERROR_OUT_OF_EPC,
        "Out of EPC memory.",
        NULL
    },
    {
        SGX_ERROR_NO_DEVICE,
        "Invalid SGX device.",
        "Please make sure SGX module is enabled in the BIOS, and install SGX driver afterwards."
    },
    {
        SGX_ERROR_MEMORY_MAP_CONFLICT,
        "Memory map conflicted.",
        NULL
    },
    {
        SGX_ERROR_INVALID_METADATA,
        "Invalid enclave metadata.",
        NULL
    },
    {
        SGX_ERROR_DEVICE_BUSY,
        "SGX device was busy.",
        NULL
    },
    {
        SGX_ERROR_INVALID_VERSION,
        "Enclave version was invalid.",
        NULL
    },
    {
        SGX_ERROR_INVALID_ATTRIBUTE,
        "Enclave was not authorized.",
        NULL
    },
    {
        SGX_ERROR_ENCLAVE_FILE_ACCESS,
        "Can't open enclave file.",
        NULL
    },
    {
        SGX_ERROR_NDEBUG_ENCLAVE,
        "The enclave is signed as product enclave, and can not be created as debuggable enclave.",
        NULL
    },
};

/* Check error conditions for loading enclave */
void print_error_message(sgx_status_t ret)
{
    size_t idx = 0;
    size_t ttl = sizeof sgx_errlist/sizeof sgx_errlist[0];

    for (idx = 0; idx < ttl; idx++) {
        if(ret == sgx_errlist[idx].err) {
            if(NULL != sgx_errlist[idx].sug)
                printf("Info: %s\n", sgx_errlist[idx].sug);
            printf("Error: %s\n", sgx_errlist[idx].msg);
            break;
        }
    }
    
    if (idx == ttl)
        printf("Error: Unexpected error occurred.\n");
}

// extern "C"{
/* Initialize the enclave:
 *   Call sgx_create_enclave to initialize an enclave instance
 */
int initialize_enclave(void)
{
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;
    
    /* Call sgx_create_enclave to initialize an enclave instance */
    /* Debug Support: set 2nd parameter to 1 */
    ret = sgx_create_enclave(ENCLAVE_FILENAME, SGX_DEBUG_FLAG, NULL, NULL, &global_eid, NULL);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        return -1;
    }
    printf("Creating Enclave with id: %ld\n", global_eid);
    return 0;
}
// }

void destroy_enclave(void)
{
    // std::cout << "Destroying Enclave with id: " << eid << std::endl;
    printf("Destroying Enclave with id: %ld\n", global_eid);
    sgx_destroy_enclave(global_eid);
}


void ocall_get_time(double* current){
    const double MILLION = 1000000.0;
    const double THOUSAND = 1000.0;
    struct timespec start;
    clock_gettime(CLOCK_REALTIME, &start);
    *current = start.tv_sec*MILLION+start.tv_nsec / THOUSAND;
}


void test_merkle_tree(){
    char buffer[HASH_LENGTH];
    // merkle::Tree tree;
    double start, end;
    // ocall_get_time(&start);
    // for(int i=0; i<row; i++){
    //     sprintf(buffer, "fa8f44eabb728d4020e7f33d1aa973faaef19de6c06679bccdc5100a3c0%05d", i);
    //     merkle::Tree::Hash hash(buffer);
    //     tree.insert(hash);
    // }
    // ocall_get_time(&end);
    // printf("Total insert time for %d is %.8f ms and each need %.8f ms\n", row, end-start, (end-start)/row);

    // auto root = tree.root();
    // ocall_get_time(&start);
    // for(int i=0; i<row; i++){
    //     auto path = tree.path(0);
    //     path->verify(root);
    // }
    // ocall_get_time(&end);
    // printf("Total verify time for %d is %.8f ms and each need %.8f ms\n", row, end-start, (end-start)/row);
    mt_t* tree = mt_create();
    ocall_get_time(&start);
    for(int i=0; i<row; i++){
        sprintf(buffer, "aaaabbbbccccddddeeeeffffaa%05d", i);
        mt_add(tree, (const unsigned char*)buffer, HASH_LENGTH);
    }
    ocall_get_time(&end);
    printf("Total insert time for %d is %.8f ms and each need %.8f ms\n", row, end-start, (end-start)/row);
    printf("after insertion size is %d\n", mt_get_size(tree));

    ocall_get_time(&start);
    for(int i=0; i<row; i++){
        sprintf(buffer, "aaaabbbbccccddddeeeeffffaa%05d", i);
        mt_verify(tree, (const unsigned char*)buffer, HASH_LENGTH, i);
    }
    ocall_get_time(&end);
    printf("Total verify time for %d is %.8f ms and each need %.8f ms\n", row, end-start, (end-start)/row);

    //use update replace delete
    ocall_get_time(&start);
    for(int i=0; i<row; i++){
        sprintf(buffer, "%031d", 0);
        mt_update(tree, (const unsigned char*)buffer, HASH_LENGTH, i);
    }
    ocall_get_time(&end);
    printf("Total delete time for %d is %.8f ms and each need %.8f ms\n", row, end-start, (end-start)/row);

    mt_delete(tree);
}

void load_data(float* input_data, float* input_label, int r, int c){ //pay attention to double ** and double [][]
    data = (float*)malloc(r*c*sizeof(float));
    label = (float*)malloc(r*sizeof(float));
    memcpy(data, input_data, r*c*sizeof(float));
    memcpy(label, input_label, r*sizeof(float));
    row = r;
    col = c;
    printf("data loaded into intermedian storage, size is (%d, %d)\n", r, c);
    test_merkle_tree();
}

void init_enclave_storage(){
    ecall_init_enclave_storage(global_eid, data, label, row, col, global_eid);
    sgx_status_t ret = SGX_SUCCESS;
    int retval = 0;
    // cnn_inference_f32_cpp(global_eid, &retval);
    ecall_training(global_eid);
    if(ret != SGX_SUCCESS){
        print_error_message(ret);
    }
}

void unlearning(uint64_t kid){
    sgx_status_t ret = SGX_SUCCESS;
    int retval = 0;
    ecall_unlearning(global_eid, kid);
    if(ret != SGX_SUCCESS){
        print_error_message(ret);
    }
}

void ocall_init_model_storage(void** model, int* network, int len){
    Model** temp = (Model**)model;
    Model* result = new Model(network, len);
    *temp = result;
}

void predict(float* data, float* label, int size){
    ecall_predict(global_eid, data, label, size);
}

/* OCall functions */
void ocall_print_string(const char *str)
{
    /* Proxy/Bridge will check the length and null-terminate 
     * the input string to prevent buffer overflow. 
     */
    printf("%s", str);
}


/* Application entry */
int SGX_CDECL main(int argc, char *argv[])
{
    (void)(argc);
    (void)(argv);


    /* Initialize the enclave */
    if(initialize_enclave() < 0){
        printf("Enter a character before exit ...\n");
        getchar();
        return -1; 
    }
 
    
    /* Utilize trusted libraries */ 
    ecall_libcxx_functions(global_eid);
    
    /* Destroy the enclave */
    sgx_destroy_enclave(global_eid);
    
    printf("Info: Cxx11DemoEnclave successfully returned.\n");

    //printf("Enter a character before exit ...\n");
    //getchar();
    return 0;
}


uint64_t xxhash(char* content, int len){
    return XXHash64::hash(content, len, 1);
}

