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

#include <stdarg.h>
#include <stdio.h>      /* vsnprintf */
#include <map>
#include <vector>
#include <algorithm>
#include <sgx_trts.h>

#include "Enclave.h"
#include "Enclave_t.h"  /* print_string */

#include "xxhash64.h"
#include "cuckoofilter.h"
#include "sha256.h"
#include "data_structure.hpp"

using cuckoofilter::CuckooFilter;

class Key{
public:
    Key(){}

    Key(float* dptr, float*lptr, Model* model_link, int col, int slice_num){
        dataPtr = dptr;
        labelPtr = lptr;
        modelPtr = model_link;
        slice = slice_num;
        len = (col+1)*sizeof(float);
        // kid = hash(dataPtr);
        float* temp = (float*)malloc((col+1)*sizeof(float*));
        memcpy(temp, dataPtr, col*sizeof(float));
        memcpy(temp+col, labelPtr, sizeof(float));
        kid = XXHash64::hash(temp, len, 1); //here need set seed
        tag = 1;
        // seed = rand(); //here has problem, need set random seed
        sgx_read_rand((unsigned char *)&seed, 4);
    }

    uint64_t getKid(){
        return kid;
    }

    float* getDataPtr(){
        return dataPtr;
    }

    float* getLabelPtr(){
        return labelPtr;
    }

    void setTag(int num){
        tag = num;
    }

private:
    uint64_t kid;
    uint64_t len;
    uint32_t tag;
    uint32_t seed;
    uint32_t slice;
    float* dataPtr;
    float* labelPtr;
    Model* modelPtr;
};

std::map<uint64_t, Key*> keyMap; //here do not know whether it is in the enclave and use enclave std lib
std::vector<Key*> keyList;
CuckooFilter<uint64_t, 12> filter(100000);

int network[] = {1, 128, 2};
int slice_size = 1;
int model_num;
std::vector<Model*> model_storage;
int model_size = 0;

int r;
int c;
uint64_t eid;

float* enclave_data_storage;
float* enclave_label_storage;


/* 
 * printf: 
 *   Invokes OCALL to display the enclave buffer to the terminal.
 */
void printf(const char *fmt, ...)
{
    char buf[BUFSIZ] = {'\0'};
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap);
    ocall_print_string(buf);
}

void ecall_libcxx_functions(void){
    int c = 1+1;
    return;
}

uint64_t xxsha256(Key* key, int col, uint64_t enclave_id){
    int hashStringBUfferLen = sizeof(float)*(col+1)+sizeof(uint64_t)+sizeof(uint64_t)+1;
    char* hashStringBuffer = (char*)malloc(hashStringBUfferLen);
    uint64_t kid = key->getKid();
    memcpy(hashStringBuffer, &kid, sizeof(uint64_t));
    memcpy(hashStringBuffer+sizeof(uint64_t), key->getDataPtr(), sizeof(float)*col);
    memcpy(hashStringBuffer+sizeof(float)*col+sizeof(uint64_t), key->getLabelPtr(), sizeof(float));
    memcpy(hashStringBuffer+sizeof(float)*(col+1)+sizeof(uint64_t), &enclave_id, sizeof(uint64_t));
    hashStringBuffer[hashStringBUfferLen-1] = '\0';
    
    char hashBuffer[33];
    
    sha256_string(hashStringBuffer, hashStringBUfferLen, hashBuffer);
    

    uint64_t result = XXHash64::hash(hashBuffer, 33, 1);

    free(hashStringBuffer);

    return result;
}

void ecall_init_enclave_storage(float* input_data, float* input_label, int row, int col, uint64_t enclave_id){
    r = row;
    c = col;
    eid = enclave_id;

    // define the nework parameter
    network[0] = col;
    model_num = (row % slice_size + row) / slice_size;
    
    //initialize the model storage
    for(int i=0; i<model_num+1; i++){
        Model* temp;
        ocall_init_model_storage((void**)&temp, network, 3);
        model_storage.push_back(temp);
    }

    sgx_read_rand((unsigned char *)(model_storage[0]->storage), model_size);

    //initialize the key list
    for(int i=0; i < row; i++){
        Key* key = new Key(input_data+(col*i), input_label+i, model_storage[i/slice_size], col, i/slice_size);
        // printf("%f\n", input[i][0]);
        // printf("%f\n", key->getDataPtr()[0]);
        keyMap[key->getKid()] = key;
        keyList.push_back(key);
        uint64_t hash = xxsha256(key, col, enclave_id);

        filter.Add(hash);
        // printf("%d", filter.Contain(hash) == cuckoofilter::Ok);
    }
}

void ecall_training(){
    //load whole data
    enclave_data_storage = (float*)malloc(r*c*sizeof(float));
    enclave_label_storage = (float*)malloc(r*sizeof(float));
    int count = 0;
    for(int i=0; i<keyList.size(); i+=1){
        Key* key = keyList[i];
        uint64_t hash = xxsha256(key, c, eid);
        if(filter.Contain(hash) == cuckoofilter::Ok){
            memcpy(enclave_data_storage+i*c*sizeof(float), key->getDataPtr(), c*sizeof(float));
            memcpy(enclave_label_storage+i*sizeof(float), key->getLabelPtr(), sizeof(float));
            count++;
        }
    }
    printf("loaded data count is %d\n", count);


    // for(int i=0; i<keyList.size(); i+=slice_size){//here may need some check process
    //     printf("%f\n", *(keyMap.find(keyList[i])->second->getDataPtr()));
    //     printf("%f\n", *(model_storage[i/slice_size]));
    //     int batch = slice_size<=keyList.size()-i?slice_size:keyList.size()-i;
    //     net_training_f32(network, keyMap.find(keyList[i])->second->getDataPtr(), keyMap.find(keyList[i])->second->getLabelPtr(), model_storage[i/slice_size], model_storage[i/slice_size+1], batch);
    // }

}

void ecall_unlearning(uint64_t kid){
    // if(keyMap.find(kid) != keyMap.end()){
    //     Key* temp = keyMap.find(kid)->second;
    //     int index = std::find(keyList.begin(), keyList.end(), kid)-keyList.begin();

    //     temp->setTag(0);
    //     // filter.Delete(kid);
        
    //     int batch = slice_size<=keyList.size()-index?slice_size-1:keyList.size()-(keyList.size()/slice_size)*slice_size-1;

    //     //here may have bug with the last slice
    //     float* tempSlice = (float*)malloc(batch*c*sizeof(float));
    //     float* tempLabel = (float*)malloc(batch*sizeof(float));
        
    //     int firstHalf = index - (index / slice_size) * slice_size;
    //     int secondHalf = (index / slice_size + 1) * slice_size - index -1;
    //     float* dataPtr = temp->getDataPtr();
    //     float* labelPtr = temp->getLabelPtr();
    //     memcpy(tempSlice, dataPtr-firstHalf, firstHalf*c*sizeof(float));
    //     memcpy(tempSlice+firstHalf*c, dataPtr+1, secondHalf*c*sizeof(float));
    //     memcpy(tempLabel, labelPtr-firstHalf, firstHalf*sizeof(float));
    //     memcpy(tempLabel+firstHalf, labelPtr+1, secondHalf*sizeof(float));

    //     net_training_f32(network, tempSlice, tempLabel, model_storage[index/slice_size], model_storage[index/slice_size+1], batch);

    //     for(int i=(index/slice_size+1)*slice_size; i<keyList.size(); i+=slice_size){//here may need some check process
    //         printf("%f\n", *(keyMap.find(keyList[i])->second->getDataPtr()));
    //         printf("%f\n", *(model_storage[i/slice_size]));
    //         int batch = slice_size<=keyList.size()-i?slice_size:keyList.size()-i;
    //         net_training_f32(network, keyMap.find(keyList[i])->second->getDataPtr(), keyMap.find(keyList[i])->second->getLabelPtr(), model_storage[i/slice_size], model_storage[i/slice_size+1], batch);
    //     }
    // }
}