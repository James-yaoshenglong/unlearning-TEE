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
#include "purchase_arch.hpp"

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

    int getSliceNum(){
        return slice;
    }

    int getTag(){
        return tag;
    }

    int getSeed(){
        return seed;
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
CuckooFilter<uint64_t, 8> filter(65536);

int network[] = {1, 128, 1};
int slice_size = 10000;
int model_num;
std::vector<Model*> model_storage;
vector<int> slice_start_index;
int real_count = 0;
vector<int> slice_state;

int r;
int c;
uint64_t eid;

// float* enclave_data_storage;
// float* enclave_label_storage;

MLP* mlp;


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

void hashModel(Model* model, Key* key){
    char* buffer = (char*)malloc(model->model_size+sizeof(uint32_t));
    memcpy(buffer, model->storage, model->model_size);
    uint32_t seed = key->getSeed();
    memcpy(buffer+model->model_size, &seed, sizeof(uint32_t));
    sha256_string(buffer, model->model_size+sizeof(uint32_t), model->hash);
}

int verifyModel(Model* model, Key* key){
    char temp[33];
    char* buffer = (char*)malloc(model->model_size+sizeof(uint32_t));
    memcpy(buffer, model->storage, model->model_size);
    uint32_t seed = key->getSeed();
    memcpy(buffer+model->model_size, &seed, sizeof(uint32_t));
    sha256_string(buffer, model->model_size+sizeof(uint32_t), temp);
    strcmp(model->hash, temp);
}

void test_filter(){
    CuckooFilter<uint64_t, 8> temp(65536);
    double start, end;
    ocall_get_time(&start);
    for(int i=0; i<r; i++){
        temp.Add(i);
    }
    ocall_get_time(&end);
    printf("Total insert time for %d is %.8f ms and each need %.8f ms\n", r, end-start, (end-start)/r);

    ocall_get_time(&start);
    for(int i=0; i<r; i++){
        temp.Contain(i);
    }
    ocall_get_time(&end);
    printf("Total query time for %d is %.8f ms and each need %.8f ms\n", r, end-start, (end-start)/r);

    ocall_get_time(&start);
    for(int i=0; i<r; i++){
        temp.Delete(i);
    }
    ocall_get_time(&end);
    printf("Total delete time for %d is %.8f ms and each need %.8f ms\n", r, end-start, (end-start)/r);
}

void ecall_init_enclave_storage(float* input_data, float* input_label, int row, int col, uint64_t enclave_id){
    r = row;
    c = col;
    eid = enclave_id;
    real_count = r;

    // define the nework parameter
    network[0] = col;
    model_num = (row % slice_size + row) / slice_size;
    
    //initialize the model storage
    for(int i=0; i<model_num+1; i++){
        Model* temp;
        ocall_init_model_storage((void**)&temp, network, 3);
        model_storage.push_back(temp);
    }

    // sgx_read_rand((unsigned char *)(model_storage[0]->storage), model_storage[0]->model_size);
    for(int i=0; i<model_storage[0]->model_size; i++){
        model_storage[0]->storage[i] = 0.01f;
    }

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
    // printf("fisrt kid is %ld\n",keyList[0]->getKid());
    printf("filter size is %d bytes\n", filter.SizeInBytes());
    test_filter();
}

void ecall_training(){
    //load whole data
    float* enclave_data_storage = (float*)malloc(r*c*sizeof(float));
    float* enclave_label_storage = (float*)malloc(r*sizeof(float));
    int count = 0;
    double start, end;
    ocall_get_time(&start);
    for(int i=0; i<keyList.size(); i+=1){
        Key* key = keyList[i];
        uint64_t hash = xxsha256(key, c, eid);
        if(filter.Contain(hash) == cuckoofilter::Ok){
            memcpy(enclave_data_storage+i*c, key->getDataPtr(), c*sizeof(float));
            memcpy(enclave_label_storage+i, key->getLabelPtr(), sizeof(float));
            count++;
        }
    }
    ocall_get_time(&end);
    printf("Total data load time for %d is %.8f ms and each need %.8f ms\n", r, end-start, (end-start)/r);
    printf("loaded data count is %d\n", count);

    mlp = new MLP(network, 0.01f, 1000);
    mlp->setModel(model_storage[0]);
    for(int i=0; i<keyList.size(); i+=slice_size){
        int size = keyList.size()<i+slice_size?keyList.size():i+slice_size;
        int current_slice_size = keyList.size()<i+slice_size?keyList.size()%slice_size:slice_size;
        slice_start_index.push_back(i);
        // printf("size is %d\n", size);
        slice_state.push_back(current_slice_size);
        // printf("current slice size is %d\n", slice_state.back());
        mlp->train(enclave_data_storage, enclave_label_storage, 22, size, model_storage[i/slice_size+1]);
        ocall_get_time(&start);
        mlp->saveModel(model_storage[i/slice_size+1]);
        hashModel(model_storage[i/slice_size+1], keyList[i]);
        ocall_get_time(&end);
        printf("Save time for model %d is %.8f ms\n", i/slice_size+1, end-start);
        // printf("%f\n", *(model_storage[0]->fc1w+1));
        // printf("%f\n", *(model_storage[1]->fc1w+1));
        printf("Save model %d\n", i/slice_size+1);
    }

    // mlp->forward(vector<float>(enclave_data_storage, enclave_data_storage+c));
    // vector<float> input(enclave_data_storage, enclave_data_storage+5000*c);
    // vector<float> result = mlp->inference(input);
    // int correct = 0;
    // for(int i=0; i<5000; i++){
    //     if(result[i] == enclave_label_storage[i]){
    //         correct++;
    //     }
    // }
    // printf("correct is %d\n", correct);
    
    // printf("label is %f %f\n", enclave_label_storage[0], result[0]);
    // for(int i=0; i<keyList.size(); i+=slice_size){//here may need some check process
    //     printf("%f\n", *(keyMap.find(keyList[i])->second->getDataPtr()));
    //     printf("%f\n", *(model_storage[i/slice_size]));
    //     int batch = slice_size<=keyList.size()-i?slice_size:keyList.size()-i;
    //     net_training_f32(network, keyMap.find(keyList[i])->second->getDataPtr(), keyMap.find(keyList[i])->second->getLabelPtr(), model_storage[i/slice_size], model_storage[i/slice_size+1], batch);
    // }
    
    //don't know why here free not work
    // free(enclave_data_storage);
    // free(enclave_label_storage);
}

void ecall_predict(float* data, float* label, int size){
    // mlp->setModel(model_storage[5]);
    int correct = 0;
    for(int i=0; i<size; i+=1000){
        int start = i;
        int end = i+1000<size?i+1000:size;
        int batch = end-start;
        vector<float> input(data+start*c, data+end*c);
        vector<float> result = mlp->inference(input);
        for(int j=0; j<batch; j++){
            if(result[j] == label[start+j]){
                correct++;
            }
        }
    }
    printf("correct is %d\n", correct);
    printf("accuracy is %f\n", ((double)correct)/size);
}

void ecall_unlearning(uint64_t kid){
    if(keyMap.find(kid) != keyMap.end()){
        Key* temp = keyMap.find(kid)->second;
        uint64_t hash = xxsha256(temp, c, eid);
        if(filter.Contain(hash) == cuckoofilter::Ok){
            filter.Delete(hash);
            temp->setTag(0);
            slice_state[temp->getSliceNum()]--;
            real_count--;

            //reload data
            float* data_storage = (float*)malloc(r*c*sizeof(float));
            float* label_storage = (float*)malloc(r*sizeof(float)); //don't know why exceed the length
            int count = 0;
            for(int i=0; i<keyList.size(); i+=1){
                Key* key = keyList[i];
                if(key->getKid() != temp->getKid() && key->getTag() != 0){
                    uint64_t hash = xxsha256(key, c, eid);
                    if(filter.Contain(hash) == cuckoofilter::Ok){
                        memcpy(data_storage+count*c, key->getDataPtr(), c*sizeof(float));
                        memcpy(label_storage+count, key->getLabelPtr(), sizeof(float));
                        count++;
                    }
                }
            }
            printf("loaded data count is %d\n", count);
            // printf("label storage is %f\n", *(enclave_label_storage+count-1));

            //unlearning
            int startSlice = temp->getSliceNum();
            // printf("start slice is %d\n", startSlice);
            double start, end;
            ocall_get_time(&start);
            if(startSlice>0){
                if(verifyModel(model_storage[startSlice], keyList[slice_start_index[startSlice-1]]) == 0){
                    printf("verifyed\n");
                }
            }
            mlp->setModel(model_storage[startSlice]);
            ocall_get_time(&end);
            printf("Model load time for %d is %.8f ms\n", startSlice, end-start);
            int size = 0;
            for(int i=0; i<slice_state.size(); i++){
                size+=slice_state[i];
                if(i>=startSlice){
                    mlp->train(data_storage, label_storage, 22, size, model_storage[i+1]);
                    mlp->saveModel(model_storage[i+1]);
                    hashModel(model_storage[i+1], keyList[slice_start_index[i]]);
                    printf("Save model %d\n", i+1);
                }
            }
            
        }

    }


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