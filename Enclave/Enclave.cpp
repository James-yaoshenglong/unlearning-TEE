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

#include <sgx_trts.h>

#include "Enclave.h"
#include "Enclave_t.h"  /* print_string */

#include "xxhash64.h"
#include "cuckoofilter.h"

using cuckoofilter::CuckooFilter;

class Key{
public:
    Key(){}

    Key(double* ptr, int col){
        dataPtr = ptr;
        len = col*sizeof(double);
        // kid = hash(dataPtr);
        kid = XXHash64::hash(ptr, len, 1); //here need set seed
        tag = 1;
        // seed = rand(); //here has problem, need set random seed
        sgx_read_rand((unsigned char *)&seed, 4);
    }

    uint64_t getKid(){
        return kid;
    }

    double* getDataPtr(){
        return dataPtr;
    }
private:
    uint64_t kid;
    uint64_t len;
    uint32_t tag;
    uint32_t seed;
    double* dataPtr;
};

std::map<uint64_t, Key*> keyMap; //here do not know whether it is in the enclave and use enclave std lib
CuckooFilter<uint64_t, 12> filter(10000);


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

void ecall_init_enclave_storage(double** input, int row, int col){
    for(int i=0; i < row; i++){
        Key* key = new Key(input[i], i);
        // printf("%f\n", input[i][0]);
        // printf("%f\n", key->getDataPtr()[0]);
        keyMap[key->getKid()] = key;
        filter.Add(key->getKid());
        printf("%d", filter.Contain(key->getKid() == cuckoofilter::Ok));
    }
    // printf("%f\n", input[1][0]);
    // Key* test = keyMap.find(1)->second;
    // printf("%f\n",test->getDataPtr()[0]);
    // printf("%d\n", keyMap.find(1)->first);
}
