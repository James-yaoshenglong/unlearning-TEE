# unlearning-TEE
The sgx implementation of paper: [Proof of Unlearning: Definitions and Instantiation]()

## Prerequisite
1. Install SGX SDK and Driver

    Follow the instructions of [linux-sgx](https://github.com/intel/linux-sgx) to build and install SGX SDK and Driver

2. Install SGX SSL Library

    Follow the instructions of [intel-sgx-ssl](https://github.com/intel/intel-sgx-ssl#linux) to build and install SGX SSL Library

3. Install SGX DNNL Library

    Follow the instructions of [sgx-dnnl](https://github.com/intel/linux-sgx/tree/master/external/dnnl) to build and install SGX DNNL Library

## Test
1. Data Preparation

    ```
    cd datasets/purchase && python3 prepare_data.py
    ```

2. Data Shard

    ```
    cd python && python3 distribution.py
    ```

3. Build

    ```
    make SGX_MODE=HW
    ```

4. Running Test

    ```
    python3 python/test.py
    ```

## Implementation Detail
1. Data structure implementation and basic data/memory operation is in [Enclave/Enclave.cpp](https://github.com/James-yaoshenglong/unlearning-TEE/blob/master/Enclave/Enclave.cpp)

2. Neural Network structure implementation for Purchase Dataset is in [Enclave/purchase_arch.cpp](https://github.com/James-yaoshenglong/unlearning-TEE/blob/master/Enclave/purchase_arch.cpp) 

3. Outside operation API implementation is in [APP/App.cpp](https://github.com/James-yaoshenglong/unlearning-TEE/blob/master/App/App.cpp)