#ifndef DATA_STRUCTURE_HPP
#define DATA_STRUCTURE_HPP


class Model{
public:
    int model_size;
    float* storage;
    float* fc1w;
    float* fc1b;
    float* fc2w;
    float* fc2b;
    char hash[33];
    Model(int* network, int len){
        model_size = 0;
        for(int i=0; i< len-1; i++){
            model_size += (network[i]+1)*network[i+1]*sizeof(float);
        }
        storage = (float*)malloc(model_size);
        fc1w = storage;
        fc1b = fc1w+network[0]*network[1];
        fc2w = fc1b+network[1];
        fc2b = fc2w+network[1]*network[2];
    }
};



#endif