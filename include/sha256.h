#ifndef SHA_256_H
#define SHA_256_H

#include <openssl/sha.h>
#include <Enclave.h>

void sha256_string(char *string, int len, char outputBuffer[33])
{
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, string, len);
    SHA256_Final(hash, &sha256);
    int i = 0;
    for(i = 0; i < SHA256_DIGEST_LENGTH; i++)
    {
        outputBuffer[i] = hash[i];
    }
    outputBuffer[32] = 0;
}


#endif

