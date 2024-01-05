#pragma once
#include <iostream>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "common.h"

// #define SORT_BLOCK_NUM 8
// #define SORT_SOCKET_NUM SORT_BLOCK_NUM

const double SEP_ALPHA = 0.5; // client proportion of data

// alternatives: 1024, 2048, 4096, 8192, 16384, 32768, 65536
const int BUFFER_SIZE = 1024;

ssize_t safeSend(int socket, const void* buffer, size_t length, int flags);
ssize_t safeRecv(int socket, void* buffer, size_t length, int flags);

// int safeSendArray(int socket, const float data[], const int len, const int block_num);
// int safeRecvArray(int socket, float data[], const int block_num);

template <typename T>
int sendArray(int socket, const T data[], const int len);

template <typename T>
int recvArray(int socket, T data[]);

int setSocketTimeout(int socket, int timeout);

int clientConnect(const char* server_ip, const int server_port);
int serverConnect(const int server_port, const float data[], const int len);

int clientSync();
int serverSync();

void clientDisconnect();
void serverDisconnect();

float clientSum(const float data[], const int len);
float clientMax(const float data[], const int len);
void clientSort(const float data[], const int len, float result[]);

float serverSum(const float data[], const int len);
float serverMax(const float data[], const int len);
void serverSort(const float data[], const int len, float result[]);


/**
 * @brief Send an array of data. Format: <len> <data>
 * @param socket Socket to send data
 * @param data Array of data
 * @param len Length of array
 * @return 0 if success, -1 if error
 */
template <typename T>
int sendArray(int socket, const T data[], const int len) {
    // <len> <data>
    ssize_t bytesSent = safeSend(socket, &len, sizeof(len), 0);
    if (bytesSent == -1)
        return -1;
    
    const int block_len = BUFFER_SIZE / sizeof(T);
    int send_len = 0;
    int send_size;
    while (send_len < len) {
        send_size = std::min(block_len, len - send_len);
        ssize_t bytesSent = safeSend(socket, data + send_len, send_size * sizeof(T), 0);
        if (bytesSent == -1)
            return -1;
        send_len += bytesSent / sizeof(T);
    }
    return 0;
}


/**
 * @brief Receive an array of data. Format: <len> <data>
 * @param socket Socket to receive data
 * @param data Array of data
 * @return Length of array if success, -1 if error, 0 if timeout
 */
template <typename T>
int recvArray(int socket, T data[]) {
    // <len> <data>
    int len;
    ssize_t bytesRead = safeRecv(socket, &len, sizeof(len), 0);
    if (bytesRead == -1)
        return -1;
    if (len <= 0) {
        std::cerr << "Error receiving array length" << std::endl;
        return -1;
    }
    
    const int block_len = BUFFER_SIZE / sizeof(T);
    int recv_len = 0;
    int recv_size;
    while (recv_len < len) {
        recv_size = std::min(block_len, len - recv_len);
        ssize_t bytesRead = safeRecv(socket, data + recv_len, recv_size * sizeof(T), 0);
        if (bytesRead == -1)
            return -1;
        else if (bytesRead == 0)
            return 0;

        recv_len += bytesRead / sizeof(T);
    }
    return len;
}