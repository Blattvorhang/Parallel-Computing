#pragma once
#include <iostream>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "common.h"

const double SEP_ALPHA = 0.6;              // client proportion of data
const int SEP = DATANUM * SEP_ALPHA;       // position of separation
const int CLIENT_DATANUM = SEP;            // number of data for client
const int SERVER_DATANUM = DATANUM - SEP;  // number of data for server

const int SORT_BLOCK_NUM = 8;
const int SORT_BLOCK_SIZE = SERVER_DATANUM / SORT_BLOCK_NUM;
const int SORT_SOCKET_NUM = SORT_BLOCK_NUM;

// alternatives: 1024, 2048, 4096, 8192, 16384, 32768, 65536
const int BUFFER_SIZE = 4096;

ssize_t safeSend(int socket, const void* buffer, size_t length, int flags);
ssize_t safeRecv(int socket, void* buffer, size_t length, int flags);

template <typename T>
int sendArray(int socket, const T data[], const int len);

template <typename T>
int recvArray(int socket, T data[]);

int serverConnect(const int server_port, const float data[], const int len);
int clientConnect(const char* server_ip, const int server_port);



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
 * @return Length of array if success, -1 if error
 */
template <typename T>
int recvArray(int socket, T data[]) {
    // <len> <data>
    int len;
    ssize_t bytesRead = safeRecv(socket, &len, sizeof(len), 0);
    if (bytesRead == -1)
        return -1;
    
    const int block_len = BUFFER_SIZE / sizeof(T);
    int recv_len = 0;
    int recv_size;
    while (recv_len < len) {
        recv_size = std::min(block_len, len - recv_len);
        ssize_t bytesRead = safeRecv(socket, data + recv_len, recv_size * sizeof(T), 0);
        if (bytesRead == -1)
            return -1;
        recv_len += bytesRead / sizeof(T);
    }
    return len;
}