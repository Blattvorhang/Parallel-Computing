#pragma once
#include <iostream>

// alternatives: 1024, 2048, 4096, 8192, 16384, 32768, 65536
const int BUFFER_SIZE = 4096;

ssize_t safeSend(int socket, const void* buffer, size_t length, int flags);
ssize_t safeRecv(int socket, void* buffer, size_t length, int flags);

template <typename T>
int sendArray(int socket, const T data[], const int len);

template <typename T>
int recvArray(int socket, T data[], const int len);

int serverConnect(const int server_port, const float data[], const int len);
int clientConnect(const char* server_ip, const int server_port);



template <typename T>
int sendArray(int socket, const T data[], const int len) {
    const int block_len = BUFFER_SIZE / sizeof(T);
    int send_len = 0;
    int send_size;
    while (send_len < len) {
        send_size = std::min(block_len, len - send_len) * sizeof(T);
        ssize_t bytesSent = safeSend(socket, data + send_len, send_size, 0);
        if (bytesSent == -1)
            return -1;
        // std::cout << "Sent " << bytesSent << " bytes" << std::endl;
        send_len += bytesSent / sizeof(T);
    }
    return 0;
}


template <typename T>
int recvArray(int socket, T data[], const int len) {
    const int block_len = BUFFER_SIZE / sizeof(T);
    int recv_len = 0;
    int recv_size;
    while (recv_len < len) {
        recv_size = std::min(block_len, len - recv_len) * sizeof(T);
        ssize_t bytesRead = safeRecv(socket, data + recv_len, recv_size, 0);
        if (bytesRead == -1)
            return -1;
        // std::cout << "Received " << bytesRead << " bytes" << std::endl;
        recv_len += bytesRead / sizeof(T);
    }
    return 0;
}