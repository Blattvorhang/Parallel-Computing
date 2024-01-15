#include <iostream>
#include "common.h"
#include "net.hpp"

ssize_t safeSend(int socket, const void* buffer, size_t length, int flags) {
    ssize_t bytesSent = send(socket, buffer, length, flags);
    if (bytesSent == -1) {
        std::cerr << "Error sending data" << std::endl;
        return -1;
    }
    return bytesSent;
}

ssize_t safeRecv(int socket, void* buffer, size_t length, int flags) {
    ssize_t bytesRead = recv(socket, buffer, length, flags);
    if (bytesRead == -1) {
        std::cerr << "Error receiving data" << std::endl;
        return -1;
        // if (errno == EAGAIN || errno == EWOULDBLOCK) {
        //     std::cerr << "Receive timeout" << std::endl;
        //     return 0;
        // } else {
        //     std::cerr << "Error receiving data" << std::endl;
        //     return -1;
        // }
    } else if (bytesRead == 0) {
        std::cout << "Server closed connection" << std::endl;
        return -1;
    }
    return bytesRead;
}


/**
 * @brief Set socket timeout
 * @param socket Socket to set timeout
 * @param timeout Timeout in milliseconds
 * @return 0 if success, -1 if error
 */
int setSocketTimeout(int socket, int timeout) {
    struct timeval tv;
    tv.tv_sec = timeout / 1000;
    tv.tv_usec = (timeout % 1000) * 1000;

    if (setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) == -1) {
        std::cerr << "Error setting socket timeout" << std::endl;
        return -1;
    }
    return 0;
}


// int safeSendArray(int socket, const float data[], const int len, const int block_num) {
//     ssize_t bytesSent = safeSend(socket, &len, sizeof(len), 0);
//     if (bytesSent == -1)
//         return -1;
    
//     const int block_len = len / block_num;
//     int block_begin = 0;
//     while (block_begin < len) {
//         int block_size = std::min(block_len, len - block_begin);

//         int ret = sendArray(socket, data + block_begin, block_size);
//         if (ret == -1) {
//             std::cerr << "Error sending array" << std::endl;
//             return -1;
//         }

//         // Wait for ack
//         char ack;
//         ssize_t bytesRecv = safeRecv(socket, &ack, sizeof(ack), 0);
//         if (bytesRecv == -1) {
//             std::cerr << "Error receiving ack" << std::endl;
//             return -1;
//         }
        
//         if (ack == 'n') {
//             std::cerr << "Partial data received, retransmitting block..." << std::endl;
//             continue;
//         }
//         else if (ack != 'y') {
//             std::cerr << "Error receiving ack" << std::endl;
//             return -1;
//         }
//         block_begin += block_size;
//     }

//     return 0;
// }


// int safeRecvArray(int socket, float data[], const int block_num) {
//     int len;
//     ssize_t bytesRead = safeRecv(socket, &len, sizeof(len), 0);
//     if (bytesRead == -1)
//         return -1;
    
//     const int block_len = len / block_num;
//     int block_begin = 0;
//     while (block_begin < len) {
//         int block_size = std::min(block_len, len - block_begin);

//         int ret = recvArray(socket, data + block_begin);
//         if (ret == -1) {
//             std::cerr << "Error receiving array" << std::endl;
//             return -1;
//         }
//         else if (ret == 0) {
//             std::cerr << "Timeout receiving array" << std::endl;
//             std::cerr << "Partial data received, request block retransmission..." << std::endl;
//             char ack = 'n';
//             ssize_t bytesSent = safeSend(socket, &ack, sizeof(ack), 0);
//             if (bytesSent == -1) {
//                 std::cerr << "Error sending ack" << std::endl;
//                 return -1;
//             }
//             continue;
//         }

//         // Send ack
//         char ack = 'y';
//         ssize_t bytesSent = safeSend(socket, &ack, sizeof(ack), 0);
//         if (bytesSent == -1) {
//             std::cerr << "Error sending ack" << std::endl;
//             return -1;
//         }
//         block_begin += block_size;
//     }

//     return 0;
// }