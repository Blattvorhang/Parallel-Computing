#include <iostream>
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
    } else if (bytesRead == 0) {
        std::cout << "Server closed connection" << std::endl;
        return -1;
    }
    return bytesRead;
}
