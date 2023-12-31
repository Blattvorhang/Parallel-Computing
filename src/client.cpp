#include <iostream>
#include <ctime>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "common.h"
#include "net.hpp"

float floatData[DATANUM];

// Template for client
int clientConnect(const char* server_ip, const int server_port) {
    int clientSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (clientSocket == -1) {
        std::cerr << "Error creating socket" << std::endl;
        return -1;
    }

    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = inet_addr(server_ip);
    serverAddr.sin_port = htons(server_port);

    if (connect(clientSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) == -1) {
        std::cerr << "Error connecting to server" << std::endl;
        close(clientSocket);
        return -1;
    }

    const char* message = "Hello, Server!";
    ssize_t bytesSent = send(clientSocket, message, strlen(message), 0);
    if (bytesSent == -1) {
        std::cerr << "Error sending data" << std::endl;
    } else {
        std::cout << "Sent data to server" << std::endl;
    }

    timespec start, end;

    // <len> <data>
    int len;
    ssize_t bytesRead = safeRecv(clientSocket, &len, sizeof(len), 0);
    clock_gettime(CLOCK_MONOTONIC, &start);
    int ret = recvArray(clientSocket, floatData, len);
    if (ret == -1) {
        std::cerr << "Error receiving array" << std::endl;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    double time_consumed = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1e9;
    std::cout << "Receiving array time consumed: " << time_consumed << "s" << std::endl;

    close(clientSocket);

    return 0;
}
