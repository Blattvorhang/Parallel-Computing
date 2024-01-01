#include <iostream>
#include <ctime>
#include <thread>
#include "common.h"
#include "speedup.h"
#include "net.hpp"

static int sumSocket, maxSocket;
static int sortSockets[SORT_SOCKET_NUM];


float clientSum(const float data[], const int len) {
    const float* client_data = data;
    const int client_len = len * SEP_ALPHA;

    float client_sum = sumSpeedUp(client_data, client_len);
    //std::cout << "Client sum: " << client_sum << std::endl;

    float server_sum;
    ssize_t bytesRecv = safeRecv(sumSocket, &server_sum, sizeof(server_sum), 0);
    if (bytesRecv == -1) {
        std::cerr << "Error receiving sum" << std::endl;
        return -1;
    }
    //std::cout << "Server sum: " << server_sum << std::endl;

    return client_sum + server_sum;
}


float clientMax(const float data[], const int len) {
    const float* client_data = data;
    const int client_len = len * SEP_ALPHA;

    float client_max = maxSpeedUp(client_data, client_len);
    //std::cout << "Client max: " << client_max << std::endl;

    float server_max;
    ssize_t bytesRecv = safeRecv(maxSocket, &server_max, sizeof(server_max), 0);
    if (bytesRecv == -1) {
        std::cerr << "Error receiving max" << std::endl;
        return -1;
    }
    //std::cout << "Server max: " << server_max << std::endl;

    return client_max > server_max ? client_max : server_max;
}


void clientSort(const float data[], const int len, float result[]) {
    //TODO
    const float* client_data = data;
    const int client_len = len * SEP_ALPHA;
}


int connectToServer(sockaddr_in serverAddr) {
    int clientSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (clientSocket == -1) {
        std::cerr << "Error creating socket" << std::endl;
        return -1;
    }
    
    if (connect(clientSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) == -1) {
        std::cerr << "Error connecting to server" << std::endl;
        close(clientSocket);
        return -1;
    }

    return clientSocket;
}


int clientConnect(const char* server_ip, const int server_port) {
    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = inet_addr(server_ip);
    serverAddr.sin_port = htons(server_port);

    sumSocket = connectToServer(serverAddr);
    if (sumSocket == -1)
        return -1;
    std::cout << "Sum socket connected" << std::endl;

    maxSocket = connectToServer(serverAddr);
    if (maxSocket == -1)
        return -1;
    std::cout << "Max socket connected" << std::endl;

    for (int i = 0; i < SORT_SOCKET_NUM; i++) {
        sortSockets[i] = connectToServer(serverAddr);
        if (sortSockets[i] == -1)
            return -1;
    }
    std::cout << "Sort sockets connected" << std::endl;

    std::cout << "Connected to server" << std::endl << std::endl;

    return 0;
}


void clientCloseSockets() {
    close(sumSocket);
    close(maxSocket);
    for (int i = 0; i < SORT_SOCKET_NUM; i++)
        close(sortSockets[i]);
}


int clientConnectTest(const char* server_ip, const int server_port) {
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
    float* floatData = new float[DATANUM];

    clock_gettime(CLOCK_MONOTONIC, &start);
    int len = recvArray(clientSocket, floatData);
    if (len == -1) {
        std::cerr << "Error receiving array" << std::endl;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    double time_consumed = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1e9;
    std::cout << "Receiving array time consumed: " << time_consumed << "s" << std::endl;

    close(clientSocket);
    delete[] floatData;

    return 0;
}
