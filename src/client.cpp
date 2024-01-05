#include <iostream>
#include <ctime>
#include <thread>
#include "common.h"
#include "speedup.h"
#include "net.hpp"

static int clientSocket;
static int sumSocket, maxSocket;
static int sortSocket;


float clientSum(const float data[], const int len) {
    const float* client_data = data;
    const int client_len = len * SEP_ALPHA;

    // Receive the sum from server asynchronously
    float server_sum;
    std::thread recvThread(safeRecv, sumSocket, &server_sum, sizeof(server_sum), 0);
    
    // Calculate the sum
    float client_sum = sumSpeedUp(client_data, client_len);

    // Wait for the sum from server
    recvThread.join();

    return client_sum + server_sum;
}


float clientMax(const float data[], const int len) {
    const float* client_data = data;
    const int client_len = len * SEP_ALPHA;

    // Receive the max from server asynchronously
    float server_max;
    std::thread recvThread(safeRecv, maxSocket, &server_max, sizeof(server_max), 0);

    // Calculate the max
    float client_max = maxSpeedUp(client_data, client_len);

    // Wait for the max from server
    recvThread.join();

    return client_max > server_max ? client_max : server_max;
}


void clientSort(const float data[], const int len, float result[]) {
    const float* client_data = data;
    const int client_len = len * SEP_ALPHA;

    float* client_result = new float[client_len];
    float* server_result = new float[len - client_len];

    // Receive the sorted array from server asynchronously
    std::thread recvThread(recvArray<float>, sortSocket, server_result);

    // Sort the array
    sortSpeedUp(client_data, client_len, client_result);

    // Wait for the sorted array from server
    recvThread.join();

    //recvArray(clientSocket, server_result);

    merge(result, client_result, server_result, client_len, len - client_len);

    delete[] client_result;
    delete[] server_result;
}


/**
 * @brief Handshake for synchronization, client sends 's' to server,
 *        server sends 'a' back to client.
 * @return 0 if success, -1 if error
 */
int clientSync() {
    const char sync = 's';
    ssize_t bytesSent = safeSend(clientSocket, &sync, sizeof(sync), 0);
    if (bytesSent == -1) {
        std::cerr << "Error sending begin" << std::endl;
        return -1;
    }

    std::cout << "Waiting for server to sync..." << std::endl;

    char sync_ack;
    ssize_t bytesRecv = safeRecv(clientSocket, &sync_ack, sizeof(sync_ack), 0);
    if (bytesRecv == -1 || sync_ack != 'a') {
        std::cerr << "Error receiving sync ack" << std::endl;
        return -1;
    }

    std::cout << "Server synchronized" << std::endl << std::endl;

    return 0;
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

    //setSocketTimeout(clientSocket, 100);
    return clientSocket;
}


int clientConnect(const char* server_ip, const int server_port) {
    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = inet_addr(server_ip);
    serverAddr.sin_port = htons(server_port);

    clientSocket = connectToServer(serverAddr);
    if (clientSocket == -1)
        return -1;
    std::cout << "Client socket connected" << std::endl;

    sumSocket = connectToServer(serverAddr);
    if (sumSocket == -1)
        return -1;
    std::cout << "Sum socket connected" << std::endl;

    maxSocket = connectToServer(serverAddr);
    if (maxSocket == -1)
        return -1;
    std::cout << "Max socket connected" << std::endl;

    sortSocket = connectToServer(serverAddr);
    if (sortSocket == -1)
        return -1;
    std::cout << "Sort sockets connected" << std::endl;

    std::cout << "Connected to server" << std::endl << std::endl;

    return 0;
}


void clientDisconnect() {
    close(sumSocket);
    close(maxSocket);
    close(sortSocket);
    close(clientSocket);

    std::cout << "Disconnected from server" << std::endl;
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
