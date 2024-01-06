#include <iostream>
#include "speedup.h"
#include "common.h"
#include "net.hpp"

static int serverSocket, clientSocket;
static int sumSocket, maxSocket;
static int sortSocket;
//static int sortSockets[SORT_SOCKET_NUM];


float serverSum(const float data[], const int len) {
    const float* server_data = data + int(len * SEP_ALPHA);
    const int server_len = len - int(len * SEP_ALPHA);

    float server_sum = sumSpeedUp(server_data, server_len);
    ssize_t bytesSent = safeSend(sumSocket, &server_sum, sizeof(server_sum), 0);
    //ssize_t bytesSent = safeSend(clientSocket, &server_sum, sizeof(server_sum), 0);
    if (bytesSent == -1) {
        std::cerr << "Error sending sum" << std::endl;
        return -1;
    }

    return server_sum;
}


float serverMax(const float data[], const int len) {
    const float* server_data = data + int(len * SEP_ALPHA);
    const int server_len = len - int(len * SEP_ALPHA);

    float server_max = maxSpeedUp(server_data, server_len);
    ssize_t bytesSent = safeSend(maxSocket, &server_max, sizeof(server_max), 0);
    //ssize_t bytesSent = safeSend(clientSocket, &server_max, sizeof(server_max), 0);
    if (bytesSent == -1) {
        std::cerr << "Error sending max" << std::endl;
        return -1;
    }

    return server_max;
}


void serverSort(const float data[], const int len, float result[]) {
    const float* server_data = data + int(len * SEP_ALPHA);
    const int server_len = len - int(len * SEP_ALPHA);
    //const int block_len = server_len / SORT_BLOCK_NUM;

    float* server_result = new float[server_len];
    sortSpeedUp(server_data, server_len, server_result);

    // TODO: asynchronously send data in blocks
    //int ret = safeSendArray(sortSockets[0], server_result, server_len, SORT_BLOCK_NUM);
    int ret = sendArray(sortSocket, server_result, server_len);
    //int ret = sendArray(clientSocket, server_result, server_len);
    if (ret == -1) {
        std::cerr << "Error sending array" << std::endl;
    }
    delete[] server_result;
}


/**
 * @brief Handshake for synchronization, client sends 's' to server,
 *        server sends 'a' back to client.
 * @return 0 if success, -1 if error
 */
int serverSync() {
    std::cout << "Waiting for client to synchronize..." << std::endl;

    char sync;
    ssize_t bytesRecv = safeRecv(clientSocket, &sync, sizeof(sync), 0);
    if (bytesRecv == -1 || sync != 's') {
        std::cerr << "Error receiving sync" << std::endl;
        return -1;
    }

    const char sync_ack = 'a';
    ssize_t bytesSent = safeSend(clientSocket, &sync_ack, sizeof(sync_ack), 0);
    if (bytesSent == -1) {
        std::cerr << "Error sending sync ack" << std::endl;
        return -1;
    }
    
    std::cout << "Client synchronized" << std::endl << std::endl;

    return 0;
}


int acceptClientConnection(int serverSocket) {
    sockaddr_in clientAddr;
    socklen_t clientAddrLen = sizeof(clientAddr);
    int clientSocket = accept(serverSocket, (struct sockaddr*)&clientAddr, &clientAddrLen);
    if (clientSocket == -1) {
        std::cerr << "Error accepting connection" << std::endl;
        close(serverSocket);
        return -1;
    }

    return clientSocket;
}


int serverConnect(const int server_port, const float data[], const int len) {
    serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket == -1) {
        std::cerr << "Error creating socket" << std::endl;
        return -1;
    }

    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(server_port);

    if (bind(serverSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) == -1) {
        std::cerr << "Error binding socket" << std::endl;
        close(serverSocket);
        return -1;
    }

    if (listen(serverSocket, 10) == -1) {
        std::cerr << "Error listening on socket" << std::endl;
        close(serverSocket);
        return -1;
    }

    std::cout << "Server listening on port " << server_port << "..." << std::endl;

    clientSocket = acceptClientConnection(serverSocket);
    if (clientSocket == -1)
        return -1;
    std::cout << "Client socket connected" << std::endl;

    sumSocket = acceptClientConnection(serverSocket);
    if (sumSocket == -1)
        return -1;
    std::cout << "Sum socket connected" << std::endl;

    maxSocket = acceptClientConnection(serverSocket);
    if (maxSocket == -1)
        return -1;
    std::cout << "Max socket connected" << std::endl;

    sortSocket = acceptClientConnection(serverSocket);
    if (sortSocket == -1)
        return -1;
    std::cout << "Sort socket connected" << std::endl;

    std::cout << "Client connected" << std::endl << std::endl;

    return 0;
}


void serverDisconnect() {
    close(sumSocket);
    close(maxSocket);
    close(sortSocket);
    close(clientSocket);
    close(serverSocket);

    std::cout << "Disconnected from client" << std::endl;
}


int serverConnectTest(const int server_port, const float data[], const int len) {
    int serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket == -1) {
        std::cerr << "Error creating socket" << std::endl;
        return -1;
    }

    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(server_port);

    if (bind(serverSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) == -1) {
        std::cerr << "Error binding socket" << std::endl;
        close(serverSocket);
        return -1;
    }

    if (listen(serverSocket, 10) == -1) {
        std::cerr << "Error listening on socket" << std::endl;
        close(serverSocket);
        return -1;
    }

    std::cout << "Server listening on port " << server_port << "..." << std::endl;

    sockaddr_in clientAddr;
    socklen_t clientAddrLen = sizeof(clientAddr);
    int clientSocket = accept(serverSocket, (struct sockaddr*)&clientAddr, &clientAddrLen);
    if (clientSocket == -1) {
        std::cerr << "Error accepting connection" << std::endl;
        close(serverSocket);
        return -1;
    }

    std::cout << "Client connected" << std::endl;

    char buffer[BUFFER_SIZE];
    ssize_t bytesRead = recv(clientSocket, buffer, sizeof(buffer), 0);
    if (bytesRead == -1) {
        std::cerr << "Error receiving data" << std::endl;
    } else {
        buffer[bytesRead] = '\0';
        std::cout << "Received data from client: " << buffer << std::endl;
    }

    std::cout << "Sending data to client..." << std::endl;
    int ret = sendArray(clientSocket, data, len);
    if (ret == -1) {
        std::cerr << "Error sending array" << std::endl;
    }

    close(clientSocket);
    close(serverSocket);

    return 0;
}