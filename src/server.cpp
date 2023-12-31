#include <iostream>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include "server.h"

// Template for server
int serverConnect(const int server_port, const float data[], const int len) {
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
    // <len> <data> -1(EOF)
    // ssize_t bytesSent = send(clientSocket, &len, sizeof(len), 0);
    // int send_len = 0;
    // const int array_size = BUFFER_SIZE / sizeof(float);
    // for (int i = 0; i < len; i += array_size) {
    //     send_len = (i + array_size < len) ? array_size : len - i;
    //     ssize_t bytesSent = send(clientSocket, data + i, send_len * sizeof(float), 0);
    //     if (bytesSent == -1) {
    //         std::cerr << "Error sending data" << std::endl;
    //     } else {
    //         std::cout << "Sent data to client" << std::endl;
    //     }
    // }
    // send_len = -1;
    // send(clientSocket, &send_len, sizeof(send_len), 0);
    for (int i = 0; i < BUFFER_SIZE / sizeof(float); i++) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
    ssize_t bytesSent = send(clientSocket, data, BUFFER_SIZE, 0);
    

    close(clientSocket);
    close(serverSocket);

    return 0;
}
