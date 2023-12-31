#include <iostream>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "server.h"
#include "common.h"

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

    float buffer[BUFFER_SIZE];
    int recv_len = 0;
    // Receive len first, if len == -1, then EOF, otherwise, receive data
    // while (true) {
    //     ssize_t bytesRead = recv(clientSocket, &recv_len, sizeof(recv_len), 0);
    //     if (bytesRead == -1) {
    //         std::cerr << "Error receiving data" << std::endl;
    //         break;
    //     } else if (bytesRead == 0) {
    //         std::cout << "Server closed connection" << std::endl;
    //         break;
    //     } else {
    //         std::cout << "Received data from server: " << recv_len << std::endl;
    //     }

    //     if (recv_len == -1) {
    //         std::cout << "Received EOF from server" << std::endl;
    //         break;
    //     }

    //     std::cout << "Receiving data from server..." << std::endl;
    //     // <len> <data> -1(EOF)
    //     for (int i = 0; i < recv_len; i += BUFFER_SIZE) {
    //         ssize_t bytesRead = recv(clientSocket, buffer, sizeof(buffer), 0);
    //         if (bytesRead == -1) {
    //             std::cerr << "Error receiving data" << std::endl;
    //             break;
    //         } else if (bytesRead == 0) {
    //             std::cout << "Server closed connection" << std::endl;
    //             break;
    //         } else {
    //             std::cout << "Received data from server: " << bytesRead << std::endl;
    //         }

    //         for (int j = 0; j < bytesRead; j++) {
    //             std::cout << buffer[j] << std::endl;
    //         }
    //     }
    // }
    ssize_t bytesRead = recv(clientSocket, floatData, BUFFER_SIZE, 0);
    for (int i = 0; i < BUFFER_SIZE / sizeof(float); i++) {
        std::cout << floatData[i] << " ";
    }
    std::cout << std::endl;


    close(clientSocket);

    return 0;
}
