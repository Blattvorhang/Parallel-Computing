#pragma once

const int BUFFER_SIZE = 65536;  // alternatives: 1024, 2048, 4096, 8192, 16384, 32768, 65536

int serverConnect(const int server_port, const float data[], const int len);