//esp code

#include <ESP8266WiFi.h>
#include <WebSocketsServer.h>

const char* ssid = "Kuwaity";
const char* password = "Aa@54321";


WebSocketsServer webSocket(81);

void webSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length) {
  if (type == WStype_TEXT) {
    String message = String((char*)payload);
    Serial.println( message);

    if (message == "check") {
      webSocket.sendTXT(num, "okay");
    } else {
      for (uint8_t i = 0; i < webSocket.connectedClients(); i++) {
        if (i != num) {  // Skip the sender
          webSocket.sendTXT(i, message);
        }
      }
    }
  }
}

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
 Serial.println("\nConnected to WiFi!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());



  webSocket.begin();
  webSocket.onEvent(webSocketEvent);
}

void loop() {
  webSocket.loop();}