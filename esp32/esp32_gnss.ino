#include <TinyGPSPlus.h>

// ---------------- OBJECTS ----------------
TinyGPSPlus gps1;
TinyGPSPlus gps2;

HardwareSerial GNSS1(1);
HardwareSerial GNSS2(2);

// ---------------- PINS ----------------
#define RX1 18
#define TX1 5
#define RX2 26
#define TX2 25

// ---------------- SETUP ----------------
void setup()
{
  Serial.begin(115200);

  GNSS1.begin(9600, SERIAL_8N1, RX1, TX1);
  GNSS2.begin(9600, SERIAL_8N1, RX2, TX2);

  Serial.println("🚀 ESP32 Dual GNSS System Started");
}

// ---------------- TIME FORMAT ----------------
String formatTime(TinyGPSPlus &gps)
{
  if (!gps.time.isValid()) return "00:00:00";

  char buf[10];
  sprintf(buf, "%02d:%02d:%02d",
          gps.time.hour(),
          gps.time.minute(),
          gps.time.second());

  return String(buf);
}

// ---------------- MAIN LOOP ----------------
void loop()
{
  // Read GNSS streams
  while (GNSS1.available())
    gps1.encode(GNSS1.read());

  while (GNSS2.available())
    gps2.encode(GNSS2.read());

  // Trigger only when new data arrives
  if (gps1.location.isUpdated() || gps2.location.isUpdated())
  {
    // -------- GPS (Receiver 1) --------
    String gps_lat = gps1.location.isValid() ? String(gps1.location.lat(), 6) : "NO_FIX";
    String gps_lon = gps1.location.isValid() ? String(gps1.location.lng(), 6) : "NO_FIX";

    int gps_sat = gps1.satellites.isValid() ? gps1.satellites.value() : 0;
    float gps_hdop = gps1.hdop.isValid() ? gps1.hdop.hdop() : 99.99;

    // -------- GNSS (Receiver 2) --------
    String gnss_lat = gps2.location.isValid() ? String(gps2.location.lat(), 6) : "NO_FIX";
    String gnss_lon = gps2.location.isValid() ? String(gps2.location.lng(), 6) : "NO_FIX";

    int gnss_sat = gps2.satellites.isValid() ? gps2.satellites.value() : 0;
    float gnss_hdop = gps2.hdop.isValid() ? gps2.hdop.hdop() : 99.99;

    // -------- SERIAL OUTPUT FORMAT --------
    // Format:
    // GPS,time,lat,lon,sat,hdop | GNSS,time,lat,lon,sat,hdop

    Serial.print("GPS,");
    Serial.print(formatTime(gps1)); Serial.print(",");
    Serial.print(gps_lat); Serial.print(",");
    Serial.print(gps_lon); Serial.print(",");
    Serial.print(gps_sat); Serial.print(",");
    Serial.print(gps_hdop);

    Serial.print("|");

    Serial.print("GNSS,");
    Serial.print(formatTime(gps2)); Serial.print(",");
    Serial.print(gnss_lat); Serial.print(",");
    Serial.print(gnss_lon); Serial.print(",");
    Serial.print(gnss_sat); Serial.print(",");
    Serial.print(gnss_hdop);

    Serial.println();
  }

  // Small delay → prevents buffer overload + keeps ~1 Hz behavior
  delay(200);
}