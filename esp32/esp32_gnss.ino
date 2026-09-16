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

// ---------------- GSV SNR TRACKER ----------------
// TinyGPSPlus does not expose per-satellite C/N0, so the raw NMEA GSV
// sentences ($GPGSV/$GLGSV/$GAGSV/$GBGSV) are parsed separately to derive an
// average signal quality (dB-Hz) per receiver over a rolling 1 s window.
struct SnrTracker {
  char buf[128];
  int len = 0;
  float sum = 0;
  int count = 0;
  unsigned long windowStart = 0;

  void resetWindow(unsigned long now) {
    if (now - windowStart >= 1000) {
      sum = 0;
      count = 0;
      windowStart = now;
    }
  }

  void feed(byte c) {
    if (c == '\n' || c == '\r') {
      if (len > 0) {
        buf[len] = 0;
        handleLine();
        len = 0;
      }
      return;
    }
    if (len < 127) buf[len++] = c;
  }

  float average() const { return count ? sum / count : 0.0f; }

 private:
  void handleLine() {
    // Only GSV sentences carry per-satellite signal strength.
    for (int i = 1; i + 3 < len; i++)
      if (buf[i] == 'G' && buf[i + 1] == 'S' && buf[i + 2] == 'V' && buf[i + 3] == ',')
        { parseGsv(); return; }
  }

  void parseGsv() {
    // $xxGSV,nsent,sentn,total,inUse, prn,el,az,snr[, prn,el,az,snr ...] *cs
    int i = 0, commas = 0;
    while (i < len && commas < 4) if (buf[i++] == ',') commas++;
    if (i >= len) return;
    i++;  // begin at the first prn field
    while (i < len && buf[i] != '*') {
      for (int k = 0; k < 4 && i < len && buf[i] != '*'; k++) {
        char tmp[8];
        int n = 0;
        while (i < len && buf[i] != ',' && buf[i] != '*') tmp[n++] = buf[i++];
        tmp[n] = 0;
        if (k == 3) {
          int v = atoi(tmp);
          if (v > 0 && v < 100) { sum += v; count++; }
        }
        if (i < len && buf[i] == ',') i++;
      }
    }
  }
};

SnrTracker snr1;
SnrTracker snr2;

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
  unsigned long now = millis();
  snr1.resetWindow(now);
  snr2.resetWindow(now);

  // Read GNSS streams (feed both TinyGPSPlus and the SNR trackers)
  while (GNSS1.available()) { byte c = GNSS1.read(); gps1.encode(c); snr1.feed(c); }
  while (GNSS2.available()) { byte c = GNSS2.read(); gps2.encode(c); snr2.feed(c); }

  // Trigger only when new data arrives
  if (gps1.location.isUpdated() || gps2.location.isUpdated())
  {
    // -------- GPS (Receiver 1) --------
    String gps_lat = gps1.location.isValid() ? String(gps1.location.lat(), 6) : "NO_FIX";
    String gps_lon = gps1.location.isValid() ? String(gps1.location.lng(), 6) : "NO_FIX";

    int gps_sat = gps1.satellites.isValid() ? gps1.satellites.value() : 0;
    float gps_hdop = gps1.hdop.isValid() ? gps1.hdop.hdop() : 99.99;
    float gps_snr = snr1.average();

    // -------- GNSS (Receiver 2) --------
    String gnss_lat = gps2.location.isValid() ? String(gps2.location.lat(), 6) : "NO_FIX";
    String gnss_lon = gps2.location.isValid() ? String(gps2.location.lng(), 6) : "NO_FIX";

    int gnss_sat = gps2.satellites.isValid() ? gps2.satellites.value() : 0;
    float gnss_hdop = gps2.hdop.isValid() ? gps2.hdop.hdop() : 99.99;
    float gnss_snr = snr2.average();

    // -------- SERIAL OUTPUT FORMAT --------
    // Format:
    // GPS,time,lat,lon,sat,hdop,snr | GNSS,time,lat,lon,sat,hdop,snr
    // snr is the 1 s average C/N0 (dB-Hz) from GSV sentences; 0 until a GSV
    // sentence has been observed.

    Serial.print("GPS,");
    Serial.print(formatTime(gps1)); Serial.print(",");
    Serial.print(gps_lat); Serial.print(",");
    Serial.print(gps_lon); Serial.print(",");
    Serial.print(gps_sat); Serial.print(",");
    Serial.print(gps_hdop, 2); Serial.print(",");
    Serial.print(gps_snr, 0);

    Serial.print("|");

    Serial.print("GNSS,");
    Serial.print(formatTime(gps2)); Serial.print(",");
    Serial.print(gnss_lat); Serial.print(",");
    Serial.print(gnss_lon); Serial.print(",");
    Serial.print(gnss_sat); Serial.print(",");
    Serial.print(gnss_hdop, 2); Serial.print(",");
    Serial.print(gnss_snr, 0);

    Serial.println();
  }

  // Small delay → prevents buffer overload + keeps ~1 Hz behavior
  delay(200);
}