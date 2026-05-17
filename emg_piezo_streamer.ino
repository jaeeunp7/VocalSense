/*
 * emg_piezo_streamer.ino
 *
 * Streams two analog channels (EMG on A0, piezo on A1) over Serial
 * at a fixed sample rate. Format: "emg_value,piezo_value\n"
 *
 * Hardware:
 *   A0  -- EMG sensor board SIG output
 *   A1  -- piezo protection circuit output (after voltage divider/AC coupling)
 *   5V  -- shared power rail for EMG board (+VS) and piezo divider
 *   GND -- shared ground for EMG board (GND, -VS) and piezo (- lead)
 *
 * Notes:
 *   - On Arduino Uno: ADC is 10-bit (0-1023), stable at 1 kHz per channel.
 *   - On Teensy 4.0: ADC is 12-bit (0-4095) -- uncomment analogReadResolution(12)
 *     in setup() and change ADC_MAX in the Python files to 4095.
 *   - Timing uses micros() rather than delay() so the sample rate stays precise
 *     even as Serial.print latency varies between calls.
 */

const unsigned long SAMPLE_RATE_HZ = 1000;          // 1 kHz per channel
const unsigned long SAMPLE_INTERVAL_US = 1000000UL / SAMPLE_RATE_HZ;
const int EMG_PIN = A0;
const int PIEZO_PIN = A1;

unsigned long next_sample_time = 0;

void setup() {
  Serial.begin(500000);
  // analogReadResolution(12);  // <-- uncomment this line if using Teensy
  next_sample_time = micros();
}

void loop() {
  unsigned long now = micros();
  if ((long)(now - next_sample_time) >= 0) {
    int emg = analogRead(EMG_PIN);
    int piezo = analogRead(PIEZO_PIN);

    // CSV line: emg,piezo\n
    Serial.print(emg);
    Serial.print(',');
    Serial.println(piezo);

    next_sample_time += SAMPLE_INTERVAL_US;
  }
}
