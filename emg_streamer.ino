/*
 * emg_streamer.ino
 *
 * Streams two analog channels (EMG on A0, audio on A1) over Serial
 * at a fixed sample rate. Format: "emg_value,audio_value\n"
 *
 * Hardware:
 *   A0 -- EMG sensor board SIG output
 *   A1 -- microphone module output (or leave floating for EMG-only mode)
 *   GND -- common ground for everything
 *
 * Notes:
 *   - On Arduino Uno: ADC is 10-bit (0-1023), max stable rate ~1 kHz for 2 channels.
 *   - On Teensy 4.0: ADC is 12-bit (0-4095), can easily hit 4 kHz for 2 channels.
 *     If using Teensy, change ADC_MAX accordingly.
 *   - Sample rate is enforced via micros() timing, not delay(), to keep it precise.
 */

const unsigned long SAMPLE_RATE_HZ = 1000;          // 1 kHz per channel
const unsigned long SAMPLE_INTERVAL_US = 1000000UL / SAMPLE_RATE_HZ;
const int EMG_PIN = A0;
const int AUDIO_PIN = A1;

unsigned long next_sample_time = 0;

void setup() {
  Serial.begin(500000);   // high baud rate so we don't bottleneck on transmission
  // analogReadResolution(12);  // <-- uncomment this line if using Teensy
  next_sample_time = micros();
}

void loop() {
  unsigned long now = micros();
  if ((long)(now - next_sample_time) >= 0) {
    int emg = analogRead(EMG_PIN);
    int audio = analogRead(AUDIO_PIN);

    // CSV line: emg,audio\n
    Serial.print(emg);
    Serial.print(',');
    Serial.println(audio);

    next_sample_time += SAMPLE_INTERVAL_US;
  }
}
