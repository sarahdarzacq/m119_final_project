#include <Arduino.h>
#include <Arduino_LSM6DS3.h>
#include <Adafruit_SSD1306.h>
#include <Wire.h>
#include "MAX30105.h"
#include "heartRate.h"

// For Testing w/o MAX30102
#define SIMULATE_HR 1

// OLED Constants
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_ADDR 0x3D
const uint32_t DISPLAY_PERIOD_MS = 5000;

// Heart-Rate Constants
const uint8_t HR_AVG_SIZE = 10;
const uint8_t HR_MIN_READINGS = 5; 
const uint32_t HR_STALE_MS = 3000;
const uint8_t BPM_MIN = 30;
const uint8_t BPM_MAX = 220;

// Sensor Declaration
MAX30105 particleSensor;
Adafruit_SSD1306 oledDisplay(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1, 400000UL, 400000UL);

// Heart-Rate Averaging Values 
uint8_t hr_values[HR_AVG_SIZE];
uint8_t hr_index = 0;
uint8_t hr_count = 0; 
uint32_t last_reading_ms = 0;

/// @brief Add heart-rate reading to the circular buffer.
/// @param bpm Heart-rate in beats per minute.
void hr_add_reading(uint32_t bpm)
{
    if (bpm < BPM_MIN || bpm > BPM_MAX) return;

    hr_values[hr_index] = bpm;
    hr_index = (hr_index + 1) % HR_AVG_SIZE;
    if (hr_count < HR_AVG_SIZE) hr_count++;
    last_reading_ms = millis();
}

/// @brief Reset heart-rate circular buffer.
void hr_reset()
{
    hr_index = 0; 
    hr_count = 0;
}

/// @brief Calculate average heart-rate over last hr_count values.
/// @return Average heart-rate rounded to nearest integer. 
uint8_t hr_get_average()
{
    if (hr_count < HR_MIN_READINGS) return 0;
    if (millis() - last_reading_ms > HR_STALE_MS) return 0;

    uint16_t sum = 0;
    for (uint8_t i = 0; i < hr_count; i++)
    {
        sum += hr_values[i];
    }

    /* Round output to nearest whole number */
    return (sum + hr_count / 2) / hr_count; 
}

/// @brief Display heart rate on OLED.
/// @param bpm Heart rate in beats per minute.
void display_heart_rate(uint8_t bpm)
{
    oledDisplay.clearDisplay();
    oledDisplay.setTextColor(SSD1306_WHITE);
    
    oledDisplay.setTextSize(1);
    oledDisplay.setCursor(0, 0);
    oledDisplay.print("Heart rate");

    oledDisplay.setTextSize(3);
    oledDisplay.setCursor(0, 20);
    if (bpm) oledDisplay.print(bpm);
    else oledDisplay.print("--");

    oledDisplay.setTextSize(1);
    oledDisplay.setCursor(70, 34);
    oledDisplay.print("BPM");

    oledDisplay.display();

    Serial.print("Display BPM: ");
    Serial.println(bpm);
}

/// @brief Update readings from MAX30102 sensor.
void hr_update()
{
#if SIMULATE_HR
    // Fake beats: HR drifts between ~70 and ~150 over a minute, with jitter.
    static uint32_t next_beat_ms = 0;
    uint32_t now = millis();
    if ((int32_t)(now - next_beat_ms) >= 0)
    {
        float   t   = now / 1000.0f;
        uint8_t bpm = 110 + 40 * sin(t / 10.0f) + random(-5, 6);
        hr_add_reading(bpm);
        next_beat_ms = now + 60000UL / bpm;
    }

#else 
    // TODO: Incorporate MAX30102

#endif
}

/// @brief Update OLED display.
void display_update()
{
    static uint32_t last_draw_ms = 0;
    uint32_t now = millis();

    if (now - last_draw_ms < DISPLAY_PERIOD_MS) return;
    last_draw_ms = now;

    display_heart_rate(hr_get_average());
}

void imu_update()
{
    // TODO: Implement.
}

void ble_update()
{
    // TODO: Implement. 
}

void recovery_update()
{
    // TODO: Implement.
}

void initialize_serial_monitor(int baud_rate)
{
    Serial.begin(baud_rate);
    Serial.println("Initializing Serial Monitor...");
    while (!Serial) { delay(10); }
    Serial.println("Initialized Serial Monitor...");
}

bool initialize_imu()
{
    if (!IMU.begin())
    {
        Serial.println("Failed to start IMU!");
        return false;
    }

    Serial.println("Successfully started IMU!");
    return true;
}

bool initialize_hr()
{
#if SIMULATE_HR
    Serial.println("HR Simulated - skipping MAX30102 init.");
    return true;
#else 
    if (!particleSensor.begin(Wire, I2C_SPEED_FAST))
    {
        Serial.println("MAX30102 was not found.");
        return false;
    }

    particleSensor.setup(0x1F, 4, 2);
    particleSensor.setPulseAmplitudeRed(0x0A); // Red LED on Low Power

    Serial.println("MAX30102 Sensor Initialized!");
    return true;
#endif
}

bool initialize_oled_display()
{
    if (!oledDisplay.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR))
    {
        Serial.println("SSD1306 Allocation Failed.");
        return false;
    }

    display_heart_rate(0);

    return true;
}

void setup()
{
    initialize_serial_monitor(115200);
    
    if (!initialize_imu())          while (true) {};
    if (!initialize_hr())           while (true) {};
    if (!initialize_oled_display()) while (true) {};
}

void loop() {
    ble_update();
    imu_update();
    hr_update();
    recovery_update();
    display_update();
}