#include <Arduino.h>
#include <Arduino_LSM6DS3.h>
#include <Adafruit_SSD1306.h>
#include <ArduinoBLE.h>
#include <Wire.h>
#include "MAX30105.h"
#include "heartRate.h"

// For Testing w/o MAX30102
#define SIMULATE_HR 1

// BLE Device
#define BLE_DEVICE_NAME "HRMon"
#define UUID_SERVICE "8e3f0001-6c1a-4b7e-9d52-1f0a3c5e7b90"
#define UUID_IMU "8e3f0002-6c1a-4b7e-9d52-1f0a3c5e7b90"
#define UUID_COMMAND "8e3f0003-6c1a-4b7e-9d52-1f0a3c5e7b90"

// Command bytes from laptop
const uint8_t CMD_REST_DETECTED = 0x01;
const uint8_t CMD_CANCEL = 0x02; 

BLEService sensorService(UUID_SERVICE);
BLECharacteristic imuCharacteristic(UUID_IMU, BLENotify, sizeof(ImuPacket), true);
BLECharacteristic commandCharacteristic(UUID_COMMAND, BLEWrite | BLEWriteWithoutResponse, 4);

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

// IMU Constants 
const uint16_t IMU_RATE_HZ = 50;
const uint32_t IMU_PERIOD_US = 1000000UL / IMU_RATE_HZ;
const uint8_t IMU_BATCH_SIZE = 10;

// IMU Packet to send to laptop
struct __attribute__((packed)) ImuPacket
{
  uint16_t seq; 
  int16_t samples[IMU_BATCH_SIZE][6]; // ax, ay, az [mg], gx, gy, gz
};

// IMU Data Collection Values
ImuPacket imu_packet;
uint8_t imu_sample_count = 0; 
uint16_t imu_seq = 0;
uint32_t imu_last_us = 0; 

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

/// @brief Send IMU packet over bluetooth to connected device.
/// @param packet Struct holding IMU data.
void imu_send_batch(const ImuPacket& packet)
{
    if (!ble_imu_subscribed()) return;
    imuCharacteristic.writeValue((const uint8_t*)&packet, sizeof(packet));

    Serial.print("IMU batch seq="); Serial.print(packet.seq);
    Serial.print(" bytes="); Serial.println(sizeof(packet));
}

/// @brief 
void imu_read_sample(float& ax, float& ay, float& az, float& gx, float& gy, float& gz)
{
    IMU.readAcceleration(ax, ay, az);
    IMU.readGyroscope(gx, gy, gz);
}

bool ble_imu_subscribed()
{
    return BLE.connected() && imuCharacteristic.subscribed();
}

void on_ble_connected(BLEDevice central)
{
    Serial.print("BLE connected: ");
    Serial.println(central.address());
}

void on_ble_disconnected(BLEDevice central)
{
    (void)central; 
    Serial.print("BLE disconnected, advertising again");
    imu_sample_count = 0;
    BLE.advertise();
}

void on_command_written(BLEDevice central, BLECharacteristic characteristic)
{
    (void)central;
    int len = characteristic.valueLength();
    if (len < 1) return;

    const uint8_t* data = characteristic.value();
    switch (data[0])
    {
        case CMD_REST_DETECTED:
            recovery_start(len >= 2 ? data[1] : 0);
            break;
        case CMD_CANCEL:
            recovery_cancel();
            break;
        default:
            Serial.print("Unknown command: ");
            Serial.println(data[0]);
            break;
    }
}

enum RecoveryState : uint8_t
{
    RECOVERY_IDLE,
    RECOVERY_MEASURING,
    RECOVERY_DONE,
    RECOVERY_INVALID
};

const uint32_t RECOVERY_WINDOW_MS = 60000;
const uint8_t HR_HISTORY_S = 15;

RecoveryState recovery_state = RECOVERY_IDLE;
uint8_t recovery_hr_start = 0;
uint8_t recovery_hr_end = 0;
uint32_t recovery_start_ms = 0; 

uint8_t hr_history[HR_HISTORY_S] = {0};
uint8_t hr_history_index = 0;
uint32_t hr_history_last_ms = 0;

uint8_t hr_peak_recent(uint8_t lookback_s)
{
    uint8_t peak = hr_get_average();
    if (lookback_s > HR_HISTORY_S) lookback_s = HR_HISTORY_S;

    for (uint8_t i = 1; i <= lookback_s; i++)
    {
        uint8_t v = hr_history[(hr_history_index + HR_HISTORY_S - i) % HR_HISTORY_S];
        if (v > peak) peak = v;
    }
}

int16_t recovery_get_hrr()
{
    if (recovery_state != RECOVERY_DONE) return 0;
    return (int16_t)recovery_hr_start - (int16_t)recovery_hr_end;
}

uint8_t recovery_seconds_left()
{
    if (recovery_state != RECOVERY_MEASURING) return 0; 
    uint32_t elapsed = millis() - recovery_start_ms;
    if (elapsed >= RECOVERY_WINDOW_MS) return 0;
    return (RECOVERY_WINDOW_MS - elapsed + 999) / 1000; 
}

// Recovery "interrupt" functions
void recovery_start(uint8_t lookback_s)
{
    (void)lookback_s;
    if (recovery_state == RECOVERY_MEASURING)
    {
        Serial.println("Recovery already running - ignoring");
        return;
    }

    recovery_hr_start = hr_peak_recent(lookback_s);
    recovery_hr_end = 0;

    if (recovery_hr_start == 0)
    {
        recovery_state = RECOVERY_INVALID;
        Serial.println("Recovery start failed: no valid HR");
        return;
    }

    recovery_start_ms = millis();
    recovery_state = RECOVERY_MEASURING;
    Serial.print("Recovery started, HR start = ");
    Serial.println(recovery_hr_start);
}

void recovery_cancel()
{
    recovery_state = RECOVERY_IDLE;
    recovery_hr_start = 0;
    recovery_hr_end = 0;
    Serial.println("Recovery cancelled");
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
    uint32_t now = micros(); 
    if (now - imu_last_us < IMU_PERIOD_US) return;

    imu_last_us += IMU_PERIOD_US;
    if (now - imu_last_us > 4 * IMU_PERIOD_US)
        imu_last_us = now;

    if (!ble_imu_subscribed()) { imu_sample_count = 0; return; }

    float ax, ay, az, gx, gy, gz;
    imu_read_sample(ax, ay, az, gx, gy, gz);

    imu_packet.samples[imu_sample_count][0] = (int16_t)(ax * 1000.0f);
    imu_packet.samples[imu_sample_count][1] = (int16_t)(ay * 1000.0f);
    imu_packet.samples[imu_sample_count][2] = (int16_t)(ax * 1000.0f);
    imu_packet.samples[imu_sample_count][3] = (int16_t)(ax * 10.0f);
    imu_packet.samples[imu_sample_count][4] = (int16_t)(ax * 10.0f);
    imu_packet.samples[imu_sample_count][5] = (int16_t)(ax * 10.0f);

    if (++imu_sample_count == IMU_BATCH_SIZE)
    {
        imu_packet.seq = imu_seq++;
        imu_send_batch(imu_packet);
        imu_sample_count = 0;
    }
}

void ble_update()
{
    BLE.poll();
}

void recovery_update()
{
    uint32_t now = millis();

    if (now - hr_history_last_ms >= 1000)
    {
        hr_history_last_ms += 1000;
        if (now - hr_history_last_ms >= 1000) hr_history_last_ms = now;

        hr_history[hr_history_index] = hr_get_average();
        hr_history_index = (hr_history_index + 1) % HR_HISTORY_S;
    }

    if (recovery_state != RECOVERY_MEASURING) return;
    if (now - recovery_start_ms < RECOVERY_WINDOW_MS) return;

    recovery_hr_end = hr_get_average();
    recovery_state = recovery_hr_end ? RECOVERY_DONE : RECOVERY_INVALID;

    Serial.print("Recovery finished: ");
    Serial.print(recovery_hr_start);
    Serial.print(" -> ");
    Serial.print(recovery_hr_end);
    Serial.print(", HRR = ");
    Serial.println(recovery_get_hrr());
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

bool initialize_ble() 
{
    if (!BLE.begin())
    {
        Serial.println("Failed to start BLE!");
        return false;
    }

    BLE.setLocalName(BLE_DEVICE_NAME);
    BLE.setDeviceName(BLE_DEVICE_NAME);
    BLE.setAdvertisedService(sensorService);

    sensorService.addCharacteristic(imuCharacteristic);
    sensorService.addCharacteristic(commandCharacteristic);
    BLE.addService(sensorService);

    BLE.setEventHandler(BLEConnected, on_ble_connected);
    BLE.setEventHandler(BLEDisconnected, on_ble_disconnected);
    commandCharacteristic.setEventHandler(BLEWritten, on_command_written);

    BLE.advertise();
    Serial.println("BLE advertising...");
    return true;
}

void setup()
{
    initialize_serial_monitor(115200);
    
    if (!initialize_imu()) while (true) {};
    if (!initialize_hr()) while (true) {};
    if (!initialize_oled_display()) while (true) {};
    if (!initialize_ble()) while (true) {};
}

void loop() {
    ble_update();
    imu_update();
    hr_update();
    recovery_update();
    display_update();
}
