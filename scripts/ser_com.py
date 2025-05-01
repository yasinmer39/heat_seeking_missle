import serial
import time

ser = serial.Serial('/dev/ttyAMA10', 115200, timeout=1)

while True:
    # Write data to STM32
    ser.write(b'ping from RPi\n')
    
    # Read 10 bytes from STM32
    rx_buffer = ser.read(10)
    if rx_buffer:
        print(f"Received: {rx_buffer.decode('utf-8', errors='ignore')}")
    
    # Delay for 1 second
    time.sleep(1)