from periphery import Serial

# Open /dev/ttyUSB0 with baudrate 115200, and defaults of 8N1, no flow control
serial = Serial("/dev/ttymxc2", 9600)

serial.write(b"Hello World!")

serial.close()