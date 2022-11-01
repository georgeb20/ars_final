from periphery import Serial

# Open /dev/ttyUSB0 with baudrate 115200, and defaults of 8N1, no flow control
uart1 = Serial("/dev/ttymxc0", 115200)

serial.write(b"Hello World!")

# Read up to 128 bytes with 500ms timeout
buf = serial.read(128, 0.5)
print("read {:d} bytes: _{:s}_".format(len(buf), buf))

serial.close()