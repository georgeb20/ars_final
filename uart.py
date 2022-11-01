from periphery import Serial

def resistance2array(resistance):
    string_res = str(resistance)
    first_digit = string_res[0]
    second_digit = string_res[1]
    num_zeros = str(len(string_res[2:]))
    return first_digit+second_digit+num_zeros

serial = Serial("/dev/ttymxc2", 9600)


resistance_array = resistance2array(101)

serial.write(bytes(resistance_array,'utf-8'))
# Open /dev/ttyUSB0 with baudrate 115200, and defaults of 8N1, no flow control


serial.close()