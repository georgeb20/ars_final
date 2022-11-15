from statistics import mode
resistors = [100,120,270,470,1000,1500,2200,2700,3900,5600,8200,10000,11000,15000,22000,27000,47000,68000,100000,110000,222000,390000,680000,1000000,4700000,5600000,10000000]

attempts=0
computed_resistance = []
while(attempts<10):
    resistance = input("Enter resistance ")
    if(int(resistance) in resistors):
        computed_resistance.append(resistance)
    attempts+=1
final_resistance = mode(computed_resistance)
print(final_resistance)