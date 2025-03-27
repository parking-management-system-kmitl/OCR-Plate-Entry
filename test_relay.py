import RPi.GPIO as GPIO
import time

in1 = 16
in2 = 18

GPIO.setmode(GPIO.BOARD)
GPIO.setup(in1, GPIO.OUT)
GPIO.setup(in2, GPIO.OUT)

GPIO.output(in1, False)
GPIO.output(in2, False)

try:
    while True:
        # เปิด in1 ปิด in2
        GPIO.output(in1, True)
        GPIO.output(in2, False)
        print("LED 1: เปิด, LED 2: ปิด")
        time.sleep(3)
        
        # ปิด in1 เปิด in2
        GPIO.output(in1, False) 
        GPIO.output(in2, True)
        print("LED 1: ปิด, LED 2: เปิด")
        time.sleep(3)

except KeyboardInterrupt:
    print("\nโปรแกรมหยุดทำงาน")
    GPIO.cleanup()