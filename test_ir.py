import RPi.GPIO as GPIO
import time

# กำหนดพิน IR sensor
IR_SENSOR_PIN = 22  # ใช้พินเดียวกับในโค้ดหลัก
RED_LIGHT_PIN = 16
GREEN_LIGHT_PIN = 18

def setup():
    # ทำความสะอาด GPIO ก่อนเริ่มต้น
    try:
        GPIO.cleanup()
    except:
        pass
    
    # ตั้งค่า GPIO mode
    GPIO.setmode(GPIO.BOARD)
    
    # ตั้งค่า GPIO pins
    GPIO.setup(IR_SENSOR_PIN, GPIO.IN)
    GPIO.setup(RED_LIGHT_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(GREEN_LIGHT_PIN, GPIO.OUT, initial=GPIO.LOW)
    
    print("GPIO setup completed")

def cleanup():
    print("Cleaning up GPIO...")
    GPIO.cleanup()
    print("Cleanup completed")

def test_ir_sensor():
    setup()
    
    try:
        print("กำลังทดสอบ IR Sensor...")
        print("กด Ctrl+C เพื่อออกจากโปรแกรม")
        print("-------------------------------")
        
        # เริ่มสถานะด้วยไฟแดง
        GPIO.output(RED_LIGHT_PIN, GPIO.HIGH)
        GPIO.output(GREEN_LIGHT_PIN, GPIO.LOW)
        
        previous_state = None
        
        while True:
            # อ่านค่าจาก IR sensor
            current_state = GPIO.input(IR_SENSOR_PIN)
            
            # แสดงผลเมื่อมีการเปลี่ยนแปลงสถานะ
            if current_state != previous_state:
                if current_state == GPIO.HIGH:
                    print("✅ ตรวจพบวัตถุ (HIGH) - เปิดไฟเขียว")
                    GPIO.output(RED_LIGHT_PIN, GPIO.LOW)
                    GPIO.output(GREEN_LIGHT_PIN, GPIO.HIGH)
                else:
                    print("❌ ไม่พบวัตถุ (LOW) - เปิดไฟแดง")
                    GPIO.output(RED_LIGHT_PIN, GPIO.HIGH)
                    GPIO.output(GREEN_LIGHT_PIN, GPIO.LOW)
                
                previous_state = current_state
            
            # แสดงสถานะปัจจุบันทุก 1 วินาที
            if int(time.time()) % 1 == 0:
                state_text = "มีวัตถุ (HIGH)" if current_state == GPIO.HIGH else "ไม่มีวัตถุ (LOW)"
                print(f"สถานะ IR Sensor: {state_text}", end="\r")
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nหยุดการทดสอบด้วย Ctrl+C")
    
    finally:
        cleanup()

def test_ir_sensor_with_debug():
    setup()
    
    try:
        print("กำลังทดสอบ IR Sensor ในโหมด Debug...")
        print("กด Ctrl+C เพื่อออกจากโปรแกรม")
        print("-------------------------------")
        
        while True:
            # อ่านค่าจาก IR sensor
            current_state = GPIO.input(IR_SENSOR_PIN)
            
            # แสดงค่าสถานะทุกครั้ง
            state_text = "มีวัตถุ (HIGH)" if current_state == GPIO.HIGH else "ไม่มีวัตถุ (LOW)"
            print(f"สถานะ IR Sensor: {state_text}")
            
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\nหยุดการทดสอบด้วย Ctrl+C")
    
    finally:
        cleanup()

def test_light_only():
    setup()
    
    try:
        print("กำลังทดสอบไฟ LED...")
        print("กด Ctrl+C เพื่อออกจากโปรแกรม")
        print("-------------------------------")
        
        print("เปิดไฟแดง")
        GPIO.output(RED_LIGHT_PIN, GPIO.HIGH)
        GPIO.output(GREEN_LIGHT_PIN, GPIO.LOW)
        time.sleep(2)
        
        print("เปิดไฟเขียว")
        GPIO.output(RED_LIGHT_PIN, GPIO.LOW)
        GPIO.output(GREEN_LIGHT_PIN, GPIO.HIGH)
        time.sleep(2)
        
        print("ปิดไฟทั้งหมด")
        GPIO.output(RED_LIGHT_PIN, GPIO.LOW)
        GPIO.output(GREEN_LIGHT_PIN, GPIO.LOW)
        time.sleep(1)
        
        print("ทดสอบกระพริบไฟแดง")
        for i in range(5):
            GPIO.output(RED_LIGHT_PIN, GPIO.HIGH)
            time.sleep(0.5)
            GPIO.output(RED_LIGHT_PIN, GPIO.LOW)
            time.sleep(0.5)
        
        print("ทดสอบกระพริบไฟเขียว")
        for i in range(5):
            GPIO.output(GREEN_LIGHT_PIN, GPIO.HIGH)
            time.sleep(0.5)
            GPIO.output(GREEN_LIGHT_PIN, GPIO.LOW)
            time.sleep(0.5)
        
        print("การทดสอบไฟเสร็จสิ้น")
    
    except KeyboardInterrupt:
        print("\nหยุดการทดสอบด้วย Ctrl+C")
    
    finally:
        cleanup()

if __name__ == "__main__":
    print("เลือกโหมดการทดสอบ:")
    print("1. ทดสอบ IR Sensor พร้อมไฟแสดงผล")
    print("2. ทดสอบ IR Sensor แบบแสดงค่าทุกครั้ง (Debug)")
    print("3. ทดสอบไฟ LED เท่านั้น")
    
    while True:
        try:
            choice = int(input("เลือกโหมด (1-3): "))
            if 1 <= choice <= 3:
                break
            else:
                print("กรุณาเลือกตัวเลือก 1-3 เท่านั้น")
        except ValueError:
            print("กรุณาป้อนตัวเลข")
    
    if choice == 1:
        test_ir_sensor()
    elif choice == 2:
        test_ir_sensor_with_debug()
    elif choice == 3:
        test_light_only()