import os
import cv2
import torch
import time
import queue
import threading
from datetime import datetime
from autoTransform.transform import process_auto_transform
from splitImage.split import process_split_image
from readLicense.read import process_read_license
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk
import RPi.GPIO as GPIO
import subprocess
import warnings



from PIL import Image, ImageTk


import requests
import io
import numpy as np
import datetime

warnings.filterwarnings("ignore")

# เพิ่มตัวแปรสำหรับเก็บเวลา OCR
ocr_times = []  # เก็บเวลาที่ใช้ในการทำ OCR
ocr_count = 0   # จำนวนครั้งที่ทำ OCR

# กำหนดค่าเริ่มต้น
YOLO_WIDTH = 640
YOLO_HEIGHT = 640
OCR_SIZE = 224
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
FPS = 60

FRAME_BUFFER_SIZE = 1  # ลดขนาด buffer ลงเพื่อลดความล่าช้า

# กำหนดค่า GPIO pins
BUTTON_PIN = 31
RED_LIGHT_PIN = 16
GREEN_LIGHT_PIN = 18
IR_SENSOR_PIN = 22

# กำหนดค่า Trigger Zone
ZONE_LEFT = 30
ZONE_RIGHT = 70
ZONE_TOP = 0
ZONE_BOTTOM = 100

stop_event = threading.Event()
last_successful_plate = None
current_frame = None  # เพิ่มตัวแปร global สำหรับเก็บ frame ปัจจุบัน
API_URL =  "http://jjsornwakii.3bbddns.com:34724/parking/entry-raspi"

# เพิ่มตัวแปรสำหรับเก็บข้อมูลเวลา
plate_detection_time = None  # เวลาเมื่อพบป้ายทะเบียน
ocr_start_time = None  # เวลาเริ่มต้นประมวลผล OCR
ocr_end_time = None  # เวลาสิ้นสุดประมวลผล OCR
button_press_time = None  # เวลากดปุ่ม
print_start_time = None  # เวลาเริ่มต้นพิมพ์
print_end_time = None  # เวลาสิ้นสุดการพิมพ์

# เพิ่มฟังก์ชันสำหรับแสดงผลเวลาในรูปแบบที่อ่านง่าย
def format_time_diff(start_time, end_time):
    """แสดงความแตกต่างของเวลาในรูปแบบที่อ่านง่าย (วินาที และมิลลิวินาที)"""
    if start_time is None or end_time is None:
        return "N/A"
    diff = (end_time - start_time) * 1000  # แปลงเป็นมิลลิวินาที
    return f"{diff:.2f} ms ({diff/1000:.3f} วินาที)"

def print_time_stats():
    """แสดงสถิติเวลาการทำงานของระบบทั้งหมด"""
    print("\n" + "="*50)
    print("📊 สถิติเวลาการทำงานของระบบ:")
    print("-"*50)
    
    if plate_detection_time and ocr_start_time:
        print(f"⏱️ เวลาตั้งแต่ตรวจพบป้ายถึงเริ่ม OCR: {format_time_diff(plate_detection_time, ocr_start_time)}")
    
    if ocr_start_time and ocr_end_time:
        print(f"⏱️ เวลาประมวลผล OCR: {format_time_diff(ocr_start_time, ocr_end_time)}")
    
    if plate_detection_time and ocr_end_time:
        print(f"⏱️ เวลาทั้งหมดตั้งแต่พบป้ายถึง OCR เสร็จสมบูรณ์: {format_time_diff(plate_detection_time, ocr_end_time)}")
    
    if button_press_time and print_start_time:
        print(f"⏱️ เวลาตั้งแต่กดปุ่มถึงเริ่มพิมพ์: {format_time_diff(button_press_time, print_start_time)}")
    
    if print_start_time and print_end_time:
        print(f"⏱️ เวลาการพิมพ์: {format_time_diff(print_start_time, print_end_time)}")
    
    if button_press_time and print_end_time:
        print(f"⏱️ เวลาทั้งหมดตั้งแต่กดปุ่มถึงพิมพ์เสร็จ: {format_time_diff(button_press_time, print_end_time)}")
    
    print("="*50 + "\n")

def setup_gpio():
    """ตั้งค่าเริ่มต้นสำหรับ GPIO ทั้งหมด"""
    # ทำความสะอาด GPIO ก่อนเริ่มต้น
    try:
        GPIO.cleanup()
    except:
        pass
    
    # ตั้งค่า GPIO mode
    GPIO.setmode(GPIO.BOARD)
    
    # ตั้งค่า GPIO pins
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(RED_LIGHT_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(GREEN_LIGHT_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(IR_SENSOR_PIN, GPIO.IN)
    
    print("🔌 GPIO setup completed")

def control_lights(red_state, green_state):
    """ควบคุมไฟ LED แบบ active-high
    
    Args:
        red_state (bool): True เพื่อเปิดไฟแดง, False เพื่อปิด
        green_state (bool): True เพื่อเปิดไฟเขียว, False เพื่อปิด
    """
    # Active-high - เปิดด้วย HIGH ปิดด้วย LOW
    GPIO.output(RED_LIGHT_PIN, GPIO.HIGH if red_state else GPIO.LOW)
    GPIO.output(GREEN_LIGHT_PIN, GPIO.HIGH if green_state else GPIO.LOW)

def turn_red():
    """เปิดไฟแดง ปิดไฟเขียว"""
    control_lights(True, False)

def turn_green():
    """เปิดไฟเขียว ปิดไฟแดง"""
    control_lights(False, True)

def run_printer_script(plate_number):
    global print_start_time, print_end_time
    
    print(f"🖨️ กำลังสั่งพิมพ์ป้ายทะเบียน: {plate_number}")
    print_start_time = time.time()
    subprocess.run(["python3", "printer/printer.py", plate_number])
    print_end_time = time.time()
    
    print(f"🖨️ การพิมพ์เสร็จสิ้น ใช้เวลา: {format_time_diff(print_start_time, print_end_time)}")
    
    # แสดงสถิติเวลาทั้งหมดหลังจากพิมพ์เสร็จ
    print_time_stats()

def button_listener(frame_queue):
    """ฟังก์ชันรอรับการกดปุ่มและจัดการการส่งข้อมูล"""
    global last_successful_plate, button_press_time
    
    while not stop_event.is_set():
        button_state = GPIO.input(BUTTON_PIN)
        if button_state == GPIO.LOW:  # เมื่อปุ่มถูกกด
            button_press_time = time.time()
            print(f"\n🔘 ปุ่มถูกกด! เวลา: {datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
            
            if last_successful_plate:
                try:
                    # ดึง frame ปัจจุบันจาก queue
                    if not frame_queue.empty():
                        current_frame = frame_queue.get()
                        
                        print(f"📤 กำลังส่งข้อมูลป้ายทะเบียน: {last_successful_plate}")
                        send_start_time = time.time()
                        
                        # ส่งข้อมูลไปยัง server
                        if send_data(API_URL, current_frame, last_successful_plate):
                            send_end_time = time.time()
                            print(f"✅ ส่งข้อมูลสำเร็จ ใช้เวลา: {format_time_diff(send_start_time, send_end_time)}")
                            run_printer_script(last_successful_plate)
                        else:
                            send_end_time = time.time()
                            print(f"❌ ไม่สามารถส่งข้อมูลได้ ใช้เวลาในการพยายาม: {format_time_diff(send_start_time, send_end_time)}")
                        
                        # ใส่ frame กลับเข้า queue
                        frame_queue.put(current_frame)
                    else:
                        print("❌ ไม่พบ frame ในคิว")
                        
                except Exception as e:
                    print(f"❌ เกิดข้อผิดพลาดในการส่งข้อมูล: {str(e)}")
                
                time.sleep(1)  # ป้องกันการกดซ้ำ
                
        time.sleep(0.1)  # ลดการใช้ CPU

def ir_light_controller():
    """ฟังก์ชันควบคุมไฟตามสถานะของ IR sensor และปุ่มกด"""
    last_state = None  # เก็บสถานะก่อนหน้าของ IR sensor
    waiting_for_ir = False  # สถานะรอ IR sensor
    green_light_active = False  # สถานะไฟเขียว
    
    # เริ่มต้นด้วยไฟแดง
    control_lights(True, False)
    print("🚦 เริ่มต้นระบบ - เปิดไฟแดง")
    
    while not stop_event.is_set():
        try:
            current_ir_state = GPIO.input(IR_SENSOR_PIN)
            button_state = GPIO.input(BUTTON_PIN)
            
            # เมื่อกดปุ่ม และไฟเขียวยังไม่ติด
            if button_state == GPIO.LOW and not green_light_active:
                print("🚦 ปุ่มถูกกด - เปิดไฟเขียว")
                control_lights(False, True)  # เปิดไฟเขียว
                green_light_active = True
                waiting_for_ir = True
                time.sleep(0.5)  # ป้องกันการกดซ้ำ
                continue
            
            # ถ้าอยู่ในโหมดรอ IR และมีวัตถุผ่าน
            if waiting_for_ir and current_ir_state != last_state:
                if current_ir_state == GPIO.HIGH:  # มีวัตถุผ่าน
                    print("🚗 IR Sensor: มีวัตถุผ่าน - รอ 2 วินาที")
                    time.sleep(2)  # รอ 2 วินาที
                    print("🚦 ครบ 2 วินาที - เปิดไฟแดง")
                    control_lights(True, False)  # เปิดไฟแดง
                    green_light_active = False
                    waiting_for_ir = False
                last_state = current_ir_state
            
            # ถ้าไม่ได้อยู่ในโหมดรอ IR ให้ไฟแดงติดตลอด
            elif not waiting_for_ir and not green_light_active:
                control_lights(True, False)  # เปิดไฟแดง
            
            time.sleep(0.1)
            
        except Exception as e:
            print(f"❌ IR controller error: {str(e)}")
            time.sleep(1)

def cleanup_gpio():
    """ทำความสะอาด GPIO เมื่อจบโปรแกรม"""
    try:
        GPIO.setmode(GPIO.BOARD)  # ต้องตั้ง mode ก่อน cleanup
        GPIO.cleanup()  # ทำความสะอาด GPIO ทั้งหมด
    except Exception as e:
        print(f"❌ GPIO cleanup error: {str(e)}")


# เพิ่มฟังก์ชันใหม่สำหรับจัดการเวลา
def get_formatted_time():
    """คืนค่าเวลาปัจจุบันในรูปแบบ HH:MM:SS"""
    return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]  # แสดงถึงมิลลิวินาที



################################# ส่งข้อมูลไป Server ####################

def send_data(api_url: str, frame: np.ndarray, license_plate: str):
    """ส่งข้อมูลไปยัง Server

    Args:
        api_url (str): URL ของ API
        frame (np.ndarray): ภาพที่จะส่ง
        license_plate (str): หมายเลขทะเบียนรถ

    Returns:
        bool: True ถ้าส่งข้อมูลสำเร็จ, False ถ้าไม่สำเร็จ
    """
    try:
        # ตั้งชื่อไฟล์ด้วยเวลาปัจจุบัน
        timestamp = datetime.datetime.now().strftime("%H%M%S")
        filename = f"{license_plate}_{timestamp}.jpg"

        # แปลงภาพเป็น JPEG buffer
        _, buffer = cv2.imencode(".jpg", frame)
        image_io = io.BytesIO(buffer)

        # เตรียมข้อมูลสำหรับส่ง
        data = {"licensePlate": license_plate}
        files = {"image": (filename, image_io, "image/jpeg")}

        # ส่งข้อมูลไปยัง API
        send_start = time.time()
        response = requests.post(api_url, data=data, files=files)
        send_time = format_time_diff(send_start, time.time())
        response_data = response.json()

        # ตรวจสอบ success จากการตอบกลับ
        if response_data.get('success') == True:
            print(f"✅ อัปโหลดสำเร็จ: {response_data}")
            print(f"⏱️ เวลาที่ใช้ในการส่งข้อมูล: {send_time}")
            # ดึงข้อมูลสำคัญจาก response
            car_id = response_data['data']['carId']
            parking_record_id = response_data['data']['parkingRecordId']
            payment_id = response_data['data']['paymentId']
            print(f"📋 รายละเอียด:")
            print(f"   - Car ID: {car_id}")
            print(f"   - Parking Record ID: {parking_record_id}")
            print(f"   - Payment ID: {payment_id}")
            return True
        else:
            print(f"❌ การส่งข้อมูลไม่สำเร็จ: {response_data}")
            print(f"⏱️ เวลาที่ใช้ในการส่งข้อมูล: {send_time}")
            return False

    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการส่งข้อมูล: {str(e)}")
        return False




def update_gui(frame, video_label):
    """อัพเดทภาพในหน้าต่าง GUI โดยใช้ double buffering"""
    try:
        # แปลงภาพเป็น RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        # สร้าง PhotoImage แบบ double buffering
        img_tk = ImageTk.PhotoImage(image=img)
        # เก็บ reference ไว้ป้องกัน garbage collection
        video_label.photo = img_tk
        # อัพเดทภาพ
        video_label.configure(image=img_tk)
    except Exception as e:
        print(f"GUI update error: {str(e)}")



def capture_frame(cap, frame_queue):
    # ตั้งค่า buffer size ของ OpenCV
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # ตั้งค่า FPS ของกล้อง
    cap.set(cv2.CAP_PROP_FPS, 1)  # ปรับเป็น 30 FPS
    
    print("📷 เริ่มต้นการจับภาพจากกล้อง")
    frame_count = 0
    start_time = time.time()
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("❌ ไม่สามารถอ่านภาพจากกล้องได้")
            stop_event.set()
            break
        
        # ล้าง frame เก่าออกจาก queue
        while not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                break
        
        frame_queue.put(frame)
        
        # แสดง FPS ทุกๆ 30 เฟรม
        frame_count += 1
        if frame_count % 30 == 0:
            current_time = time.time()
            elapsed = current_time - start_time
            fps = frame_count / elapsed
            print(f"📊 กล้อง FPS: {fps:.2f}")
            frame_count = 0
            start_time = current_time
        

def process_frame(frame_queue, model, last_ocr_time, trigger_zone, model_path, font_path, 
                 ocr_results_queue, video_label, plate_label, time_label, root):
    global last_successful_plate, plate_detection_time, ocr_start_time, ocr_end_time, ocr_times, ocr_count
  
    #global last_successful_plate, plate_detection_time, ocr_start_time, ocr_end_time

    # ตัวแปรสำหรับการติดตามสถานะ
    object_in_zone = False
    last_successful_plate = None
    waiting_for_exit = False
    trigger_start_time = 0
    last_ocr_time = time.time()  # เริ่มต้นเวลา OCR
    min_confidence_threshold = 0.6  # ค่าความเชื่อมั่นขั้นต่ำสำหรับ OCR
    last_gui_update = time.time()
    GUI_UPDATE_INTERVAL = 1/30  # 30 FPS
    
    # ตัวแปรสำหรับวัด FPS
    frame_count = 0
    fps_start_time = time.time()
    processing_times = []
    
    print("🔍 เริ่มต้นการประมวลผลภาพ")
    
    while not stop_event.is_set():
        try:
            process_start_time = time.time()
            
            original_frame = frame_queue.get(timeout=1)
            
            yolo_frame = cv2.resize(original_frame.copy(), (YOLO_WIDTH, YOLO_HEIGHT))
            
            height, width = original_frame.shape[:2]
            scale_x = width / YOLO_WIDTH
            scale_y = height / YOLO_HEIGHT
            
            actual_trigger_zone = (
                (int(trigger_zone[0][0] * scale_x), int(trigger_zone[0][1] * scale_y)),
                (int(trigger_zone[1][0] * scale_x), int(trigger_zone[1][1] * scale_y))
            )
            
            cv2.rectangle(original_frame, actual_trigger_zone[0], actual_trigger_zone[1], (255, 0, 0), 2)

            # เริ่มจับเวลา YOLO
            yolo_start_time = time.time()
            results = model.predict(yolo_frame, verbose=False)[0]
            yolo_time = time.time() - yolo_start_time
            
            # ตรวจสอบว่ามีป้ายทะเบียนอยู่ใน trigger zone หรือไม่
            plate_in_zone = False

            if len(results.boxes) > 0:
                for box in results.boxes:
                    coords = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf)
                    cls = int(box.cls)

                    if conf < model.conf:
                        continue
                    
                    xmin, ymin, xmax, ymax = coords
                    real_xmin = int(xmin * scale_x)
                    real_ymin = int(ymin * scale_y)
                    real_xmax = int(xmax * scale_x)
                    real_ymax = int(ymax * scale_y)

                    ocr_time_gap = 1

                    if cls in [0, 1]:  # car หรือ licenseplate
                        real_cx = (real_xmin + real_xmax) // 2
                        real_cy = (real_ymin + real_ymax) // 2

                        cv2.rectangle(original_frame, (real_xmin, real_ymin), (real_xmax, real_ymax), (0, 255, 0), 2)
                        label = f"{['car', 'licenseplate'][cls]}: {conf:.2%}"
                        cv2.putText(original_frame, label, (real_xmin, real_ymin - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * max(scale_x, scale_y), (0, 255, 0), 2)

                        # เช็คว่าป้ายทะเบียนอยู่ใน Trigger Zone หรือไม่ โดยดูทั้งกรอบ
                        in_trigger_zone = (
                            real_xmin >= actual_trigger_zone[0][0] and
                            real_xmax <= actual_trigger_zone[1][0] and
                            real_ymin >= actual_trigger_zone[0][1] and
                            real_ymax <= actual_trigger_zone[1][1]
                        )
                        
                        if cls == 1:  # licenseplate
                            if in_trigger_zone:
                                plate_in_zone = True
                                
                                if not waiting_for_exit:  # ถ้าไม่ได้รอให้ป้ายทะเบียนออกจาก zone
                                    if not object_in_zone:
                                        object_in_zone = True
                                        trigger_start_time = time.time()
                                        plate_detection_time = time.time()
                                        print(f"\n🔍 ตรวจพบป้ายทะเบียนใน Zone เวลา: {get_formatted_time()}")
                                    elif time.time() - trigger_start_time >= 1:
                                        current_time = time.time()
                                        if current_time - last_ocr_time >= ocr_time_gap:
                                            print(f"🔄 เริ่มกระบวนการ OCR เวลา: {get_formatted_time()}")
                                            ocr_start_time = time.time()
                                            plate_img = original_frame[real_ymin:real_ymax, real_xmin:real_xmax]
                                            plate_img = cv2.resize(plate_img, (OCR_SIZE, OCR_SIZE))

                                            try:
                                                # เริ่มจับเวลา OCR
                                                ocr_start_time = time.time()
                                                
                                                # preprocessing แบบเดียวกับไฟล์ที่ 1
                                                plate_img = cv2.resize(plate_img, (OCR_SIZE, OCR_SIZE))
                                                plate_img_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                                                plate_img_eq = cv2.equalizeHist(plate_img_gray)
                                                plate_img = cv2.cvtColor(plate_img_eq, cv2.COLOR_GRAY2BGR)
                                                
                                                # transform
                                                transformed_img = process_auto_transform(plate_img)
                                                
                                                # split
                                                top_img, _ = process_split_image(transformed_img)
                                                
                                                # read license
                                                results, confidence = process_read_license(top_img, model_path, font_path)
                                                
                                                # จบการจับเวลา OCR
                                                ocr_time = time.time() - ocr_start_time
                                                ocr_times.append(ocr_time)
                                                ocr_count += 1
                                                
                                                # คำนวณค่าเฉลี่ย
                                                avg_ocr_time = sum(ocr_times) / len(ocr_times)
                                                
                                                # แสดงข้อมูลเวลา OCR
                                                print(f"\n===== OCR ครั้งที่ {ocr_count} =====")
                                                print(f"OCR Time: {ocr_time*1000:.2f} ms")
                                                print(f"Average OCR Time: {avg_ocr_time*1000:.2f} ms")
                                                print(f"===================================")
                                                
                                                ocr_end_time = time.time()
                                                total_ocr_time = ocr_end_time - ocr_start_time

                                                text = ''.join([char for char, _, _ in results])
                                                
                                                if text and confidence >= min_confidence_threshold:  # ตรวจสอบความเชื่อมั่น
                                                    current_time = get_formatted_time()
                                                    
                                                    # ตรวจสอบว่าเป็นป้ายทะเบียนใหม่หรือไม่
                                                    if text != last_successful_plate:
                                                        print(f"\n📌 ตรวจพบป้ายทะเบียนใหม่:")
                                                        print(f"📝 ป้ายทะเบียน: {text}")
                                                        print(f"🎯 ความเชื่อมั่น: {confidence:.2%}")
                                                        print(f"🕒 เวลา: {current_time}")
                                                        print(f"⏱️ เวลาตั้งแต่พบป้ายจนถึงอ่านเสร็จ: {format_time_diff(plate_detection_time, ocr_end_time)}")
                                                        
                                                        # ส่งข้อมูลไปยัง GUI
                                                        ocr_results_queue.put({
                                                            'plate': text,
                                                            'time': current_time,
                                                            'confidence': confidence
                                                        })
                                                        
                                                        last_successful_plate = text
                                                        waiting_for_exit = True  # รอให้ป้ายทะเบียนออกจาก zone
                                                    
                                                    # อัพเดทเวลา OCR ล่าสุด
                                                    last_ocr_time = time.time()
                                                else:
                                                    print(f"❌ OCR ไม่ผ่านเกณฑ์ความเชื่อมั่น: {confidence:.2%} < {min_confidence_threshold:.2%}")

                                            except Exception as e:
                                                print(f"❌ OCR Error: {str(e)}")
                                                continue
                            else:
                                object_in_zone = False
            
            # กำหนดสีของ trigger zone ตามสถานะ
            trigger_zone_color = (0, 0, 255) if plate_in_zone else (255, 0, 0)  # แดงถ้ามีป้าย น้ำเงินถ้าไม่มี
            cv2.rectangle(original_frame, actual_trigger_zone[0], actual_trigger_zone[1], trigger_zone_color, 2)

            # ถ้าไม่มีป้ายทะเบียนใน zone และกำลังรอให้ป้ายออก
            if not plate_in_zone and waiting_for_exit:
                waiting_for_exit = False  # รีเซ็ตสถานะ เพื่อให้สามารถ OCR ป้ายถัดไปได้
                print(f"🚗 ป้ายทะเบียน {last_successful_plate} ออกจาก trigger zone แล้ว")
                
                # ยังคงเก็บค่าป้ายทะเบียนไว้เพื่อการส่งข้อมูล แต่พร้อมรับป้ายใหม่
                # last_successful_plate = None  # ไม่รีเซ็ตค่า เพื่อให้สามารถส่งข้อมูลได้

            
            # ควบคุมความถี่การอัพเดท GUI
            current_time = time.time()
            if current_time - last_gui_update >= GUI_UPDATE_INTERVAL:
                display_frame = cv2.resize(original_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                # ใช้ root.after เพื่อให้การอัพเดท GUI อยู่ใน main thread
                root.after(0, lambda: update_gui(display_frame, video_label))
                last_gui_update = current_time

            
            # อัพเดทข้อมูลป้ายทะเบียนและเวลาผ่าน main thread
            try:
                result = ocr_results_queue.get_nowait()
                if isinstance(result, dict):
                    plate_text = result.get('plate', '')
                    time_text = result.get('time', '')
                    root.after(0, lambda: plate_label.config(text=f"ทะเบียน: {plate_text}"))
                    root.after(0, lambda: time_label.config(text=f"เวลา: {time_text}"))
            except queue.Empty:
                pass

            # คำนวณเวลาที่ใช้ในการประมวลผลเฟรมนี้
            process_end_time = time.time()
            process_time = process_end_time - process_start_time
            processing_times.append(process_time)
            
            # อัพเดท FPS ทุก 30 เฟรม
            frame_count += 1
            if frame_count % 30 == 0:
                current_time = time.time()
                elapsed = current_time - fps_start_time
                fps = frame_count / elapsed
                avg_process_time = sum(processing_times) / len(processing_times) * 1000  # แปลงเป็น ms
                
                print(f"📊 สถิติการประมวลผล:")
                print(f"   - Processing FPS: {fps:.2f}")
                print(f"   - เวลาประมวลผลเฉลี่ย: {avg_process_time:.2f} ms")
                print(f"   - เวลา YOLO เฉลี่ย: {yolo_time*1000:.2f} ms")
                
                # รีเซ็ตตัวแปร
                frame_count = 0
                fps_start_time = current_time
                processing_times = []

            # ตรวจสอบการกดปิดโปรแกรม
            if not root.winfo_exists():
                print("🚪 หน้าต่างถูกปิด")
                stop_event.set()
                break

        except queue.Empty:
            continue
        except Exception as e:
            print(f"❌ ข้อผิดพลาดในการประมวลผล: {str(e)}")


def main():
    try:
        model_path = "readLicense/EfficientNet_model3.pth"
        yolo_model_path = "detectCar/yolov8/best_v8_2.pt"
        font_path = "AnantasonReno-SemiExpanded-Italic.otf"
        capture_dir = "captured_plates"
        os.makedirs(capture_dir, exist_ok=True)

        # ตั้งค่า GPIO
        setup_gpio()

        # ตั้งค่า ML
        model = YOLO(yolo_model_path)
        model.conf = 0.5
        model.max_det = 1

        # เตรียม camera
        #cap = cv2.VideoCapture("rtsp://admin:kmitl2025@192.168.1.64:554/stream")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("ไม่สามารถเปิดกล้องได้")
            
        # ตั้งค่า buffer ของเว็บแคม
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # คำนวณ trigger zone
        trigger_zone = (
            (int(YOLO_WIDTH * ZONE_LEFT/100), int(YOLO_HEIGHT * ZONE_TOP/100)),
            (int(YOLO_WIDTH * ZONE_RIGHT/100), int(YOLO_HEIGHT * ZONE_BOTTOM/100))
        )
        
        frame_queue = queue.Queue(maxsize=FRAME_BUFFER_SIZE)
        ocr_results_queue = queue.Queue()
        last_ocr_time = 0
        stop_event.clear()

        # สร้าง GUI
        root = tk.Tk()
        root.title("ระบบตรวจจับป้ายทะเบียน")
        
        # สร้าง frame หลัก
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # สร้าง label สำหรับแสดงวิดีโอ
        video_label = ttk.Label(main_frame)
        video_label.pack(pady=20)
        
        # สร้าง frame สำหรับข้อมูลป้ายทะเบียนแบบแนวตั้ง
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, padx=20, pady=5)
        
        # สร้าง labels สำหรับแสดงข้อมูลด้วยขนาดตัวอักษรที่ใหญ่ขึ้น
        plate_label = ttk.Label(info_frame, text="ทะเบียน: -", font=("Helvetica", 150))
        plate_label.pack(fill=tk.X, pady=20)
        
        time_label = ttk.Label(info_frame, text="เวลา: -", font=("Helvetica", 150))
        time_label.pack(fill=tk.X, pady=20)

        # สร้าง threads
        button_thread = threading.Thread(target=button_listener, args=(frame_queue,), daemon=True)
        ir_thread = threading.Thread(target=ir_light_controller, daemon=True)
        capture_thread = threading.Thread(target=capture_frame, args=(cap, frame_queue), daemon=True)
        
        # ส่ง root ให้ process_frame ด้วย
        process_thread = threading.Thread(
            target=process_frame,
            args=(frame_queue, model, last_ocr_time, trigger_zone, model_path, font_path,
                  ocr_results_queue, video_label, plate_label, time_label, root),
            daemon=True
        )

        # เริ่ม threads
        button_thread.start()
        ir_thread.start()
        capture_thread.start()
        process_thread.start()

        root.mainloop()

        print("กำลังปิดโปรแกรม...")
        stop_event.set()

        # รอให้ threads จบการทำงาน
        for thread in [capture_thread, process_thread, button_thread, ir_thread]:
            thread.join(timeout=2)

    except Exception as e:
        print(f"\nเกิดข้อผิดพลาด: {str(e)}")
        stop_event.set()
    finally:
        stop_event.set()
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        cleanup_gpio()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nปิดโปรแกรมด้วย Ctrl+C")
    finally:
        stop_event.set()
        cleanup_gpio()