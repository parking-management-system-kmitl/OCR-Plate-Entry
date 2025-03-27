import cv2

def main():
    # ลองเปิดกล้องตั้งแต่ 0 ถึง 10 (สามารถเพิ่มหรือลดตามจำนวนที่คาดว่ามีกล้องเชื่อมต่อ)
    connected_cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            connected_cameras.append(i)
        cap.release()  # ปิดการเชื่อมต่อกับกล้องหลังตรวจสอบแล้ว

    if connected_cameras:
        print(f"กล้องที่เชื่อมต่ออยู่: {connected_cameras}")
    else:
        print("ไม่มีการเชื่อมต่อกล้อง")

if __name__ == "__main__":
    main()