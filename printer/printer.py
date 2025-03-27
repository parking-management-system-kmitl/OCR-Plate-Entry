import sys

from escpos.printer import Usb
from PIL import Image, ImageDraw, ImageFont
import qrcode

p = Usb(0x0483, 0x070B, 0)

try:
    # รับค่าหมายเลขทะเบียนจาก argument
    plate_number = sys.argv[1] if len(sys.argv) > 1 else "UNKNOWN"
    #plate_number = "kmitl"

    print(plate_number)

    line = u'--------------------------------------------'
    text1 = u'KMITL PARKING'
    text2 = u'บัตรจอดรถ '
    #text3 = u'Date: 17/08/2024  Time: 11:00:56'
    text4 = u'เลขทะเบียน'
    text5 = f'{plate_number}'  # ใช้ตัวแปรแทนค่าเดิม
    text6_1 = u'สแกน QR code นี้'
    text6_2 = u'เพื่อตรวจสอบค่าจอดรถ'
    text7_1 = u'กรุณาชำระค่าบริการก่อนนำรถ'
    text7_2 = u'ออกจากพื้นที่จอดรถ'
    text8 = u'ข้อกำหนดและเงื่อนไข'
    text9_1 = u'1.นำกล้องโทรศัพท์มือถือสแกน QR code ที่บัตรจอดรถ'
    text9_2 = u'เพื่อตรวจสอบและชำระค่าที่จอดรถ'
    text10_1 = u'2.หากซื้อสินค้าและบริการภายในศูนย์การค้าครบ 500 บาท'
    text10_2 = u'ขึ้นไปสามารถใช้ส่วนลดค่าที่จอดได้ที่ประชาสัมพันธ์'
    text10_3 = u'(ตามที่เงื่อนไขกำหนด)'
    text11 = u'*** บัตรหายปรับ 200 บาท ***'
    text12_1 = u'กรุณาเก็บรักษาบัตรจอดรถ'
    text12_2 = u'กรณีบัตรหายติดต่อประชาสัมพันธ์'

    print("โหลด Font")
    # Import Fontsใใ
    font_large = ImageFont.truetype('fonts/THSarabun.ttf', 48)  # ฟอนต์ขนาดใหญ่
    font_medium = ImageFont.truetype('fonts/THSarabun.ttf', 36)  # ฟอนต์ขนาดกลาง
    font_small = ImageFont.truetype('fonts/THSarabun.ttf', 30)  # ฟอนต์ขนาดเล็ก
    font_mini = ImageFont.truetype('fonts/THSarabun.ttf',24) # ฟอนต์ขนาดเล็กมาก
    # Bold Fonts
    font_large_bold = ImageFont.truetype('fonts/THSarabun-Bold.ttf', 48)  # ฟอนต์ขนาดใหญ่
    font_medium_bold = ImageFont.truetype('fonts/THSarabun-Bold.ttf', 36)  # ฟอนต์ขนาดกลาง
    font_small_bold = ImageFont.truetype('fonts/THSarabun-Bold.ttf', 30)  # ฟอนต์ขนาดเล็ก
    

    print("โหลด Font สำเร็จ")

    # สร้างภาพ
    image = Image.new('RGB', (384, 930), 'white')  # ปรับขนาดให้พอเหมาะ
    draw = ImageDraw.Draw(image)

    # ฟังก์ชันคำนวณตำแหน่งกลาง
    def draw_centered_text(text, y_position, font, draw):
        # หาขนาดของข้อความ
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        x_position = (image.width - text_width) // 2  # คำนวณตำแหน่งกลาง
        draw.text((x_position-5, y_position), text, font=font, fill='black')

    draw_centered_text(text1, 10, font_medium_bold, draw)  # บจก.เซ็นทรัล สแควร์
    draw_centered_text(text2, 50, font_medium_bold, draw) # บัตรจอดรถ
    #draw_centered_text(text3, 90, font_small, draw) # Date
    draw_centered_text(line, 110, font_medium, draw) # Line
    draw_centered_text(text4, 140, font_medium, draw) # เลขทะเบียน
    draw_centered_text(text5, 160, font_large_bold, draw) # 2กฐ452

    # สร้าง QR Code
    qr = qrcode.QRCode(
        version=1,  # ขนาดของ QR Code
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(f'http://jjsornwakii.3bbddns.com:34721/landing?id={plate_number}')  # ฝังทะเบียนรถใน QR
    qr.make(fit=True)

    # Create QR Code Image
    qr_image = qr.make_image(fill='black', back_color='white')

    # QR Code Size
    qr_image = qr_image.resize((300, 300))

    # คำนวณตำแหน่ง X ของ QR Code ให้อยู่กลาง
    qr_x_position = (image.width - qr_image.width) // 2  # ตำแหน่งกลางในแนวนอน

    # แทรก QR Code
    image.paste(qr_image, (qr_x_position-10, 210))  # QR Code ตำแหน่งระหว่าง text5 และ text6

    # Text After QR Code
    draw_centered_text(text6_1, 500,font_large_bold, draw)  # แสกน QR Code นี้
    draw_centered_text(text6_2, 540,font_large_bold, draw)  # เพื่อชำระค่าจอดรถ
    draw_centered_text(line, 580, font_medium, draw) # Line
    draw_centered_text(text7_1, 610,font_small_bold,draw) # กรุณาชำระค่าบริการก่อนนำรถ
    draw_centered_text(text7_2, 635, font_small_bold, draw) # ออกจากพื้นที่จอดรถ
    draw_centered_text(text8, 670,font_medium_bold,draw) # ข้อกำหนดและเงื่อนไข
    
    # draw_centered_text(text9_1, 710,font_mini,draw) 
    # draw_centered_text(text9_2, 730,font_mini,draw) 
    # draw_centered_text(text10_1, 750,font_mini,draw) 
    # draw_centered_text(text10_2, 770,font_mini,draw) 
    # draw_centered_text(text10_3, 790,font_mini,draw) 
    
    draw.text((10, 710), text9_1, font=font_mini, fill='black') 
    draw.text((10, 735), text9_2, font=font_mini, fill='black') 
    draw.text((10, 760), text10_1, font=font_mini, fill='black') 
    draw.text((10, 785), text10_2, font=font_mini, fill='black') 
    draw.text((10, 810), text10_3, font=font_mini, fill='black') 
    
    draw_centered_text(text11, 850,font_medium_bold,draw) 
    draw_centered_text(text12_1, 885,font_small,draw) 
    draw_centered_text(text12_2, 910,font_small,draw) 
    
    
    # draw.text((10, 370), text8, font=font_small, fill='black')  # ฟอนต์ขนาดเล็ก
    # draw.text((10, 410), text9, font=font_small, fill='black')  # ฟอนต์ขนาดเล็ก
    # draw.text((10, 450), text10, font=font_small, fill='black')  # ฟอนต์ขนาดเล็ก
    # draw.text((10, 490), text11, font=font_small, fill='black')  # ฟอนต์ขนาดเล็ก
    # draw.text((10, 530), text12, font=font_small, fill='black')  # ฟอนต์ขนาดเล็ก
    # draw.text((10, 570), text13, font=font_small, fill='black')  # ฟอนต์ขนาดเล็ก
    # draw.text((10, 610), "--------------------------------", font=font_small, fill='black')  # ฟอนต์ขนาดเล็ก
    
    # ส่งภาพไปยังเครื่องพิมพ์
    p.image(image)

    # ตัดกระดาษ
    p.cut()

except Exception as e:
    print("PRINTER ERROR: ", e)
