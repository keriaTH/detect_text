import cv2
import pytesseract
import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from skimage.exposure import is_low_contrast

# Đường dẫn đến Tesseract executable (Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Biến global để lưu tọa độ crop
cropping = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0
original_image = None

def mouse_crop(event, x, y, flags, param):
    global x_start, y_start, x_end, y_end, cropping, original_image
    
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping:
            x_end, y_end = x, y
            # Vẽ hình chữ nhật để hiển thị vùng đang được chọn
            img_copy = original_image.copy()
            cv2.rectangle(img_copy, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            cv2.imshow('Image', img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        x_end, y_end = x, y
        cropping = False
        
        # Đảm bảo tọa độ không âm và theo thứ tự đúng
        x_start, x_end = min(x_start, x_end), max(x_start, x_end)
        y_start, y_end = min(y_start, y_end), max(y_start, y_end)

def crop_image(image, x_start, y_start, x_end, y_end):
    return image[y_start:y_end, x_start:x_end]

def extract_text(image):
    # Chuyển ảnh sang grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # OCR text
    text = pytesseract.image_to_string(gray)
    return text

def check_text_contrast(image):
    # Chuyển sang không gian màu LAB
    lab_img = rgb2lab(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Kiểm tra độ tương phản
    is_difficult = is_low_contrast(lab_img)
    
    if is_difficult:
        return "Text có thể khó đọc do độ tương phản thấp"
    return "Text có độ tương phản tốt"

def create_text_mask(image):
    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Áp dụng threshold để tạo mặt nạ nhị phân
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Tạo kernel cho morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    
    # Áp dụng morphology để làm sạch nhiễu
    mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return mask

def analyze_text_contrast(image, mask):
    # Chuyển BGR sang RGB
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Tạo mặt nạ boolean
    text_pixels = mask == 0
    background_pixels = mask == 255
    
    # Lấy các pixel text và background
    text_rgb = rgb_img[text_pixels]
    bg_rgb = rgb_img[background_pixels]
    
    if len(text_rgb) > 0 and len(bg_rgb) > 0:
        # Chuyển sang LAB
        text_lab = rgb2lab(text_rgb.reshape(-1, 1, 3))
        bg_lab = rgb2lab(bg_rgb.reshape(-1, 1, 3))
        
        # Tính độ sáng trung bình
        text_L = np.mean(text_lab[:, 0, 0])
        bg_L = np.mean(bg_lab[:, 0, 0])
        
        # Tính độ tương phản
        contrast_ratio = abs(text_L - bg_L)
        
        # Phân tích chi tiết
        analysis = ""
        if contrast_ratio < 15:
            analysis = "Text rất khó đọc:\n"
            analysis += f"- Độ tương phản quá thấp ({contrast_ratio:.1f})\n"
            analysis += "- Cần tăng độ tương phản giữa text và nền"
        elif contrast_ratio < 30:
            analysis = "Text khó đọc:\n"
            analysis += f"- Độ tương phản thấp ({contrast_ratio:.1f})\n"
            analysis += "- Nên cải thiện độ tương phản"
        elif contrast_ratio < 45:
            analysis = "Text đọc được nhưng chưa tốt:\n"
            analysis += f"- Độ tương phản trung bình ({contrast_ratio:.1f})\n"
            analysis += "- Có thể cải thiện thêm"
        else:
            analysis = "Text dễ đọc:\n"
            analysis += f"- Độ tương phản tốt ({contrast_ratio:.1f})\n"
            analysis += "- Đạt tiêu chuẩn về khả năng đọc"
            
        # Thêm thông tin về độ sáng
        analysis += f"\nĐộ sáng text: {text_L:.1f}"
        analysis += f"\nĐộ sáng nền: {bg_L:.1f}"
        
        return analysis
    
    return "Không tìm thấy text trong vùng đã chọn"

def main():
    global original_image
    # Đường dẫn ảnh input
    image_path = r"E:\Python_code\Xu li anh\anh1.jpg"
    
    # Đọc ảnh
    original_image = cv2.imread(image_path)
    img_copy = original_image.copy()
    
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', mouse_crop)
    
    print("Kéo chuột để chọn vùng cần crop. Nhấn 'c' để xác nhận, 'r' để chọn lại, 'q' để thoát.")
    
    while True:
        cv2.imshow('Image', img_copy)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            if x_end - x_start > 0 and y_end - y_start > 0:
                cropped_img = crop_image(original_image, x_start, y_start, x_end, y_end)
                
                # Tạo mặt nạ text
                text_mask = create_text_mask(cropped_img)
                
                # Phân tích độ tương phản của text
                contrast_analysis = analyze_text_contrast(cropped_img, text_mask)
                print("Kết quả phân tích:", contrast_analysis)
                
                # Trích xuất text
                text = extract_text(cropped_img)
                print("Text được nhận dạng:", text)
                
                # Hiển thị ảnh đã crop
                cv2.imshow("Cropped Image", cropped_img)
                cv2.imshow("Text Mask", text_mask)
                
        elif key == ord('r'):
            img_copy = original_image.copy()
            
        elif key == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()