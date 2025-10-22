import cv2
import pytesseract
import numpy as np

# Cấu hình đường dẫn Tesseract (Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Biến global để lưu tọa độ crop
cropping = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0
original_image = None


# ====================== HÀM HỖ TRỢ ======================

def mouse_crop(event, x, y, flags, param):
    """Xử lý thao tác kéo chuột để chọn vùng crop"""
    global x_start, y_start, x_end, y_end, cropping, original_image
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True
    elif event == cv2.EVENT_MOUSEMOVE and cropping:
        x_end, y_end = x, y
        img_copy = original_image.copy()
        cv2.rectangle(img_copy, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        cv2.imshow('Ảnh gốc', img_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        x_end, y_end = x, y
        cropping = False
        x_start, x_end = min(x_start, x_end), max(x_start, x_end)
        y_start, y_end = min(y_start, y_end), max(y_start, y_end)


def crop_image(image, x_start, y_start, x_end, y_end):
    """Cắt vùng ảnh"""
    return image[y_start:y_end, x_start:x_end]


def extract_text(image):
    """Nhận dạng văn bản từ vùng ảnh"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    gray = cv2.medianBlur(gray, 3)
    text = pytesseract.image_to_string(gray, lang='eng')
    return text.strip()


# ====================== PHÂN TÍCH ĐỘ TƯƠNG PHẢN ======================

def luminance(rgb):
    """Tính độ sáng tương đối của 1 màu RGB (theo chuẩn WCAG)"""
    rgb = np.array(rgb) / 255.0
    r, g, b = [
        (c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
        for c in rgb
    ]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(rgb1, rgb2):
    """Tính tỷ lệ tương phản giữa 2 màu"""
    L1, L2 = luminance(rgb1), luminance(rgb2)
    return (max(L1, L2) + 0.05) / (min(L1, L2) + 0.05)


def create_text_mask(image):
    """Tạo mặt nạ text bằng threshold + morphology"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return mask


def analyze_text_contrast(image, mask):
    """Phân tích độ tương phản giữa text và background"""
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    text_mask = mask < 128
    bg_mask = mask >= 128

    if np.sum(text_mask) == 0 or np.sum(bg_mask) == 0:
        return "Không tìm thấy text", None, None, None

    avg_text_rgb = np.mean(rgb_img[text_mask], axis=0)
    avg_bg_rgb = np.mean(rgb_img[bg_mask], axis=0)
    ratio = round(contrast_ratio(avg_text_rgb, avg_bg_rgb), 2)

    # Phân loại mức độ dễ đọc
    if ratio < 3:
        label, color = "Rat kho doc", (0, 0, 255)
    elif ratio < 4.5:
        label, color = "Kho doc", (0, 165, 255)
    else:
        label, color = "de doc", (0, 255, 0)

    result_text = f"{label} (Contrast={ratio})"
    return result_text, ratio, color, (avg_text_rgb, avg_bg_rgb)


# ====================== CHƯƠNG TRÌNH CHÍNH ======================

def main():
    global original_image

    # Cho phép nhập đường dẫn ảnh từ bàn phím
    image_path = input("Nhập đường dẫn ảnh cần OCR: ").strip()
    original_image = cv2.imread(image_path)

    if original_image is None:
        print("Không thể đọc ảnh! Hãy kiểm tra lại đường dẫn.")
        return

    # Hiển thị ảnh lớn
    cv2.namedWindow('Ảnh gốc', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Ảnh gốc', 1200, 800)
    cv2.setMouseCallback('Ảnh gốc', mouse_crop)

    print("Kéo chuột để chọn vùng crop. Nhấn 'c' để xác nhận, 'r' để chọn lại, 'q' để thoát.")

    while True:
        cv2.imshow('Ảnh gốc', original_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            if x_end - x_start > 0 and y_end - y_start > 0:
                cropped_img = crop_image(original_image, x_start, y_start, x_end, y_end)

                # Tạo mask và phân tích độ tương phản
                text_mask = create_text_mask(cropped_img)
                contrast_result, ratio, color, _ = analyze_text_contrast(cropped_img, text_mask)
                text = extract_text(cropped_img)

                # In kết quả
                print("Kết quả phân tích:", contrast_result)
                print("Text OCR:", text)

                # Hiển thị kết quả
                display_img = cropped_img.copy()
                cv2.putText(display_img, contrast_result, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(display_img, f"OCR: {text}", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cv2.namedWindow("Kết quả", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Kết quả", 900, 600)
                cv2.imshow("Kết quả", display_img)

                cv2.namedWindow("Mặt nạ text", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Mặt nạ text", 900, 600)
                cv2.imshow("Mặt nạ text", text_mask)

        elif key == ord('r'):
            print("Đã chọn lại vùng mới.")
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
