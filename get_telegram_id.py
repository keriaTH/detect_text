"""
Chạy file này để lấy Telegram Chat ID của bạn.

Bước 1: Nhắn @BotFather trên Telegram → gõ /newbot → đặt tên → copy token
Bước 2: Nhắn bot của bạn 1 tin bất kỳ (vd: "hello")
Bước 3: Chạy file này, nhập token → copy chat_id hiện ra
Bước 4: Điền token và chat_id vào hi.py
"""

import requests

token = input("Nhập Telegram Bot Token: ").strip()

try:
    resp = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=10)
    data = resp.json()

    if not data.get("ok"):
        print(f"\n❌ Token không hợp lệ: {data.get('description')}")
        exit()

    updates = data.get("result", [])
    if not updates:
        print("\n⚠️  Chưa có tin nhắn nào.")
        print("→ Hãy nhắn bot của bạn 1 tin bất kỳ rồi chạy lại file này.")
        exit()

    print("\n✅ Tìm thấy các cuộc trò chuyện:\n")
    seen = set()
    for upd in updates:
        msg = upd.get("message", {})
        chat = msg.get("chat", {})
        chat_id   = chat.get("id")
        chat_type = chat.get("type", "")
        name = chat.get("first_name") or chat.get("title") or "?"
        username = chat.get("username", "")

        if chat_id and chat_id not in seen:
            seen.add(chat_id)
            print(f"  Tên     : {name} {('@'+username) if username else ''}")
            print(f"  Loại    : {chat_type}")
            print(f"  Chat ID : {chat_id}  ← copy cái này")
            print()

    print("─" * 40)
    print("Điền vào hi.py:")
    print(f'  "telegram_token"      : "{token}"')
    print(f'  "telegram_allowed_ids": [{list(seen)[0] if seen else "???"}]')

except Exception as e:
    print(f"\n❌ Lỗi: {e}")
