"""
會議室預約相關 Tools
提供查詢大樓、查詢會議室、預約、查詢已預約、取消等功能
"""

from langchain_core.tools import tool
from typing import Optional, List
from datetime import datetime, timedelta
import random
import uuid


# ============================================================
# 模擬資料庫 (實際使用時替換為真實資料庫或 MCP)
# ============================================================

# 可預約大樓資料
BUILDINGS_DB = {
    "A": {"name": "A棟 - 總部大樓", "floors": 10, "address": "台北市信義區信義路100號"},
    "B": {"name": "B棟 - 研發中心", "floors": 8, "address": "台北市內湖區內湖路200號"},
    "C": {"name": "C棟 - 營運大樓", "floors": 6, "address": "台北市南港區南港路300號"},
}

# 會議室資料
ROOMS_DB = {
    "A": [
        {"id": "A-101", "name": "大會議室", "capacity": 30, "floor": 1, "equipment": ["投影機", "視訊設備", "白板"]},
        {"id": "A-201", "name": "中型會議室A", "capacity": 15, "floor": 2, "equipment": ["投影機", "白板"]},
        {"id": "A-202", "name": "中型會議室B", "capacity": 15, "floor": 2, "equipment": ["投影機", "視訊設備"]},
        {"id": "A-301", "name": "小型會議室", "capacity": 6, "floor": 3, "equipment": ["電視螢幕"]},
    ],
    "B": [
        {"id": "B-101", "name": "創意討論室", "capacity": 20, "floor": 1, "equipment": ["投影機", "白板", "視訊設備"]},
        {"id": "B-201", "name": "技術會議室", "capacity": 12, "floor": 2, "equipment": ["投影機", "大螢幕"]},
        {"id": "B-301", "name": "腦力激盪室", "capacity": 8, "floor": 3, "equipment": ["白板", "便利貼牆"]},
    ],
    "C": [
        {"id": "C-101", "name": "客戶接待室", "capacity": 10, "floor": 1, "equipment": ["投影機", "視訊設備", "茶水服務"]},
        {"id": "C-201", "name": "培訓教室", "capacity": 40, "floor": 2, "equipment": ["投影機", "麥克風", "錄影設備"]},
    ],
}

# 模擬預約資料庫 (實際使用時應該是持久化儲存)
RESERVATIONS_DB: List[dict] = [
    {
        "id": "RES-001",
        "user_id": "user001",
        "room_id": "A-101",
        "building": "A",
        "date": "2025-12-16",
        "start_time": "09:00",
        "end_time": "10:00",
        "title": "週一例會",
        "created_at": "2025-12-14T10:00:00",
    },
    {
        "id": "RES-002",
        "user_id": "user001",
        "room_id": "B-201",
        "building": "B",
        "date": "2025-12-17",
        "start_time": "14:00",
        "end_time": "16:00",
        "title": "技術討論",
        "created_at": "2025-12-14T11:00:00",
    },
]


# ============================================================
# Tool 1: 查詢可預約大樓
# ============================================================
@tool
def get_available_buildings() -> str:
    """
    查詢所有可預約的大樓列表。
    當用戶想要預約會議室或詢問有哪些大樓可以選擇時使用此工具。
    
    Returns:
        可預約大樓的詳細資訊
    """
    result = "【可預約大樓列表】\n\n"
    
    for building_code, info in BUILDINGS_DB.items():
        result += f"📍 {info['name']}\n"
        result += f"   代碼: {building_code}\n"
        result += f"   樓層數: {info['floors']} 層\n"
        result += f"   地址: {info['address']}\n\n"
    
    result += "💡 提示：請告訴我您想預約哪棟大樓，以及預約日期，我可以幫您查詢可用的會議室。"
    return result


# ============================================================
# Tool 2: 查詢可預約會議室 (by 大樓、日期)
# ============================================================
@tool
def get_available_rooms(building_code: str, date: str) -> str:
    """
    查詢指定大樓在特定日期的可預約會議室。
    
    Args:
        building_code: 大樓代碼 (A, B, 或 C)
        date: 查詢日期，格式為 YYYY-MM-DD (例如: 2025-12-16)
    
    Returns:
        該大樓在指定日期可預約的會議室列表及可用時段
    """
    building_code = building_code.upper()
    
    # 驗證大樓代碼
    if building_code not in BUILDINGS_DB:
        return f"❌ 錯誤：找不到大樓代碼 '{building_code}'。有效的大樓代碼為: {', '.join(BUILDINGS_DB.keys())}"
    
    # 驗證日期格式
    try:
        query_date = datetime.strptime(date, "%Y-%m-%d")
        if query_date.date() < datetime.now().date():
            return "❌ 錯誤：無法查詢過去的日期，請選擇今天或之後的日期。"
    except ValueError:
        return "❌ 錯誤：日期格式不正確，請使用 YYYY-MM-DD 格式（例如: 2025-12-16）。"
    
    building_info = BUILDINGS_DB[building_code]
    rooms = ROOMS_DB.get(building_code, [])
    
    # 取得該日期已預約的時段
    booked_slots = {}
    for res in RESERVATIONS_DB:
        if res["building"] == building_code and res["date"] == date:
            room_id = res["room_id"]
            if room_id not in booked_slots:
                booked_slots[room_id] = []
            booked_slots[room_id].append(f"{res['start_time']}-{res['end_time']}")
    
    result = f"【{building_info['name']} - {date} 可預約會議室】\n\n"
    
    # 定義可預約時段
    time_slots = ["09:00-10:00", "10:00-11:00", "11:00-12:00", 
                  "13:00-14:00", "14:00-15:00", "15:00-16:00", 
                  "16:00-17:00", "17:00-18:00"]
    
    for room in rooms:
        result += f"🚪 {room['name']} ({room['id']})\n"
        result += f"   容納人數: {room['capacity']} 人\n"
        result += f"   樓層: {room['floor']}F\n"
        result += f"   設備: {', '.join(room['equipment'])}\n"
        
        # 顯示可用時段
        room_booked = booked_slots.get(room['id'], [])
        available_slots = [slot for slot in time_slots if slot not in room_booked]
        
        if available_slots:
            result += f"   ✅ 可用時段: {', '.join(available_slots[:4])}"
            if len(available_slots) > 4:
                result += f" ...等 {len(available_slots)} 個時段"
            result += "\n"
        else:
            result += f"   ❌ 當日已無可用時段\n"
        
        result += "\n"
    
    result += "💡 提示：請告訴我您要預約的會議室ID、時間區段和會議主題，我可以幫您完成預約。"
    return result


# ============================================================
# Tool 3: 預約會議室
# ============================================================
@tool
def book_meeting_room(
    room_id: str,
    date: str,
    start_time: str,
    end_time: str,
    title: str,
    user_id: str = "default_user"
) -> str:
    """
    預約會議室。
    
    Args:
        room_id: 會議室ID (例如: A-101, B-201)
        date: 預約日期，格式為 YYYY-MM-DD
        start_time: 開始時間，格式為 HH:MM (例如: 09:00)
        end_time: 結束時間，格式為 HH:MM (例如: 10:00)
        title: 會議主題
        user_id: 預約者的用戶ID (預設為 default_user)
    
    Returns:
        預約結果，包含預約確認編號或錯誤訊息
    """
    # 解析 room_id 取得大樓代碼
    building_code = room_id.split("-")[0].upper() if "-" in room_id else ""
    
    # 驗證大樓代碼
    if building_code not in BUILDINGS_DB:
        return f"❌ 預約失敗：無效的會議室ID '{room_id}'。"
    
    # 驗證會議室是否存在
    rooms = ROOMS_DB.get(building_code, [])
    room = next((r for r in rooms if r["id"].upper() == room_id.upper()), None)
    if not room:
        available_rooms = [r["id"] for r in rooms]
        return f"❌ 預約失敗：找不到會議室 '{room_id}'。{building_code}棟可用的會議室: {', '.join(available_rooms)}"
    
    # 驗證日期
    try:
        booking_date = datetime.strptime(date, "%Y-%m-%d")
        if booking_date.date() < datetime.now().date():
            return "❌ 預約失敗：無法預約過去的日期。"
    except ValueError:
        return "❌ 預約失敗：日期格式不正確，請使用 YYYY-MM-DD 格式。"
    
    # 驗證時間格式
    try:
        start = datetime.strptime(start_time, "%H:%M")
        end = datetime.strptime(end_time, "%H:%M")
        if start >= end:
            return "❌ 預約失敗：結束時間必須晚於開始時間。"
    except ValueError:
        return "❌ 預約失敗：時間格式不正確，請使用 HH:MM 格式（例如: 09:00）。"
    
    # 檢查時段是否已被預約
    for res in RESERVATIONS_DB:
        if (res["room_id"].upper() == room_id.upper() and 
            res["date"] == date):
            # 檢查時間是否重疊
            res_start = datetime.strptime(res["start_time"], "%H:%M")
            res_end = datetime.strptime(res["end_time"], "%H:%M")
            if not (end <= res_start or start >= res_end):
                return f"❌ 預約失敗：該時段已被預約（{res['start_time']}-{res['end_time']} {res['title']}）。"
    
    # 建立預約
    reservation_id = f"RES-{uuid.uuid4().hex[:6].upper()}"
    new_reservation = {
        "id": reservation_id,
        "user_id": user_id,
        "room_id": room_id.upper(),
        "building": building_code,
        "date": date,
        "start_time": start_time,
        "end_time": end_time,
        "title": title,
        "created_at": datetime.now().isoformat(),
    }
    RESERVATIONS_DB.append(new_reservation)
    
    # 取得大樓和會議室資訊
    building_name = BUILDINGS_DB[building_code]["name"]
    
    result = "✅ 預約成功！\n\n"
    result += f"📋 預約確認單\n"
    result += f"{'='*40}\n"
    result += f"預約編號: {reservation_id}\n"
    result += f"會議主題: {title}\n"
    result += f"大樓: {building_name}\n"
    result += f"會議室: {room['name']} ({room_id})\n"
    result += f"日期: {date}\n"
    result += f"時間: {start_time} - {end_time}\n"
    result += f"容納人數: {room['capacity']} 人\n"
    result += f"設備: {', '.join(room['equipment'])}\n"
    result += f"{'='*40}\n\n"
    result += f"💡 提示：請記住您的預約編號 {reservation_id}，如需取消可使用此編號。"
    
    return result


# ============================================================
# Tool 4: 查詢已預約會議室 (by user_id)
# ============================================================
@tool
def get_user_reservations(user_id: str = "default_user") -> str:
    """
    查詢用戶已預約的會議室列表。
    
    Args:
        user_id: 用戶ID，用於查詢該用戶的所有預約
    
    Returns:
        該用戶的所有預約記錄
    """
    user_reservations = [r for r in RESERVATIONS_DB if r["user_id"] == user_id]
    
    if not user_reservations:
        return f"📭 用戶 {user_id} 目前沒有任何會議室預約記錄。\n\n💡 提示：如需預約會議室，請告訴我您想預約的大樓和日期。"
    
    # 按日期排序
    user_reservations.sort(key=lambda x: (x["date"], x["start_time"]))
    
    result = f"【用戶 {user_id} 的預約記錄】\n\n"
    result += f"共 {len(user_reservations)} 筆預約\n"
    result += f"{'='*40}\n\n"
    
    for i, res in enumerate(user_reservations, 1):
        building_name = BUILDINGS_DB.get(res["building"], {}).get("name", res["building"])
        
        # 取得會議室名稱
        rooms = ROOMS_DB.get(res["building"], [])
        room = next((r for r in rooms if r["id"] == res["room_id"]), None)
        room_name = room["name"] if room else res["room_id"]
        
        # 判斷是否為過去的預約
        res_date = datetime.strptime(res["date"], "%Y-%m-%d").date()
        is_past = res_date < datetime.now().date()
        status = "⏰ 已結束" if is_past else "📅 即將到來"
        
        result += f"{i}. {status}\n"
        result += f"   預約編號: {res['id']}\n"
        result += f"   會議主題: {res['title']}\n"
        result += f"   大樓: {building_name}\n"
        result += f"   會議室: {room_name} ({res['room_id']})\n"
        result += f"   日期時間: {res['date']} {res['start_time']}-{res['end_time']}\n"
        result += "\n"
    
    result += f"{'='*40}\n"
    result += "💡 提示：如需取消預約，請提供預約編號。"
    
    return result


# ============================================================
# Tool 5: 取消會議室預約
# ============================================================
@tool
def cancel_reservation(reservation_id: str, user_id: str = "default_user") -> str:
    """
    取消會議室預約。
    
    Args:
        reservation_id: 要取消的預約編號 (例如: RES-001)
        user_id: 用戶ID，用於驗證預約所有權
    
    Returns:
        取消結果
    """
    # 尋找預約
    reservation = None
    reservation_index = None
    
    for i, res in enumerate(RESERVATIONS_DB):
        if res["id"].upper() == reservation_id.upper():
            reservation = res
            reservation_index = i
            break
    
    if not reservation:
        return f"❌ 取消失敗：找不到預約編號 '{reservation_id}'。\n\n💡 提示：請使用「查詢我的預約」功能確認正確的預約編號。"
    
    # 驗證所有權
    if reservation["user_id"] != user_id:
        return f"❌ 取消失敗：您沒有權限取消此預約（預約編號: {reservation_id}）。"
    
    # 檢查是否為過去的預約
    res_date = datetime.strptime(reservation["date"], "%Y-%m-%d").date()
    if res_date < datetime.now().date():
        return f"❌ 取消失敗：無法取消已過期的預約。"
    
    # 取得詳細資訊用於確認訊息
    building_name = BUILDINGS_DB.get(reservation["building"], {}).get("name", reservation["building"])
    rooms = ROOMS_DB.get(reservation["building"], [])
    room = next((r for r in rooms if r["id"] == reservation["room_id"]), None)
    room_name = room["name"] if room else reservation["room_id"]
    
    # 執行取消
    RESERVATIONS_DB.pop(reservation_index)
    
    result = "✅ 預約已成功取消！\n\n"
    result += f"📋 取消確認\n"
    result += f"{'='*40}\n"
    result += f"預約編號: {reservation_id}\n"
    result += f"會議主題: {reservation['title']}\n"
    result += f"大樓: {building_name}\n"
    result += f"會議室: {room_name} ({reservation['room_id']})\n"
    result += f"原預約時間: {reservation['date']} {reservation['start_time']}-{reservation['end_time']}\n"
    result += f"{'='*40}\n\n"
    result += "💡 提示：如需重新預約，請告訴我您想預約的大樓和日期。"
    
    return result


# ============================================================
# 工具列表導出
# ============================================================

# 預約流程工具 (Booking Agent 使用)
BOOKING_TOOLS = [
    get_available_buildings,
    get_available_rooms,
    book_meeting_room,
]

# 查詢工具 (Query Agent 使用)
QUERY_TOOLS = [
    get_available_buildings,
    get_available_rooms,
]

# 管理工具 (Management Agent 使用)
MANAGEMENT_TOOLS = [
    get_user_reservations,
    cancel_reservation,
]

# 所有工具
MEETING_ROOM_TOOLS = [
    get_available_buildings,
    get_available_rooms,
    book_meeting_room,
    get_user_reservations,
    cancel_reservation,
]
