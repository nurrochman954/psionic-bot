# agent_memory.py

import os
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional

ASIA_JAKARTA = timezone(timedelta(hours=7))
BASE_DIR = os.path.join("storage", "memory")

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _user_dir(user_id: int) -> str:
    d = os.path.join(BASE_DIR, str(user_id))
    _ensure_dir(d)
    return d

def _daily_path(user_id: int, date_str: Optional[str] = None) -> str:
    if not date_str:
        date_str = datetime.now(ASIA_JAKARTA).strftime("%Y-%m-%d")
    dd = os.path.join(_user_dir(user_id), "daily")
    _ensure_dir(dd)
    return os.path.join(dd, f"{date_str}.json")

def _rolling_path(user_id: int) -> str:
    return os.path.join(_user_dir(user_id), "rolling.json")

def append_turn(user_id: int, question: str, answer: str):
    p = _daily_path(user_id)
    data = {"turns": [], "daily_summary": ""}
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    data["turns"].append({"q": question, "a": answer})
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def update_daily_summary(user_id: int, summary_text: str):
    p = _daily_path(user_id)
    data = {"turns": [], "daily_summary": ""}
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    data["daily_summary"] = summary_text.strip()
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def read_daily(user_id: int, date_str: Optional[str] = None) -> Dict:
    p = _daily_path(user_id, date_str)
    if not os.path.exists(p):
        return {"turns": [], "daily_summary": ""}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def update_rolling_summary(user_id: int, summary_text: str):
    p = _rolling_path(user_id)
    data = {"summary": "", "updated_at": ""}
    data["summary"] = summary_text.strip()
    data["updated_at"] = datetime.now(ASIA_JAKARTA).isoformat()
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def read_rolling_summary(user_id: int) -> str:
    p = _rolling_path(user_id)
    if not os.path.exists(p):
        return ""
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("summary", "")

def yesterday_date_str() -> str:
    return (datetime.now(ASIA_JAKARTA) - timedelta(days=1)).strftime("%Y-%m-%d")
