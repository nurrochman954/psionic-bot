import json
import os
import agent_memory as mem

def test_append_and_read_daily(tmp_path, monkeypatch):
    # arahkan BASE_DIR ke temp
    monkeypatch.setattr(mem, "BASE_DIR", os.path.join(tmp_path, "memory"))
    user = 42

    mem.append_turn(user, "Q1", "A1")
    mem.append_turn(user, "Q2", "A2")
    data = mem.read_daily(user)
    assert len(data["turns"]) == 2
    assert data["turns"][0]["q"] == "Q1"

    mem.update_daily_summary(user, "hari produktif")
    data2 = mem.read_daily(user)
    assert data2["daily_summary"] == "hari produktif"

def test_rolling_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(mem, "BASE_DIR", os.path.join(tmp_path, "memory"))
    user = 7
    assert mem.read_rolling_summary(user) == ""
    mem.update_rolling_summary(user, "ringkas kemarin")
    assert mem.read_rolling_summary(user).startswith("ringkas kemarin")
