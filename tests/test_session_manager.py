from agent_session import SessionManager

def test_session_lifecycle_and_bump():
    sm = SessionManager()
    s = sm.start(user_id=1, channel_id=2, default_coll="psy", style="terapis", mode="ringkas")
    assert s.is_on and s.default_collection == "psy"
    assert sm.get(1,2).turns == 0
    sm.bump_turn(1,2); sm.bump_turn(1,2)
    assert sm.get(1,2).turns == 2
    sm.set_topic(1,2,"uji")
    assert sm.get(1,2).topic == "uji"
    sm.end(1,2)
    assert sm.get(1,2).is_on is False
