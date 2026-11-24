import pytest
import gameplay.vehicle_seat_handoff as mod

CASES = [
    ("refresh_input_on_handoff", (1,)),
    ("atomic_occupy", (1,)),
    ("reclaim_on_disconnect", (1,)),
    ("clear_seat_first", (1,)),
    ("reset_on_respawn", (1,)),
    ("exit_with_current_xform", (1,)),
    ("board", (1,)),
    ("switch_seat", (1,)),
    ("takeover", (1,)),
    ("destroy", (1,)),
    ("validate_vehicle", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
