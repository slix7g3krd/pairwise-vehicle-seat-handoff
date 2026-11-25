"""六个缺陷修复的行为测试。

每个测试类对应一个修复点；全部使用独立的 VehicleWorld，避免污染全局状态。
"""

import threading

import pytest

from gameplay.vehicle_seat_handoff import (
    OperationResult,
    Transform,
    VehicleWorld,
    atomic_occupy,
    audit,
    board,
    clear_seat_first,
    destroy,
    determinism,
    exit_with_current_xform,
    fallback,
    metrics,
    reclaim_on_disconnect,
    refresh_input_on_handoff,
    register_player,
    reset_on_respawn,
    spawn_vehicle,
    switch_seat,
    takeover,
    validate_vehicle,
)


@pytest.fixture
def world():
    return VehicleWorld()


def make_world(seat_count=2):
    w = VehicleWorld()
    spawn_vehicle("v1", seat_count=seat_count, transform=Transform(10, 0, 5), world=w)
    register_player("p1", world=w)
    register_player("p2", world=w)
    return w


# --- 修复 1：移交刷新输入 ---------------------------------------------------


class TestRefreshInputOnHandoff:
    def test_switch_to_driver_seat_rebinds_driver_input(self, world):
        w = make_world()
        assert board("v1", "p1", 0, world=w).ok
        assert board("v1", "p2", 1, world=w).ok
        # p2 是乘客，驾驶键映射为 noop
        assert w.players["p2"].input_map["steer"] == "noop"

        result = switch_seat("v1", "p2", 0, world=w)
        assert result.ok
        # 拿到方向盘后立刻拥有驾驶映射
        assert w.players["p2"].input_map["steer"] == "vehicle_steer"
        # 原驾驶员退化为乘客，驾驶键不再能操纵方向盘
        assert w.players["p1"].input_map["steer"] == "noop"
        assert w.vehicles["v1"].owner == "p2"

    def test_explicit_refresh_entry(self, world):
        w = make_world()
        board("v1", "p1", 0, world=w)
        board("v1", "p2", 1, world=w)
        result = refresh_input_on_handoff("v1", "p1", "p2", world=w)
        assert isinstance(result, OperationResult) and result.ok
        assert w.players["p2"].input_map["throttle"] == "vehicle_throttle"
        assert w.players["p1"].input_map["throttle"] == "noop"

    def test_input_maps_are_independent_copies(self, world):
        w = make_world()
        board("v1", "p1", 0, world=w)
        board("v1", "p2", 1, world=w)
        w.players["p2"].input_map["steer"] = "hacked"
        # 重新刷新后拿到全新字典，不被外部篡改污染
        refresh_input_on_handoff("v1", "p1", "p2", world=w)
        assert w.players["p2"].input_map["steer"] == "vehicle_steer"

    def test_exit_restores_on_foot_input(self, world):
        w = make_world()
        board("v1", "p1", 0, world=w)
        assert w.players["p1"].input_map.get("look") is None
        exit_with_current_xform("v1", "p1", world=w)
        assert w.players["p1"].input_map["look"] == "on_foot_look"


# --- 修复 2：占座原子校验 ---------------------------------------------------


class TestAtomicOccupy:
    def test_two_players_cannot_take_same_seat(self, world):
        w = make_world()
        r1 = atomic_occupy("v1", "p1", 0, world=w)
        r2 = atomic_occupy("v1", "p2", 0, world=w)
        assert r1.ok and not r2.ok
        assert r2.code == "seat_occupied"
        assert w.vehicles["v1"].seats[0].occupant == "p1"

    def test_concurrent_occupy_only_one_wins(self, world):
        w = make_world()
        outcomes = []

        def race(pid):
            outcomes.append(atomic_occupy("v1", pid, 0, world=w).ok)

        threads = [threading.Thread(target=race, args=(p,))
                   for p in ("p1", "p2", "p3")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert outcomes.count(True) == 1
        assert w.vehicles["v1"].seats[0].occupant in ("p1", "p2", "p3")

    def test_stale_async_query_rejected_by_generation(self, world):
        """模拟异步查询座位实体：查询返回时座位代际已变化。"""
        w = make_world()
        first = atomic_occupy("v1", "p1", 0, world=w)
        stale_generation = first.data["seat_generation"] - 1
        # p1 离开后座位空出，但 p2 拿着旧令牌来占座必须被拒绝
        exit_with_current_xform("v1", "p1", world=w)
        stale = atomic_occupy("v1", "p2", 0, generation=stale_generation, world=w)
        assert not stale.ok and stale.code == "stale_generation"
        # 携带当前令牌的请求正常通过
        fresh = atomic_occupy("v1", "p2", 0,
                              generation=w.vehicles["v1"].seats[0].generation, world=w)
        assert fresh.ok

    def test_stale_generation_rejected_before_business_checks(self, world):
        """代际失配优先于一切业务判断（玩家已在车内也不能绕过）。"""
        w = make_world(seat_count=3)
        board("v1", "p1", 0, world=w)
        board("v1", "p2", 1, world=w)
        stale = w.vehicles["v1"].seats[2].generation - 1
        result = atomic_occupy("v1", "p1", 2, generation=stale, world=w)
        assert not result.ok and result.code == "stale_generation"
        # p1 仍在原座位，未被迟到回调改动
        assert w.vehicles["v1"].seats[0].occupant == "p1"

    def test_respawn_generation_rejects_inflight_request(self, world):
        w = make_world()
        board("v1", "p1", 0, world=w)
        old_generation = w.vehicles["v1"].generation
        destroy("v1", world=w)
        reset_on_respawn("v1", Transform(20, 0, 0), world=w)
        # 旧代际的换座请求不能作用于重生后的新车
        result = switch_seat("v1", "p1", 1, generation=old_generation, world=w)
        assert not result.ok and result.code == "stale_generation"


# --- 修复 3：掉线回收所有权 -------------------------------------------------


class TestReclaimOnDisconnect:
    def test_driver_disconnect_releases_ownership(self, world):
        w = make_world()
        board("v1", "p1", 0, world=w)
        board("v1", "p2", 1, world=w)
        result = reclaim_on_disconnect("v1", "p1", world=w)
        assert result.ok and result.data["was_driver"] is True
        assert w.vehicles["v1"].owner is None
        assert w.vehicles["v1"].driver_seat.occupant is None
        assert w.players["p1"].online is False

    def test_new_player_can_takeover_after_disconnect(self, world):
        w = make_world(seat_count=3)
        register_player("p3", world=w)
        board("v1", "p1", 0, world=w)
        reclaim_on_disconnect("v1", "p1", world=w)
        result = takeover("v1", "p3", world=w)
        assert result.ok
        assert w.vehicles["v1"].owner == "p3"
        assert w.players["p3"].input_map["steer"] == "vehicle_steer"

    def test_passenger_can_takeover_after_driver_disconnect(self, world):
        w = make_world(seat_count=3)
        board("v1", "p1", 0, world=w)
        board("v1", "p2", 2, world=w)  # p2 已是车内乘客
        reclaim_on_disconnect("v1", "p1", world=w)
        result = takeover("v1", "p2", world=w)
        assert result.ok
        assert w.vehicles["v1"].owner == "p2"
        assert w.vehicles["v1"].driver_seat.occupant == "p2"
        assert w.players["p2"].input_map["steer"] == "vehicle_steer"

    def test_takeover_blocked_while_owner_online(self, world):
        w = make_world()
        board("v1", "p1", 0, world=w)
        result = takeover("v1", "p2", world=w)
        assert not result.ok and result.code == "owner_conflict"
        assert w.vehicles["v1"].owner == "p1"

    def test_passenger_disconnect_keeps_vehicle_drivable(self, world):
        w = make_world()
        board("v1", "p1", 0, world=w)
        board("v1", "p2", 1, world=w)
        reclaim_on_disconnect("v1", "p2", world=w)
        assert w.vehicles["v1"].owner == "p1"
        assert w.vehicles["v1"].seats[1].occupant is None


# --- 修复 4：销毁先清乘客 ---------------------------------------------------


class TestClearSeatFirst:
    def test_destroy_evicts_passengers_before_releasing_components(self, world):
        w = make_world()
        board("v1", "p1", 0, world=w)
        board("v1", "p2", 1, world=w)
        result = destroy("v1", world=w)
        assert result.ok
        assert result.data["cleared"] == 2
        assert all(s.occupant is None for s in w.vehicles["v1"].seats)
        assert w.vehicles["v1"].destroyed is True
        for pid in ("p1", "p2"):
            p = w.players[pid]
            assert p.seat_vid is None and p.seat_index is None
            # 下车点来自当前变换 (10,0,5) + 偏移，绝不是原点
            assert p.last_ground_transform.x == pytest.approx(11.0)

    def test_clear_seat_first_standalone(self, world):
        w = make_world()
        board("v1", "p1", 0, world=w)
        result = clear_seat_first("v1", world=w)
        assert result.ok and result.data["count"] == 1
        assert w.vehicles["v1"].destroyed is False  # 只清人，未销毁

    def test_operations_rejected_after_destroy(self, world):
        w = make_world()
        destroy("v1", world=w)
        assert not board("v1", "p1", 0, world=w).ok
        assert not takeover("v1", "p1", world=w).ok


# --- 修复 5：重生重置占座 ---------------------------------------------------


class TestResetOnRespawn:
    def test_respawn_does_not_inherit_occupancy(self, world):
        w = make_world()
        board("v1", "p1", 0, world=w)
        board("v1", "p2", 1, world=w)
        old_seat_ids = {id(s) for s in w.vehicles["v1"].seats}

        result = reset_on_respawn("v1", Transform(30, 0, 8), world=w)
        assert result.ok
        vehicle = w.vehicles["v1"]
        assert all(s.occupant is None for s in vehicle.seats)
        assert vehicle.owner is None
        assert vehicle.transform == Transform(30, 0, 8)
        # 全新座位对象，旧占用表连引用都没留下来
        assert {id(s) for s in vehicle.seats}.isdisjoint(old_seat_ids)
        # 玩家侧的乘车关系也已解除
        assert w.players["p1"].seat_vid is None
        assert w.players["p2"].seat_vid is None

    def test_respawn_bumps_generation(self, world):
        w = make_world()
        before = w.vehicles["v1"].generation
        reset_on_respawn("v1", world=w)
        assert w.vehicles["v1"].generation == before + 1

    def test_respawn_then_board_works_fresh(self, world):
        w = make_world()
        board("v1", "p1", 0, world=w)
        reset_on_respawn("v1", world=w)
        assert board("v1", "p2", 0, world=w).ok
        assert w.vehicles["v1"].owner == "p2"


# --- 修复 6：下车用当前变换 -------------------------------------------------


class TestExitWithCurrentXform:
    def test_exit_uses_live_transform_not_cached(self, world):
        w = make_world()  # 出生变换 (10, 0, 5)
        board("v1", "p1", 0, world=w)
        # 载具行驶到新位置
        w.vehicles["v1"].transform = Transform(99, 0, -7)
        result = exit_with_current_xform("v1", "p1", world=w)
        assert result.ok
        xform = result.data["transform"]
        assert xform.x == pytest.approx(100.0)
        assert xform.z == pytest.approx(-7.0)
        assert w.players["p1"].last_ground_transform == xform

    def test_exit_never_teleports_to_origin(self, world):
        w = make_world()
        board("v1", "p1", 0, world=w)
        w.vehicles["v1"].transform = Transform(5, 5, 5)
        exit_with_current_xform("v1", "p1", world=w)
        pos = w.players["p1"].last_ground_transform
        assert (pos.x, pos.y, pos.z) != (0.0, 0.0, 0.0)

    def test_fallback_uses_ground_transform_instead_of_origin(self, world):
        w = make_world()
        board("v1", "p1", 0, world=w)
        # 模拟载具变换异常丢失为原点：兜底必须使用玩家记录的地面变换
        w.players["p1"].last_ground_transform = Transform(42, 0, 42)
        w.vehicles["v1"].transform = Transform(0, 0, 0)
        result = fallback("v1", world=w)
        assert result.ok
        assert result.data["recovered"][0]["transform"]["x"] == 42


# --- 流程与工具入口 ---------------------------------------------------------


class TestFlowAndUtilities:
    def test_swap_seats_atomically(self, world):
        w = make_world()
        board("v1", "p1", 0, world=w)
        board("v1", "p2", 1, world=w)
        result = switch_seat("v1", "p1", 1, world=w)
        assert result.ok and result.data["swapped_with"] == "p2"
        assert w.vehicles["v1"].seats[0].occupant == "p2"
        assert w.vehicles["v1"].seats[1].occupant == "p1"
        assert w.vehicles["v1"].owner == "p2"

    def test_board_auto_picks_empty_seat(self, world):
        w = make_world()
        r1 = board("v1", "p1", world=w)
        r2 = board("v1", "p2", world=w)
        assert r1.seat == 0 and r2.seat == 1
        register_player("p3", world=w)
        assert board("v1", "p3", world=w).code == "vehicle_full"

    def test_validate_detects_inconsistent_owner(self, world):
        w = make_world()
        board("v1", "p1", 0, world=w)
        assert validate_vehicle("v1", world=w).ok
        w.vehicles["v1"].owner = "ghost"
        result = validate_vehicle("v1", world=w)
        assert not result.ok and result.data["problems"]

    def test_metrics_and_audit(self, world):
        w = make_world()
        board("v1", "p1", 0, world=w)
        m = metrics(world=w)
        assert m["vehicles"] == 1 and m["counters"]["board_ok"] >= 1
        events = audit("v1", world=w)
        assert isinstance(events, list) and events
        assert all(e["vid"] == "v1" for e in events)

    def test_determinism_state_digest_stable(self, world):
        w = make_world()
        board("v1", "p1", 0, world=w)
        d1 = determinism("v1", world=w)
        d2 = determinism("v1", world=w)
        assert d1 == d2 and d1["digest"]["owner"] == "p1"

    def test_determinism_event_order_independent(self, world):
        events_a = [
            {"seq": 1, "type": "seat_occupied", "vid": "v", "pid": "a", "seat": 0},
            {"seq": 2, "type": "boarded", "vid": "v", "pid": "b", "seat": 1},
        ]
        events_b = list(reversed(events_a))
        assert determinism(events_a, world=world)["applied"] == \
               determinism(events_b, world=world)["applied"]

    def test_missing_vehicle_returns_failure_not_none(self, world):
        for fn in (atomic_occupy, reclaim_on_disconnect, clear_seat_first,
                   destroy, reset_on_respawn, validate_vehicle):
            result = fn("nope", world=world)
            assert result is not None and not result.ok

    def test_full_lifecycle(self, world):
        """综合演练：上车 → 移交 → 掉线 → 接管 → 下车 → 销毁 → 重生。"""
        w = make_world(seat_count=3)
        register_player("p3", world=w)
        board("v1", "p1", 0, world=w)
        board("v1", "p2", 1, world=w)

        switch_seat("v1", "p2", 0, world=w)
        assert w.vehicles["v1"].owner == "p2"
        assert w.players["p1"].input_map["steer"] == "noop"

        reclaim_on_disconnect("v1", "p2", world=w)
        assert takeover("v1", "p3", world=w).ok

        w.vehicles["v1"].transform = Transform(70, 0, 0)
        exit_result = exit_with_current_xform("v1", "p3", world=w)
        assert exit_result.data["transform"].x == pytest.approx(71.0)

        destroy("v1", world=w)
        reset_on_respawn("v1", Transform(0, 0, 0), world=w)
        assert validate_vehicle("v1", world=w).ok
        assert board("v1", "p1", 0, world=w).ok
