"""载具座位与控制权移交。

本模块针对六个具体缺陷给出修复实现：

1. 移交刷新输入 —— :func:`refresh_input_on_handoff` / :func:`switch_seat`
   座位（含驾驶员控制权）移交时重新绑定本地输入映射，原乘客按驾驶键不会
   再操纵方向盘。
2. 占座原子校验 —— :func:`atomic_occupy`
   「查占用 → 写入」在同一把可重入锁内完成，并带代际（generation）令牌，
   异步查询返回后的过期请求会被拒绝，两个人不可能坐进同一个座位。
3. 掉线回收所有权 —— :func:`reclaim_on_disconnect`
   驾驶员掉线后释放座位并把载具所有权置空，载具可被新玩家接管，不会
   僵在原地。
4. 销毁先清乘客 —— :func:`clear_seat_first` / :func:`destroy`
   载具销毁时先把所有乘客安全下车（使用当前变换），再释放座位组件，
   下车回调不再持有已释放组件 / 缓存的旧变换。
5. 重生重置占座 —— :func:`reset_on_respawn`
   新载具拥有全新的占用表与代际令牌，绝不继承旧载具的占用关系；携带旧
   代际令牌的在途请求会被新载具拒绝。
6. 下车用当前变换 —— :func:`exit_with_current_xform`
   下车位置一律取自载具当前世界变换，而非建座时缓存的变换，避免把玩家
   传送到世界原点。

公共入口同时接受整数 id 或对应的领域对象，并且全部保证返回非 None 的
结构化结果。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping


# ---------------------------------------------------------------------------
# 值对象
# ---------------------------------------------------------------------------


# 输入角色
INPUT_ROLE_DRIVER = "driver"      # 驾驶员：油门 / 刹车 / 方向盘
INPUT_ROLE_PASSENGER = "passenger"  # 乘客：仅交互键，无驾驶控制
INPUT_ROLE_ONFOOT = "on_foot"     # 步行：移动 / 视角


def build_input_map(role: str) -> dict[str, str]:
    """按角色生成一份全新的输入映射（每次返回独立字典，禁止别名复用）。"""
    if role == INPUT_ROLE_DRIVER:
        return {
            "throttle": "vehicle_throttle",
            "brake": "vehicle_brake",
            "steer": "vehicle_steer",
            "interact": "vehicle_interact",
        }
    if role == INPUT_ROLE_PASSENGER:
        return {
            "throttle": "noop",
            "brake": "noop",
            "steer": "noop",
            "interact": "vehicle_interact",
        }
    # 步行
    return {
        "move_forward": "on_foot_forward",
        "move_back": "on_foot_back",
        "strafe_left": "on_foot_left",
        "strafe_right": "on_foot_right",
        "look": "on_foot_look",
        "interact": "world_interact",
    }


@dataclass(frozen=True)
class Transform:
    """世界变换（不可变）。"""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0

    def offset(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> "Transform":
        """返回平移后的新变换，原对象不变。"""
        return replace(self, x=self.x + dx, y=self.y + dy, z=self.z + dz)


ORIGIN = Transform()


@dataclass
class Seat:
    """载具座位。index 0 约定为驾驶员座。"""

    index: int
    occupant: str | None = None
    generation: int = 0  # 每次占用变更自增，用于作废过期的异步占座请求


@dataclass
class Vehicle:
    vid: str
    seats: list[Seat]
    transform: Transform = ORIGIN
    owner: str | None = None       # 控制权拥有者（驾驶员）
    destroyed: bool = False
    generation: int = 0            # 载具代际；重生后自增，旧请求全部作废
    spawn_transform: Transform = ORIGIN

    @property
    def driver_seat(self) -> Seat:
        return self.seats[0]

    def seat(self, index: int) -> Seat | None:
        if 0 <= index < len(self.seats):
            return self.seats[index]
        return None

    def occupied_indices(self) -> list[int]:
        return [s.index for s in self.seats if s.occupant is not None]

    def seat_of(self, pid: str) -> Seat | None:
        for seat in self.seats:
            if seat.occupant == pid:
                return seat
        return None


@dataclass
class Player:
    pid: str
    online: bool = True
    seat_vid: str | None = None
    seat_index: int | None = None
    input_map: dict[str, str] = field(default_factory=lambda: build_input_map(INPUT_ROLE_ONFOOT))
    # 最近一次站在地面上的变换，作为任何兜底路径的安全下车点（永不回退到原点）
    last_ground_transform: Transform = ORIGIN


@dataclass(frozen=True)
class OperationResult:
    """所有操作的统一返回结构，``ok`` 之外附带上下文，保证非 None。"""

    ok: bool
    code: str
    message: str = ""
    vid: str | None = None
    pid: str | None = None
    seat: int | None = None
    generation: int | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:  # 方便调用方直接 if result: 判断
        return self.ok


# ---------------------------------------------------------------------------
# 世界（含全局可重入锁：所有「查 → 改」都在同一把锁内原子完成）
# ---------------------------------------------------------------------------


class VehicleWorld:
    """内存中的载具 / 玩家状态与全部领域逻辑。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.vehicles: dict[str, Vehicle] = {}
        self.players: dict[str, Player] = {}
        self.events: list[dict[str, Any]] = []
        self._seq = 0
        self.counters: dict[str, int] = {}

    # ----- 基础工具 -------------------------------------------------------

    def _emit(self, etype: str, **payload: Any) -> dict[str, Any]:
        """记录审计事件。seq 单调，作为确定性排序的权威依据，不依赖墙钟。"""
        self._seq += 1
        event = {"seq": self._seq, "type": etype}
        event.update(payload)
        self.events.append(event)
        return event

    def bump(self, name: str, n: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + n

    def _vehicle(self, vid: str) -> Vehicle | None:
        return self.vehicles.get(vid)

    def _player(self, pid: str) -> Player | None:
        return self.players.get(pid)

    def _require(self, vid: str, pid: str | None = None) -> tuple[Vehicle | None, Player | None, OperationResult | None]:
        vehicle = self._vehicle(vid)
        if vehicle is None:
            return None, None, OperationResult(False, "vehicle_not_found", f"载具 {vid} 不存在", vid=vid)
        if vehicle.destroyed:
            return None, None, OperationResult(False, "vehicle_destroyed", f"载具 {vid} 已销毁", vid=vid,
                                               generation=vehicle.generation)
        player = self._player(pid) if pid is not None else None
        if pid is not None and player is None:
            return vehicle, None, OperationResult(False, "player_not_found", f"玩家 {pid} 不存在",
                                                  vid=vid, pid=pid)
        return vehicle, player, None

    def _role_for(self, vehicle: Vehicle, seat_index: int) -> str:
        return INPUT_ROLE_DRIVER if seat_index == 0 else INPUT_ROLE_PASSENGER

    def _rebind(self, player: Player, role: str) -> None:
        """刷新本地输入映射：整体替换为全新字典，杜绝旧映射残留。"""
        player.input_map = build_input_map(role)

    # ----- 修复 2：原子占座 -----------------------------------------------

    def atomic_occupy(
        self,
        vid: str,
        pid: str,
        seat_index: int,
        generation: int | None = None,
    ) -> OperationResult:
        """原子占座。

        「查询座位是否空闲 → 写入占用」在同一把锁内同步完成；异步查询座位
        实体的回调若携带过期的座位代际令牌，会在这里被拒绝，因此两个玩家
        不可能通过同一个空座校验后坐进同一个座位。
        """
        with self._lock:
            vehicle, player, err = self._require(vid, pid)
            if err is not None:
                self.bump("occupy_rejected")
                return err
            assert vehicle is not None and player is not None

            seat = vehicle.seat(seat_index)
            if seat is None:
                self.bump("occupy_rejected")
                return OperationResult(False, "seat_not_found", f"座位 {seat_index} 不存在",
                                       vid=vid, pid=pid, seat=seat_index,
                                       generation=vehicle.generation)

            if not player.online:
                self.bump("occupy_rejected")
                return OperationResult(False, "player_offline", f"玩家 {pid} 已掉线",
                                       vid=vid, pid=pid, seat=seat_index,
                                       generation=vehicle.generation)

            # 代际校验优先于一切业务判断：重生 / 座位变更后，旧异步请求的
            # 令牌必然失配，无论玩家当前状态如何都必须拒绝，防止迟到的
            # 异步回调把新状态覆盖掉。
            if generation is not None and generation != seat.generation:
                self.bump("occupy_stale_generation")
                self._emit("occupy_rejected", vid=vid, pid=pid, seat=seat_index,
                           reason="stale_generation", generation=vehicle.generation)
                return OperationResult(False, "stale_generation",
                                       "占座请求基于过期的座位状态，已拒绝",
                                       vid=vid, pid=pid, seat=seat_index,
                                       generation=vehicle.generation)

            # 已坐在本车的玩家只能换座，不能重复占座
            current = vehicle.seat_of(pid)
            if current is not None and current.index != seat_index:
                self.bump("occupy_rejected")
                return OperationResult(False, "already_seated",
                                       f"玩家已坐在座位 {current.index}",
                                       vid=vid, pid=pid, seat=current.index,
                                       generation=vehicle.generation)

            # 临界区：查 + 写不可分割
            if seat.occupant is not None:
                self.bump("occupy_conflict")
                self._emit("occupy_rejected", vid=vid, pid=pid, seat=seat_index,
                           reason="occupied", generation=vehicle.generation)
                return OperationResult(False, "seat_occupied",
                                       f"座位 {seat_index} 已被 {seat.occupant} 占用",
                                       vid=vid, pid=pid, seat=seat_index,
                                       generation=vehicle.generation)

            seat.occupant = pid
            seat.generation += 1
            player.seat_vid = vid
            player.seat_index = seat_index

            if seat_index == 0:
                # 驾驶员座即控制权
                vehicle.owner = pid
            self._rebind(player, self._role_for(vehicle, seat_index))
            self.bump("occupy_ok")
            self._emit("seat_occupied", vid=vid, pid=pid, seat=seat_index,
                       generation=vehicle.generation, seat_generation=seat.generation)
            return OperationResult(True, "occupied", vid=vid, pid=pid, seat=seat_index,
                                   generation=vehicle.generation,
                                   data={"seat_generation": seat.generation})

    # ----- 上车 -----------------------------------------------------------

    def board(self, vid: str, pid: str, seat_index: int | None = None) -> OperationResult:
        """玩家上车；未指定座位时优先驾驶员座，其后取第一个空座。"""
        with self._lock:
            vehicle, player, err = self._require(vid, pid)
            if err is not None:
                return err
            assert vehicle is not None and player is not None

            if player.seat_vid is not None:
                return OperationResult(False, "already_in_vehicle",
                                       f"玩家已在载具 {player.seat_vid} 上",
                                       vid=vid, pid=pid, generation=vehicle.generation)

            if seat_index is None:
                target = next((s for s in vehicle.seats if s.occupant is None), None)
                if target is None:
                    self.bump("board_rejected")
                    return OperationResult(False, "vehicle_full", "载具已满员",
                                           vid=vid, pid=pid, generation=vehicle.generation)
                seat_index = target.index

            result = self.atomic_occupy(vid, pid, seat_index)
            if result.ok:
                self.bump("board_ok")
                self._emit("boarded", vid=vid, pid=pid, seat=seat_index,
                           generation=vehicle.generation)
            return result

    # ----- 修复 1：座位切换 / 控制权移交时刷新输入 ------------------------

    def switch_seat(
        self,
        vid: str,
        pid: str,
        target_index: int,
        generation: int | None = None,
    ) -> OperationResult:
        """车内换座；若目标座有人则执行原子双向交换。

        控制权随驾驶员座（0 号座）移交，所有受影响玩家的本地输入映射都在
        同一临界区内重新绑定 —— 拿到方向盘的乘客立刻获得驾驶映射，交出方
        向盘的前驾驶员立刻退化为乘客映射，驾驶键不再串到方向盘上。
        """
        with self._lock:
            vehicle, player, err = self._require(vid, pid)
            if err is not None:
                return err
            assert vehicle is not None and player is not None

            if generation is not None and generation != vehicle.generation:
                self.bump("switch_stale_generation")
                return OperationResult(False, "stale_generation",
                                       "换座请求基于旧载具代际，已拒绝",
                                       vid=vid, pid=pid, generation=vehicle.generation)

            current = vehicle.seat_of(pid)
            if current is None:
                return OperationResult(False, "not_in_vehicle", "玩家不在该载具上",
                                       vid=vid, pid=pid, generation=vehicle.generation)
            target = vehicle.seat(target_index)
            if target is None:
                return OperationResult(False, "seat_not_found", f"座位 {target_index} 不存在",
                                       vid=vid, pid=pid, generation=vehicle.generation)
            if target.index == current.index:
                return OperationResult(True, "no_op", "已在目标座位",
                                       vid=vid, pid=pid, seat=current.index,
                                       generation=vehicle.generation)

            other_pid = target.occupant
            if other_pid is not None:
                # 原子双向交换：两个座位的占用一起改，中间不存在空窗
                other = self._player(other_pid)
                target.occupant = pid
                current.occupant = other_pid
                target.generation += 1
                current.generation += 1
                player.seat_index = target.index
                if other is not None:
                    other.seat_index = current.index
                swapped = [pid, other_pid]
            else:
                target.occupant = pid
                current.occupant = None
                target.generation += 1
                current.generation += 1
                player.seat_index = target.index
                swapped = [pid]

            # 控制权始终跟随驾驶员座，并刷新每个受影响玩家的输入映射
            driver = vehicle.driver_seat.occupant
            vehicle.owner = driver
            for affected in swapped:
                affected_player = self._player(affected)
                if affected_player is None:
                    continue
                seat = vehicle.seat_of(affected)
                role = self._role_for(vehicle, seat.index) if seat else INPUT_ROLE_ONFOOT
                self._rebind(affected_player, role)

            self.bump("switch_ok")
            self._emit("seat_switched", vid=vid, pid=pid,
                       from_seat=current.index, to_seat=target.index,
                       swapped_with=other_pid, generation=vehicle.generation)
            return OperationResult(True, "switched", vid=vid, pid=pid,
                                   seat=target_index, generation=vehicle.generation,
                                   data={"swapped_with": other_pid, "owner": driver})

    def refresh_input_on_handoff(
        self,
        vid: str,
        driver_pid: str | None = None,
        passenger_pid: str | None = None,
    ) -> OperationResult:
        """控制权移交后显式刷新双方输入映射（修复 1 的独立入口）。

        - 新驾驶员（passenger_pid，或当前实际占用驾驶员座的玩家）绑定驾驶映射；
        - 交出控制权的原驾驶员（driver_pid）退化为乘客 / 步行映射；
        - 任一方缺失时只刷新存在的一方，永不返回 None。
        """
        with self._lock:
            vehicle = self._vehicle(vid)
            if vehicle is None:
                return OperationResult(False, "vehicle_not_found", vid=vid)
            touched: list[str] = []

            new_driver_pid = passenger_pid or vehicle.driver_seat.occupant
            new_driver = self._player(new_driver_pid) if new_driver_pid else None
            if new_driver is not None:
                self._rebind(new_driver, INPUT_ROLE_DRIVER)
                touched.append(new_driver_pid)

            old_driver = self._player(driver_pid) if driver_pid else None
            if old_driver is not None and old_driver is not new_driver:
                # 移交后原驾驶员不再拥有控制权：仍在车上（哪怕物理座位尚未
                # 切换）给乘客映射，已离车则回退步行映射 —— 绝不保留驾驶映射。
                role = INPUT_ROLE_PASSENGER if vehicle.seat_of(driver_pid) else INPUT_ROLE_ONFOOT
                self._rebind(old_driver, role)
                touched.append(driver_pid)

            self.bump("input_refreshed")
            self._emit("input_refreshed", vid=vid, touched=touched,
                       generation=vehicle.generation)
            return OperationResult(True, "input_refreshed", vid=vid,
                                   generation=vehicle.generation,
                                   data={"touched": touched})

    # ----- 修复 3：掉线回收所有权 -----------------------------------------

    def reclaim_on_disconnect(self, vid: str, pid: str) -> OperationResult:
        """玩家掉线：释放座位、回收载具所有权并刷新输入。

        驾驶员掉线后 owner 置空，载具不再被一个永远不会回来的持有者锁死，
        其他在线玩家可以通过 :meth:`takeover` 接管。
        """
        with self._lock:
            vehicle = self._vehicle(vid)
            if vehicle is None:
                return OperationResult(False, "vehicle_not_found", vid=vid, pid=pid)
            player = self._player(pid)
            if player is None:
                return OperationResult(False, "player_not_found", vid=vid, pid=pid)

            player.online = False
            seat = vehicle.seat_of(pid)
            was_driver = seat is not None and seat.index == 0

            if seat is not None:
                seat.occupant = None
                seat.generation += 1
                player.seat_vid = None
                player.seat_index = None
            self._rebind(player, INPUT_ROLE_ONFOOT)

            if vehicle.owner == pid:
                vehicle.owner = None  # 回收所有权，载具回到可接管状态

            self.bump("disconnect_reclaimed" if was_driver else "disconnect_passenger")
            self._emit("disconnect_reclaimed", vid=vid, pid=pid,
                       seat=seat.index if seat else None, was_driver=was_driver,
                       generation=vehicle.generation)
            return OperationResult(True, "reclaimed", vid=vid, pid=pid,
                                   seat=seat.index if seat else None,
                                   generation=vehicle.generation,
                                   data={"was_driver": was_driver, "owner": vehicle.owner})

    # ----- 修复 6：下车一律使用当前变换 -----------------------------------

    def _exit_locked(self, vehicle: Vehicle, player: Player) -> Transform:
        """调用方须持锁。基于载具当前变换计算下车点并解除座位关系。"""
        seat = vehicle.seat_of(player.pid)
        if seat is not None:
            seat.occupant = None
            seat.generation += 1
        if vehicle.owner == player.pid:
            vehicle.owner = None
        player.seat_vid = None
        player.seat_index = None
        self._rebind(player, INPUT_ROLE_ONFOOT)

        # 关键：取载具“当前”世界变换，而不是座位/载具创建时缓存的旧变换
        exit_xform = vehicle.transform.offset(dx=1.0)
        player.last_ground_transform = exit_xform
        return exit_xform

    def exit_with_current_xform(self, vid: str, pid: str) -> OperationResult:
        """玩家下车，下车点由载具当前变换派生（修复 6）。"""
        with self._lock:
            vehicle, player, err = self._require(vid, pid)
            if err is not None:
                return err
            assert vehicle is not None and player is not None
            if vehicle.seat_of(pid) is None:
                return OperationResult(False, "not_in_vehicle", "玩家不在该载具上",
                                       vid=vid, pid=pid, generation=vehicle.generation)

            xform = self._exit_locked(vehicle, player)
            self.bump("exit_ok")
            self._emit("exited", vid=vid, pid=pid, transform=vars(xform),
                       generation=vehicle.generation)
            return OperationResult(True, "exited", vid=vid, pid=pid,
                                   generation=vehicle.generation,
                                   data={"transform": xform})

    # ----- 修复 4：销毁前先清乘客 -----------------------------------------

    def clear_seat_first(self, vid: str) -> OperationResult:
        """先把车上所有乘客按当前变换安全请下车，再释放座位组件。"""
        with self._lock:
            vehicle = self._vehicle(vid)
            if vehicle is None:
                return OperationResult(False, "vehicle_not_found", vid=vid)

            # 快照占用者，避免遍历过程中修改列表
            occupants = [s.occupant for s in vehicle.seats if s.occupant is not None]
            exits: list[dict[str, Any]] = []
            for pid in occupants:
                player = self._player(pid)
                if player is None:
                    continue
                xform = self._exit_locked(vehicle, player)
                exits.append({"pid": pid, "transform": vars(xform)})

            vehicle.owner = None
            self.bump("seat_cleared", len(occupants))
            self._emit("seats_cleared", vid=vid, occupants=occupants,
                       generation=vehicle.generation)
            return OperationResult(True, "seats_cleared", vid=vid,
                                   generation=vehicle.generation,
                                   data={"exits": exits, "count": len(occupants)})

    def destroy(self, vid: str) -> OperationResult:
        """销毁载具：顺序固定为 清乘客 → 释放座位组件 → 标记销毁。

        乘客回调在座位组件释放前就已执行完，且拿到的是当前变换，不会再
        引用已释放组件或被传送到世界原点。
        """
        with self._lock:
            vehicle = self._vehicle(vid)
            if vehicle is None:
                return OperationResult(False, "vehicle_not_found", vid=vid)
            if vehicle.destroyed:
                return OperationResult(True, "already_destroyed", vid=vid,
                                       generation=vehicle.generation)

            clear_result = self.clear_seat_first(vid)  # 第一步：先清人
            # 第二步：释放座位组件（代际自增，作废所有在途座位请求）
            for seat in vehicle.seats:
                seat.occupant = None
                seat.generation += 1
            # 第三步：标记销毁
            vehicle.destroyed = True
            vehicle.owner = None
            vehicle.generation += 1

            self.bump("destroy_ok")
            self._emit("vehicle_destroyed", vid=vid,
                       generation=vehicle.generation,
                       cleared=clear_result.data.get("count", 0))
            return OperationResult(True, "destroyed", vid=vid,
                                   generation=vehicle.generation,
                                   data={"cleared": clear_result.data.get("count", 0)})

    # ----- 修复 5：重生重置占座 -------------------------------------------

    def reset_on_respawn(
        self,
        vid: str,
        transform: Transform | None = None,
        seat_count: int | None = None,
    ) -> OperationResult:
        """载具重生：全新占用表 + 新代际令牌，旧载具的占用关系不继承。"""
        with self._lock:
            old = self._vehicle(vid)
            if old is None:
                return OperationResult(False, "vehicle_not_found", vid=vid)

            # 先把旧车上还挂着的玩家关系解掉（正常流程里销毁前已清，这里兜底）
            for seat in old.seats:
                if seat.occupant is not None:
                    player = self._player(seat.occupant)
                    if player is not None:
                        player.seat_vid = None
                        player.seat_index = None
                        self._rebind(player, INPUT_ROLE_ONFOOT)

            n_seats = seat_count if seat_count is not None else len(old.seats)
            spawn = transform if transform is not None else old.spawn_transform
            new_generation = old.generation + 1
            # 全新座位对象 —— 不是把旧列表清空复用，杜绝任何残留引用
            new_seats = [Seat(index=i) for i in range(n_seats)]
            old.seats = new_seats
            old.transform = spawn
            old.spawn_transform = spawn
            old.owner = None
            old.destroyed = False
            old.generation = new_generation

            self.bump("respawn_ok")
            self._emit("vehicle_respawned", vid=vid, generation=new_generation,
                       seats=n_seats, transform=vars(spawn))
            return OperationResult(True, "respawned", vid=vid,
                                   generation=new_generation,
                                   data={"seats": n_seats, "transform": spawn})

    # ----- 接管 -----------------------------------------------------------

    def takeover(self, vid: str, pid: str) -> OperationResult:
        """在线玩家接管一辆无人持有（或持有者已掉线）的载具。"""
        with self._lock:
            vehicle, player, err = self._require(vid, pid)
            if err is not None:
                return err
            assert vehicle is not None and player is not None

            if not player.online:
                return OperationResult(False, "player_offline", vid=vid, pid=pid,
                                       generation=vehicle.generation)

            driver_pid = vehicle.driver_seat.occupant
            if driver_pid is not None:
                driver = self._player(driver_pid)
                if driver is not None and driver.online:
                    return OperationResult(False, "owner_conflict",
                                           f"载具仍由在线玩家 {driver_pid} 持有",
                                           vid=vid, pid=pid,
                                           generation=vehicle.generation)
                # 持有者掉线：先回收再接管（修复 3 的闭环）
                self.reclaim_on_disconnect(vid, driver_pid)

            # 车内乘客接管 = 换入驾驶座（走换座流程，支持原子交换）；
            # 车外玩家接管 = 原子占用驾驶座。
            current = vehicle.seat_of(pid)
            if current is not None:
                result = self.switch_seat(vid, pid, 0)
            else:
                result = self.atomic_occupy(vid, pid, 0)
            if result.ok:
                self.bump("takeover_ok")
                self._emit("takeover", vid=vid, pid=pid,
                           generation=vehicle.generation)
            return result

    # ----- 校验 / 指标 / 审计 / 兜底 / 确定性 -----------------------------

    def validate_vehicle(self, vid: str) -> OperationResult:
        """校验载具状态不变量：占用唯一、owner 与驾驶员座一致。"""
        with self._lock:
            vehicle = self._vehicle(vid)
            if vehicle is None:
                return OperationResult(False, "vehicle_not_found", vid=vid)

            seen: dict[str, int] = {}
            problems: list[str] = []
            for seat in vehicle.seats:
                if seat.occupant is not None:
                    if seat.occupant in seen:
                        problems.append(
                            f"玩家 {seat.occupant} 同时占用座位 {seen[seat.occupant]} 与 {seat.index}"
                        )
                    seen[seat.occupant] = seat.index
            driver = vehicle.driver_seat.occupant
            if vehicle.owner != driver:
                problems.append(f"owner={vehicle.owner!r} 与驾驶员座占用者 {driver!r} 不一致")
            for pid, index in seen.items():
                player = self._player(pid)
                if player is None:
                    problems.append(f"占用者 {pid} 不存在")
                elif player.seat_vid != vid or player.seat_index != index:
                    problems.append(f"玩家 {pid} 的座位回指关系损坏")

            ok = not problems
            self.bump("validate_ok" if ok else "validate_failed")
            return OperationResult(ok, "valid" if ok else "invalid",
                                   vid=vid, generation=vehicle.generation,
                                   data={"problems": problems,
                                         "occupied": vehicle.occupied_indices()})

    def metrics(self, _arg: Any = None) -> dict[str, Any]:
        """返回指标快照（计数器 + 当前规模），永远非 None。"""
        with self._lock:
            return {
                "counters": dict(self.counters),
                "vehicles": len(self.vehicles),
                "players": len(self.players),
                "events": len(self.events),
                "occupied_seats": sum(len(v.occupied_indices()) for v in self.vehicles.values()),
            }

    def audit(self, vid: str | None = None) -> list[dict[str, Any]]:
        """返回审计事件副本，可按载具过滤；永远返回列表（非 None）。"""
        with self._lock:
            if vid is None:
                return [dict(e) for e in self.events]
            return [dict(e) for e in self.events if e.get("vid") == vid]

    def fallback(self, vid: str | None = None) -> OperationResult:
        """兜底恢复：把载具上所有玩家安全移到各自最近地面变换并重置输入。

        即使载具状态异常，也不允许把人传送到世界原点 —— 优先使用载具当前
        变换，其次使用玩家记录的 last_ground_transform。
        """
        with self._lock:
            vehicle = self._vehicle(vid) if vid is not None else None
            if vehicle is None:
                self._emit("fallback_skipped", vid=vid, reason="vehicle_not_found")
                return OperationResult(True, "fallback_no_op", vid=vid,
                                       data={"recovered": []})

            recovered: list[dict[str, Any]] = []
            for seat in list(vehicle.seats):
                pid = seat.occupant
                if pid is None:
                    continue
                player = self._player(pid)
                if player is not None:
                    safe = vehicle.transform if vehicle.transform != ORIGIN else player.last_ground_transform
                    player.last_ground_transform = safe
                    self._rebind(player, INPUT_ROLE_ONFOOT)
                    player.seat_vid = None
                    player.seat_index = None
                    recovered.append({"pid": pid, "transform": vars(safe)})
                seat.occupant = None
                seat.generation += 1
            vehicle.owner = None
            self.bump("fallback_ok")
            self._emit("fallback_recovered", vid=vid, recovered=recovered,
                       generation=vehicle.generation)
            return OperationResult(True, "fallback_recovered", vid=vid,
                                   generation=vehicle.generation,
                                   data={"recovered": recovered})

    def determinism(self, payload: Any) -> dict[str, Any]:
        """确定性工具：对同类输入给出与处理顺序无关的稳定摘要。

        - 传入事件列表：按 (type, vid, pid, seat) 规范化排序后逐条应用，
          乱序输入产生相同结果；
        - 传入载具 id：返回该载具状态的规范化摘要（排序后的占用表等），
          可用于重放比对。
        """
        with self._lock:
            if isinstance(payload, Iterable) and not isinstance(payload, (str, bytes, Mapping)):
                events = [dict(e) for e in payload]
                events.sort(key=lambda e: (
                    str(e.get("type", "")),
                    str(e.get("vid", "")),
                    str(e.get("pid", "")),
                    e.get("seat") if e.get("seat") is not None else -1,
                    e.get("seq", 0),
                ))
                applied = [self._canonical_event(e) for e in events]
                return {"mode": "events", "count": len(events), "applied": applied}

            vid = str(payload)
            vehicle = self._vehicle(vid)
            if vehicle is None:
                return {"mode": "state", "vid": vid, "exists": False, "digest": None}
            return {"mode": "state", "vid": vid, "exists": True,
                    "digest": self.state_digest(vid)}

    def _canonical_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        """事件规范化：固定键顺序、剔除易变的墙钟字段。"""
        keep = ("seq", "type", "vid", "pid", "seat", "from_seat", "to_seat",
                "generation", "reason")
        return {k: event[k] for k in keep if k in event}

    def state_digest(self, vid: str) -> dict[str, Any]:
        """载具状态的确定性摘要（集合类字段全部排序）。"""
        vehicle = self.vehicles[vid]
        return {
            "vid": vehicle.vid,
            "generation": vehicle.generation,
            "destroyed": vehicle.destroyed,
            "owner": vehicle.owner,
            "transform": vars(vehicle.transform),
            "occupancy": sorted(
                (s.index, s.occupant, s.generation)
                for s in vehicle.seats
                if s.occupant is not None
            ),
            "seat_count": len(vehicle.seats),
        }


# ---------------------------------------------------------------------------
# 模块级默认世界与公共入口
# ---------------------------------------------------------------------------

_default_world = VehicleWorld()


def reset_world() -> VehicleWorld:
    """重置全局默认世界（主要供测试使用），返回新的世界实例。"""
    global _default_world
    _default_world = VehicleWorld()
    return _default_world


def get_world() -> VehicleWorld:
    return _default_world


def _id_of(value: Any) -> str:
    """公共入口同时接受整数 / 字符串 id 或领域对象。"""
    if value is None:
        return ""
    if hasattr(value, "vid"):
        return value.vid
    if hasattr(value, "pid"):
        return value.pid
    return str(value)


def spawn_vehicle(vid: Any, seat_count: int = 2,
                  transform: Transform = ORIGIN,
                  world: VehicleWorld | None = None) -> Vehicle:
    """在世界中创建一辆载具。"""
    world = world or _default_world
    sid = _id_of(vid)
    with world._lock:
        vehicle = Vehicle(
            vid=sid,
            seats=[Seat(index=i) for i in range(seat_count)],
            transform=transform,
            spawn_transform=transform,
        )
        world.vehicles[sid] = vehicle
        world._emit("vehicle_spawned", vid=sid, seats=seat_count,
                    transform=vars(transform))
        return vehicle


def register_player(pid: Any, world: VehicleWorld | None = None) -> Player:
    """在世界中注册一名玩家。"""
    world = world or _default_world
    sid = _id_of(pid)
    with world._lock:
        player = Player(pid=sid)
        world.players[sid] = player
        return player


# --- 六个修复对应的公共入口（签名宽松，单 id 调用也合法） -----------------


def refresh_input_on_handoff(vid: Any, driver_pid: Any = None,
                             passenger_pid: Any = None,
                             world: VehicleWorld | None = None) -> OperationResult:
    """修复 1：控制权移交后刷新本地输入映射。"""
    world = world or _default_world
    return world.refresh_input_on_handoff(
        _id_of(vid),
        _id_of(driver_pid) if driver_pid is not None else None,
        _id_of(passenger_pid) if passenger_pid is not None else None,
    )


def atomic_occupy(vid: Any, pid: Any = None, seat_index: int = 0,
                  generation: int | None = None,
                  world: VehicleWorld | None = None) -> OperationResult:
    """修复 2：锁内原子占座 + 代际令牌拒绝过期异步请求。"""
    world = world or _default_world
    return world.atomic_occupy(_id_of(vid),
                               _id_of(pid) if pid is not None else "",
                               seat_index, generation)


def reclaim_on_disconnect(vid: Any, pid: Any = None,
                          world: VehicleWorld | None = None) -> OperationResult:
    """修复 3：掉线释放座位并回收载具所有权。"""
    world = world or _default_world
    return world.reclaim_on_disconnect(_id_of(vid), _id_of(pid) if pid is not None else "")


def clear_seat_first(vid: Any, world: VehicleWorld | None = None) -> OperationResult:
    """修复 4：销毁 / 释放组件前先把乘客安全请下车。"""
    world = world or _default_world
    return world.clear_seat_first(_id_of(vid))


def reset_on_respawn(vid: Any, transform: Transform | None = None,
                     seat_count: int | None = None,
                     world: VehicleWorld | None = None) -> OperationResult:
    """修复 5：重生载具获得全新占用表与代际令牌。"""
    world = world or _default_world
    return world.reset_on_respawn(_id_of(vid), transform, seat_count)


def exit_with_current_xform(vid: Any, pid: Any = None,
                            world: VehicleWorld | None = None) -> OperationResult:
    """修复 6：下车位置取自载具当前变换而非缓存旧变换。"""
    world = world or _default_world
    return world.exit_with_current_xform(_id_of(vid),
                                         _id_of(pid) if pid is not None else "")


# --- 流程入口 ----------------------------------------------------------------


def board(vid: Any, pid: Any = None, seat_index: int | None = None,
          world: VehicleWorld | None = None) -> OperationResult:
    """上车流程（内部走原子占座）。"""
    world = world or _default_world
    return world.board(_id_of(vid), _id_of(pid) if pid is not None else "", seat_index)


def switch_seat(vid: Any, pid: Any = None, target_index: int = 1,
                generation: int | None = None,
                world: VehicleWorld | None = None) -> OperationResult:
    """换座 / 座位交换，控制权随驾驶员座移交并刷新双方输入。"""
    world = world or _default_world
    return world.switch_seat(_id_of(vid),
                             _id_of(pid) if pid is not None else "",
                             target_index, generation)


def takeover(vid: Any, pid: Any = None,
             world: VehicleWorld | None = None) -> OperationResult:
    """接管无主（或原持有者掉线）的载具。"""
    world = world or _default_world
    return world.takeover(_id_of(vid), _id_of(pid) if pid is not None else "")


def destroy(vid: Any, world: VehicleWorld | None = None) -> OperationResult:
    """销毁载具：先清乘客，再释放座位组件，最后标记销毁。"""
    world = world or _default_world
    return world.destroy(_id_of(vid))


def validate_vehicle(vid: Any, world: VehicleWorld | None = None) -> OperationResult:
    """校验载具状态不变量。"""
    world = world or _default_world
    return world.validate_vehicle(_id_of(vid))


def metrics(_arg: Any = None, world: VehicleWorld | None = None) -> dict[str, Any]:
    """指标快照。"""
    world = world or _default_world
    return world.metrics(_arg)


def audit(vid: Any = None, world: VehicleWorld | None = None) -> list[dict[str, Any]]:
    """审计事件列表（按载具过滤可选），永远非 None。"""
    world = world or _default_world
    if vid is None:
        return world.audit(None)
    return world.audit(_id_of(vid))


def fallback(vid: Any = None, world: VehicleWorld | None = None) -> OperationResult:
    """兜底恢复：安全移出全部乘客，杜绝原点传送。"""
    world = world or _default_world
    return world.fallback(_id_of(vid) if vid is not None else None)


def determinism(payload: Any = None,
                world: VehicleWorld | None = None) -> dict[str, Any]:
    """确定性摘要 / 规范化事件重放。"""
    world = world or _default_world
    return world.determinism(payload)
