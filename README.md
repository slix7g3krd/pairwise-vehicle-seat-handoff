# pairwise-vehicle-seat-handoff

## 领域

游戏、图形与多媒体

## 主题

载具座位与控制权移交

## 说明

输入刷新、原子占座与掉线回收。

初始状态下 `gameplay/vehicle_seat_handoff.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
