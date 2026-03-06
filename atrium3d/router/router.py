import numpy as np
from typing import Dict

def routing(results_code: Dict, steps_per_move: int = 15, pause_frames: int = 5) -> Dict:
    """
    为原子生成连续的 3D 穿梭轨迹。
    steps_per_move: 每次移动切分的插值帧数（控制飞行速度）。
    pause_frames: 到达计算位后，悬停做门的停顿帧数（模拟打激光的时间）。
    """
    stage_positions = results_code.get("stage_positions", [])
    n_qubits = results_code.get("n_qubits", 0)

    if not stage_positions or n_qubits == 0:
        return results_code

    routing_frames = []

    # 遍历所有 stage 的状态转换
    for i in range(len(stage_positions) - 1):
        pos_start = stage_positions[i]
        pos_end = stage_positions[i + 1]

        # 1. 飞行段 (Flying Frames)
        for step in range(steps_per_move):
            alpha = step / float(steps_per_move)
            current_frame = {}
            for q in range(n_qubits):
                q_str = str(q)
                start_xyz = np.array(pos_start[q_str])
                end_xyz = np.array(pos_end[q_str])
                
                # 直线插值 (Linear Interpolation)
                # 进阶版本可在此处替换为 3D A* 避碰路径
                curr_xyz = start_xyz * (1 - alpha) + end_xyz * alpha
                current_frame[q_str] = curr_xyz.tolist()
            
            routing_frames.append(current_frame)
            
        # 2. 悬停做门段 (Hovering/Gate Frames)
        # 让原子在计算位停顿几帧，以便在动画中看清正在做量子门
        for _ in range(pause_frames):
            routing_frames.append(pos_end)

    results_code["routing_frames"] = routing_frames
    results_code["routing_steps_per_move"] = steps_per_move
    return results_code