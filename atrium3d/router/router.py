from typing import Dict, Optional

def routing(
    results_code: Dict,
    steps_per_move: Optional[int] = None,
    pause_frames: Optional[int] = None,
) -> Dict:
    """
    记录 routing 的时间离散参数（供 animator 做插值渲染）。
    本函数不再生成/保存逐帧 routing_frames，避免 results_code 过大。

    steps_per_move / pause_frames: 若为 None，则从 results_code 读取（可由 setting JSON 提供）。
    """
    if steps_per_move is None:
        steps_per_move = results_code.get("routing_steps_per_move", 15)
    if pause_frames is None:
        pause_frames = results_code.get("routing_pause_frames", 5)
    results_code["routing_steps_per_move"] = steps_per_move
    results_code["routing_pause_frames"] = pause_frames
    return results_code