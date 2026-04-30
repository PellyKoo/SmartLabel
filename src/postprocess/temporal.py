from collections import Counter

from src.utils.logger import get_logger

logger = get_logger(__name__)


def temporal_smooth(labels: list[str], window: int = 3) -> list[str]:
    """
    时序中值滤波平滑。消除孤立噪声点。

    对标签序列中的每个位置，取其周围 window 大小窗口内的众数作为平滑后的值。
    窗口两端边界不做平滑，保留原值。

    示例：
        输入: [normal, normal, fatigue, normal, normal]
        输出: [normal, normal, normal, normal, normal]
        （中间的孤立 fatigue 被平滑掉）

    Args:
        labels: 时间序列标签列表
        window: 滤波窗口大小（奇数，建议 3 或 5）

    Returns:
        平滑后的标签列表
    """
    if not labels or window < 3:
        return list(labels)

    # 确保 window 为奇数
    if window % 2 == 0:
        window += 1

    half = window // 2
    smoothed = list(labels)

    for i in range(half, len(labels) - half):
        window_labels = labels[i - half: i + half + 1]
        counter = Counter(window_labels)
        smoothed[i] = counter.most_common(1)[0][0]

    logger.debug(f"时序平滑完成: {len(labels)} 个标签, 窗口={window}")
    return smoothed


def detect_segments(labels: list[str], timestamps: list[float],
                     min_duration_sec: float = 2.0) -> list[dict]:
    """
    连续相同标签合并为片段，过滤掉过短的片段。

    将时间序列中连续相同标签的区间合并为一个片段，
    时长低于 min_duration_sec 的短片段合并到前一个片段中。

    Args:
        labels: 时间序列标签
        timestamps: 对应的时间戳（秒），与 labels 等长
        min_duration_sec: 最短片段时长，低于此值的片段合并到前一个

    Returns:
        片段列表: [{"start_sec", "end_sec", "label", "duration_sec"}, ...]
    """
    if not labels or not timestamps:
        return []

    if len(labels) != len(timestamps):
        raise ValueError(
            f"labels 和 timestamps 长度不匹配: {len(labels)} vs {len(timestamps)}"
        )

    # 合并连续相同标签为原始片段
    raw_segments = []
    current_label = labels[0]
    start_idx = 0

    for i in range(1, len(labels)):
        if labels[i] != current_label:
            raw_segments.append({
                "start_sec": timestamps[start_idx],
                "end_sec": timestamps[i],
                "label": current_label,
            })
            current_label = labels[i]
            start_idx = i

    # 最后一个片段
    raw_segments.append({
        "start_sec": timestamps[start_idx],
        "end_sec": timestamps[-1],
        "label": current_label,
    })

    # 过滤过短片段：合并到前一个片段
    filtered = []
    for seg in raw_segments:
        seg["duration_sec"] = round(seg["end_sec"] - seg["start_sec"], 2)

        if seg["duration_sec"] < min_duration_sec and filtered:
            # 合并到前一个片段
            filtered[-1]["end_sec"] = seg["end_sec"]
            filtered[-1]["duration_sec"] = round(
                filtered[-1]["end_sec"] - filtered[-1]["start_sec"], 2
            )
        else:
            filtered.append(seg)

    logger.debug(
        f"片段检测完成: {len(raw_segments)} 个原始片段 -> {len(filtered)} 个过滤后片段"
    )
    return filtered


def compute_segment_statistics(clips: list[dict]) -> dict[str, float]:
    """
    计算各类别的总时长统计。

    Args:
        clips: 片段列表 [{"label": str, "duration_sec": float}, ...]

    Returns:
        {label: total_duration_sec} 映射
    """
    stats = {}
    for clip in clips:
        label = clip["label"]
        duration = clip.get("duration_sec", clip["end_sec"] - clip["start_sec"])
        stats[label] = round(stats.get(label, 0) + duration, 2)
    return stats
