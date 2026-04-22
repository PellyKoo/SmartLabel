import csv
import json
import os
import shutil
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from src.engine.base import VideoClipResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

# HTML 报告模板
_QC_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>SmartLabel 质检报告</title>
<style>
body {{ font-family: -apple-system, "Microsoft YaHei", sans-serif; margin: 20px; background: #f5f5f5; }}
.container {{ max-width: 1200px; margin: 0 auto; }}
h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
h2 {{ color: #555; margin-top: 30px; }}
.stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
.stat-card {{ background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }}
.stat-value {{ font-size: 2em; font-weight: bold; color: #4CAF50; }}
.stat-label {{ color: #888; margin-top: 5px; }}
.stat-card.warning .stat-value {{ color: #FF9800; }}
.stat-card.danger .stat-value {{ color: #f44336; }}
table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 15px 0; }}
th, td {{ padding: 10px 15px; text-align: left; border-bottom: 1px solid #eee; }}
th {{ background: #4CAF50; color: white; }}
tr:hover {{ background: #f9f9f9; }}
.review-card {{ background: white; border-radius: 8px; padding: 15px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: flex; gap: 15px; align-items: center; }}
.review-card img {{ width: 120px; height: 90px; object-fit: cover; border-radius: 4px; }}
.review-info {{ flex: 1; }}
.label-compare {{ display: flex; gap: 10px; align-items: center; }}
.label {{ padding: 3px 8px; border-radius: 4px; font-size: 0.9em; }}
.label-human {{ background: #E3F2FD; color: #1565C0; }}
.label-engine {{ background: #FFF3E0; color: #E65100; }}
.arrow {{ color: #999; }}
.reason {{ color: #666; font-size: 0.85em; margin-top: 5px; }}
.escalated {{ color: #9C27B0; font-size: 0.8em; }}
.bar-chart {{ display: flex; align-items: flex-end; gap: 8px; height: 200px; margin: 20px 0; padding: 10px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
.bar-item {{ display: flex; flex-direction: column; align-items: center; flex: 1; }}
.bar {{ width: 100%; background: #4CAF50; border-radius: 4px 4px 0 0; min-height: 2px; transition: height 0.3s; }}
.bar-label {{ font-size: 0.75em; color: #666; margin-top: 5px; writing-mode: vertical-lr; text-orientation: mixed; max-height: 80px; overflow: hidden; }}
.bar-value {{ font-size: 0.75em; color: #333; font-weight: bold; }}
</style>
</head>
<body>
<div class="container">
<h1>SmartLabel 质检报告</h1>

<div class="stats-grid">
    <div class="stat-card">
        <div class="stat-value">{total_checked}</div>
        <div class="stat-label">总检查数</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{pass_count}</div>
        <div class="stat-label">通过</div>
    </div>
    <div class="stat-card {review_card_class}">
        <div class="stat-value">{review_count}</div>
        <div class="stat-label">待复核</div>
    </div>
    <div class="stat-card {review_card_class}">
        <div class="stat-value">{review_ratio_pct}%</div>
        <div class="stat-label">异议率</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{escalated_count}</div>
        <div class="stat-label">VLM 升级复核</div>
    </div>
</div>

<h2>各类别异议率</h2>
<div class="bar-chart">
{category_bars}
</div>

<h2>各类别详细统计</h2>
<table>
<tr><th>类别</th><th>总数</th><th>待复核</th><th>异议率</th></tr>
{category_rows}
</table>

<h2>典型错误模式 (Top 10)</h2>
<table>
<tr><th>错误模式</th><th>出现次数</th></tr>
{error_pattern_rows}
</table>

<h2>待复核样本 (前 {max_review_display} 条)</h2>
{review_cards}

</div>
</body>
</html>
"""


class ReportGenerator:
    """报告生成器"""

    def generate_qc_html_report(self, qc_results: dict, output_path: str):
        """
        生成 HTML 质检报告。

        内容：总体统计、各类别异议率、待复核样本、典型错误模式、VLM 升级统计。
        """
        total = qc_results.get("total_checked", 0)
        pass_count = qc_results.get("pass_count", 0)
        review_count = qc_results.get("review_count", 0)
        review_ratio = qc_results.get("review_ratio", 0)
        escalated_count = qc_results.get("escalated_count", 0)
        category_stats = qc_results.get("category_review_stats", {})
        review_samples = qc_results.get("review_samples", [])
        error_cases = qc_results.get("error_cases", [])

        # 异议率颜色
        review_card_class = "danger" if review_ratio > 0.1 else ("warning" if review_ratio > 0.05 else "")

        # 类别柱状图
        max_ratio = max((s["ratio"] for s in category_stats.values()), default=0.01) or 0.01
        category_bars = ""
        for lbl, stats in sorted(category_stats.items()):
            bar_height = int(stats["ratio"] / max_ratio * 150) + 2
            category_bars += (
                f'<div class="bar-item">'
                f'<div class="bar-value">{stats["ratio"]*100:.1f}%</div>'
                f'<div class="bar" style="height:{bar_height}px"></div>'
                f'<div class="bar-label">{lbl}</div>'
                f'</div>\n'
            )

        # 类别表格
        category_rows = ""
        for lbl, stats in sorted(category_stats.items()):
            category_rows += (
                f'<tr><td>{lbl}</td><td>{stats["total"]}</td>'
                f'<td>{stats["review"]}</td><td>{stats["ratio"]*100:.2f}%</td></tr>\n'
            )

        # 错误模式
        error_pattern_rows = ""
        for i, ec in enumerate(error_cases[:10]):
            count = ec["count"]
            if ec["samples"]:
                sample = ec["samples"][0]
                pattern = f'{sample["human_label"]} → {sample["engine_label"]}'
            else:
                pattern = "未知"
            error_pattern_rows += f'<tr><td>{pattern}</td><td>{count}</td></tr>\n'

        # 待复核样本卡片
        max_display = 50
        review_cards = ""
        for r in review_samples[:max_display]:
            escalated_tag = '<span class="escalated">[VLM 复核]</span>' if r.get("escalated") else ""
            reason = f'<div class="reason">理由: {r.get("vlm_reason", "")}</div>' if r.get("vlm_reason") else ""
            conf = f' ({r["confidence"]:.2f})' if r.get("confidence") is not None else ""

            review_cards += f"""<div class="review-card">
    <img src="file://{r['image_path']}" onerror="this.style.display='none'" alt="">
    <div class="review-info">
        <div class="label-compare">
            <span class="label label-human">人工: {r['human_label']}</span>
            <span class="arrow">→</span>
            <span class="label label-engine">引擎: {r['engine_label']}{conf}</span>
            {escalated_tag}
        </div>
        {reason}
        <div style="color:#999;font-size:0.8em">{os.path.basename(r['image_path'])}</div>
    </div>
</div>\n"""

        html = _QC_HTML_TEMPLATE.format(
            total_checked=total,
            pass_count=pass_count,
            review_count=review_count,
            review_ratio_pct=f"{review_ratio * 100:.2f}",
            review_card_class=review_card_class,
            escalated_count=escalated_count,
            category_bars=category_bars,
            category_rows=category_rows,
            error_pattern_rows=error_pattern_rows,
            review_cards=review_cards,
            max_review_display=max_display,
        )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"HTML 质检报告已生成: {output_path}")

    def generate_qc_csv_report(self, qc_results: dict, output_path: str):
        """CSV 质检报告：每行一条样本"""
        all_results = qc_results.get("all_results", [])
        if not all_results:
            all_results = qc_results.get("review_samples", [])

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "image_path", "human_label", "engine_label",
                "confidence", "is_consistent", "escalated", "vlm_reason"
            ])
            for r in all_results:
                writer.writerow([
                    r.get("image_path", ""),
                    r.get("human_label", ""),
                    r.get("engine_label", ""),
                    r.get("confidence", ""),
                    r.get("is_consistent", ""),
                    r.get("escalated", False),
                    r.get("vlm_reason", ""),
                ])

        logger.info(f"CSV 质检报告已生成: {output_path}")

    def export_error_cases(self, qc_results: dict, output_dir: str,
                            num_workers: int = 4):
        """
        导出错误 case。

        输出结构：
            output_dir/
            ├── yawning_as_normal/
            │   ├── img001.jpg
            │   ├── img001.json
            │   └── ...
            ├── smoking_as_phone_call/
            │   └── ...
        """
        review_samples = qc_results.get("review_samples", [])
        if not review_samples:
            logger.info("无待复核样本，跳过错误case导出")
            return

        os.makedirs(output_dir, exist_ok=True)

        def _export_one(sample: dict):
            human = sample.get("human_label", "unknown")
            engine = sample.get("engine_label", "unknown")
            folder_name = f"{human}_as_{engine}"
            folder_path = os.path.join(output_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            img_path = sample.get("image_path", "")
            if not os.path.exists(img_path):
                return

            filename = os.path.basename(img_path)
            dst_img = os.path.join(folder_path, filename)
            if not os.path.exists(dst_img):
                shutil.copy2(img_path, dst_img)

            # 写 metadata JSON
            stem = os.path.splitext(filename)[0]
            meta_path = os.path.join(folder_path, f"{stem}.json")
            meta = {
                "human_label": human,
                "engine_label": engine,
                "confidence": sample.get("confidence"),
                "escalated": sample.get("escalated", False),
                "vlm_reason": sample.get("vlm_reason", ""),
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            executor.map(_export_one, review_samples)

        logger.info(f"错误case已导出: {output_dir}, 共 {len(review_samples)} 条")

    def generate_video_report(self, video_result: VideoClipResult, output_path: str):
        """视频分类报告：时间轴可视化 + 各类时长统计"""
        clips = video_result.clips
        stats = video_result.statistics
        total_duration = sum(stats.values()) if stats else 0

        # 颜色映射
        colors = ["#4CAF50", "#FF9800", "#f44336", "#2196F3", "#9C27B0",
                   "#00BCD4", "#795548", "#607D8B", "#E91E63", "#3F51B5"]
        label_set = sorted(set(c["label"] for c in clips))
        color_map = {lbl: colors[i % len(colors)] for i, lbl in enumerate(label_set)}

        # 时间轴 SVG
        svg_width = 800
        svg_height = 60
        timeline_svg = f'<svg width="{svg_width}" height="{svg_height}" style="background:#eee;border-radius:4px">\n'
        if total_duration > 0:
            for clip in clips:
                x = clip["start_sec"] / total_duration * svg_width
                w = max(1, (clip.get("duration_sec", clip["end_sec"] - clip["start_sec"]) / total_duration * svg_width))
                color = color_map.get(clip["label"], "#999")
                timeline_svg += (
                    f'  <rect x="{x:.1f}" y="0" width="{w:.1f}" height="{svg_height}" '
                    f'fill="{color}" opacity="0.8">'
                    f'<title>{clip["label"]} ({clip["start_sec"]:.1f}s - {clip["end_sec"]:.1f}s)</title></rect>\n'
                )
        timeline_svg += '</svg>'

        # 图例
        legend = ' '.join(
            f'<span style="display:inline-block;width:14px;height:14px;background:{color_map[l]}"></span> {l}'
            for l in label_set
        )

        # 片段表格
        clip_rows = ""
        for c in clips:
            dur = c.get("duration_sec", c["end_sec"] - c["start_sec"])
            clip_rows += (
                f'<tr><td>{c["start_sec"]:.1f}s</td><td>{c["end_sec"]:.1f}s</td>'
                f'<td>{c["label"]}</td><td>{dur:.1f}s</td></tr>\n'
            )

        # 统计表格
        stats_rows = ""
        for lbl, dur in sorted(stats.items()):
            pct = dur / total_duration * 100 if total_duration > 0 else 0
            stats_rows += f'<tr><td>{lbl}</td><td>{dur:.1f}s</td><td>{pct:.1f}%</td></tr>\n'

        html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>视频分类报告</title>
<style>
body {{ font-family: -apple-system,"Microsoft YaHei",sans-serif; margin:20px; background:#f5f5f5; }}
.container {{ max-width:1000px; margin:0 auto; }}
h1 {{ color:#333; border-bottom:2px solid #2196F3; padding-bottom:10px; }}
h2 {{ color:#555; }}
table {{ width:100%; border-collapse:collapse; background:white; border-radius:8px; overflow:hidden; box-shadow:0 2px 4px rgba(0,0,0,0.1); margin:15px 0; }}
th,td {{ padding:8px 12px; text-align:left; border-bottom:1px solid #eee; }}
th {{ background:#2196F3; color:white; }}
.legend {{ margin:10px 0; display:flex; gap:15px; flex-wrap:wrap; }}
</style></head>
<body><div class="container">
<h1>视频分类报告</h1>
<p>视频: {video_result.video_path}</p>
<p>总时长: {total_duration:.1f}s | 片段数: {len(clips)}</p>

<h2>时间轴</h2>
{timeline_svg}
<div class="legend">{legend}</div>

<h2>各类时长统计</h2>
<table><tr><th>类别</th><th>时长</th><th>占比</th></tr>
{stats_rows}</table>

<h2>片段详情</h2>
<table><tr><th>开始</th><th>结束</th><th>类别</th><th>时长</th></tr>
{clip_rows}</table>

</div></body></html>"""

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"视频分类报告已生成: {output_path}")

    def generate_preannotate_summary(self, pa_results: dict, output_path: str):
        """预标注汇总报告"""
        total = pa_results.get("total", 0)
        category_counts = pa_results.get("category_counts", {})
        uncertain_count = pa_results.get("uncertain_count", 0)

        rows = ""
        for lbl, count in sorted(category_counts.items()):
            pct = count / total * 100 if total > 0 else 0
            rows += f'<tr><td>{lbl}</td><td>{count}</td><td>{pct:.1f}%</td></tr>\n'

        html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>预标注汇总</title>
<style>
body {{ font-family: -apple-system,"Microsoft YaHei",sans-serif; margin:20px; background:#f5f5f5; }}
.container {{ max-width:800px; margin:0 auto; }}
h1 {{ color:#333; border-bottom:2px solid #4CAF50; padding-bottom:10px; }}
table {{ width:100%; border-collapse:collapse; background:white; border-radius:8px; overflow:hidden; box-shadow:0 2px 4px rgba(0,0,0,0.1); margin:15px 0; }}
th,td {{ padding:10px 15px; text-align:left; border-bottom:1px solid #eee; }}
th {{ background:#4CAF50; color:white; }}
.stats {{ display:flex; gap:20px; margin:20px 0; }}
.stat {{ background:white; padding:15px 25px; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.1); }}
.stat b {{ font-size:1.5em; color:#4CAF50; }}
</style></head>
<body><div class="container">
<h1>预标注汇总报告</h1>
<div class="stats">
    <div class="stat"><b>{total}</b><br>总图片数</div>
    <div class="stat"><b>{len(category_counts)}</b><br>类别数</div>
    <div class="stat"><b>{uncertain_count}</b><br>不确定</div>
</div>
<h2>各类别统计</h2>
<table><tr><th>类别</th><th>数量</th><th>占比</th></tr>
{rows}</table>
</div></body></html>"""

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"预标注汇总报告已生成: {output_path}")
