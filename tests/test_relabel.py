"""
验证标签修改功能：
1. relabel_classification (io 层) - 普通文件 + 符号链接 + CSV 更新 + 冲突处理
2. ResultBrowser 编辑 UI 开关 + label_changed 信号
3. PreAnnotateTab 预览 + 标签修改集成
4. QualityCheckTab 人工标签修改集成
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PyQt5.QtCore import Qt, QCoreApplication
QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
from PyQt5.QtWidgets import QApplication

from src.io.classification_io import relabel_classification


def test_relabel_regular_file():
    print("\n" + "=" * 60)
    print("测试 1: relabel_classification - 普通文件")
    print("=" * 60)

    root = tempfile.mkdtemp(prefix="relabel_test_")
    try:
        os.makedirs(os.path.join(root, "normal"))
        os.makedirs(os.path.join(root, "fatigue"))
        src_path = os.path.join(root, "normal", "img001.jpg")
        with open(src_path, "w") as f:
            f.write("fake jpg")

        new_path = relabel_classification(
            root, "img001.jpg", "normal", "fatigue"
        )

        assert not os.path.exists(src_path), "原文件应被移走"
        assert os.path.exists(new_path), f"新文件应存在: {new_path}"
        assert new_path == os.path.join(root, "fatigue", "img001.jpg")
        print("✅ 文件正确移动 normal -> fatigue")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_relabel_collision():
    print("\n" + "=" * 60)
    print("测试 2: 冲突时抛 FileExistsError")
    print("=" * 60)

    root = tempfile.mkdtemp(prefix="relabel_collision_")
    try:
        os.makedirs(os.path.join(root, "normal"))
        os.makedirs(os.path.join(root, "fatigue"))
        for cls in ("normal", "fatigue"):
            with open(os.path.join(root, cls, "img.jpg"), "w") as f:
                f.write(cls)

        try:
            relabel_classification(root, "img.jpg", "normal", "fatigue")
        except FileExistsError as e:
            print(f"✅ 目标冲突抛 FileExistsError: {e}")
        else:
            raise AssertionError("应抛 FileExistsError")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_relabel_symlink():
    print("\n" + "=" * 60)
    print("测试 3: symlink 重建（指向原目标）")
    print("=" * 60)

    # Windows 上 symlink 可能需要管理员权限或开启开发者模式
    root = tempfile.mkdtemp(prefix="relabel_symlink_")
    try:
        real_file = os.path.join(root, "source.jpg")
        with open(real_file, "w") as f:
            f.write("real image data")

        normal_dir = os.path.join(root, "normal")
        os.makedirs(normal_dir)

        link_path = os.path.join(normal_dir, "img001.jpg")
        try:
            os.symlink(real_file, link_path)
        except OSError as e:
            print(f"⚠️  当前环境无法创建 symlink（{e}），跳过本测试")
            return

        new_path = relabel_classification(
            root, "img001.jpg", "normal", "distracted"
        )
        assert not os.path.lexists(link_path), "原链接应被删除"
        assert os.path.lexists(new_path), "新位置应有链接"
        assert os.path.islink(new_path), "应仍是 symlink"
        # 读取新链接目标，应指向原 real_file（绝对路径）
        target = os.readlink(new_path)
        assert os.path.normpath(target) == os.path.normpath(real_file), \
            f"链接目标错误: {target}"
        print("✅ symlink 正确重建，指向原文件")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_relabel_with_csv_update():
    print("\n" + "=" * 60)
    print("测试 4: CSV 同步更新")
    print("=" * 60)

    root = tempfile.mkdtemp(prefix="relabel_csv_")
    try:
        os.makedirs(os.path.join(root, "normal"))
        os.makedirs(os.path.join(root, "fatigue"))
        src = os.path.join(root, "normal", "img001.jpg")
        with open(src, "w") as f:
            f.write("x")

        csv_path = os.path.join(root, "classification_results.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("image_path,predicted_class,confidence\n")
            f.write("/some/orig/img001.jpg,normal,0.9\n")
            f.write("/some/orig/img002.jpg,fatigue,0.8\n")

        relabel_classification(
            root, "img001.jpg", "normal", "fatigue", csv_path=csv_path
        )

        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        assert "img001.jpg,fatigue,0.9" in lines[1], f"CSV 未更新: {lines[1]}"
        assert "img002.jpg,fatigue,0.8" in lines[2], "其他行应不受影响"
        print("✅ CSV 中 img001 的类别已更新为 fatigue，其他行未动")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_result_browser_edit_toggle():
    print("\n" + "=" * 60)
    print("测试 5: ResultBrowser 编辑 UI 开关 + label_changed 信号")
    print("=" * 60)

    app = QApplication.instance() or QApplication(sys.argv)
    from src.gui.widgets import ResultBrowser

    browser = ResultBrowser()
    # offscreen 下 isVisible() 不可靠；用 isHidden() 查 visible flag
    assert browser._edit_group.isHidden(), "默认应隐藏编辑区"
    assert browser._editable_categories is None
    print("✅ 默认不显示编辑 UI")

    # 先加数据
    browser.set_items([{
        "image_path": "/fake/img1.jpg",
        "title": "img1.jpg",
        "subtitle": "→ normal",
        "current_label": "normal",
        "meta": {"predicted_class": "normal"},
    }])

    # 启用编辑
    browser.set_editable_categories(["normal", "fatigue", "distracted"])
    assert not browser._edit_group.isHidden(), "启用后 visible flag 应为 True"
    assert browser._editable_categories == ["normal", "fatigue", "distracted"]
    assert browser._category_cb.count() == 3
    # 选中项的 current_label=normal 应被预选
    assert browser._category_cb.currentText() == "normal"
    print("✅ 启用编辑 UI，下拉框预选当前标签 normal")

    # 切换下拉 → 点击修改 → 应发射信号
    received = []

    def on_changed(item, new_label):
        received.append((item, new_label))

    browser.label_changed.connect(on_changed)
    browser._category_cb.setCurrentText("fatigue")
    browser._modify_btn.click()
    app.processEvents()

    assert len(received) == 1
    assert received[0][1] == "fatigue"
    print(f"✅ 点击修改发射 label_changed(item, 'fatigue')")

    # 禁用
    browser.set_editable_categories(None)
    assert browser._edit_group.isHidden()
    assert browser._editable_categories is None
    print("✅ 传 None 关闭编辑 UI")


def test_preannotate_preview_and_edit():
    print("\n" + "=" * 60)
    print("测试 6: PreAnnotateTab 预览 + 模拟标签修改")
    print("=" * 60)

    app = QApplication.instance() or QApplication(sys.argv)
    from src.gui.widgets import PreAnnotateTab

    # 准备临时数据
    data_dir = tempfile.mkdtemp(prefix="pa_preview_")
    try:
        # 造几张假图（cv2 能读 jpg 头，这里只存空文件也行，scan_images 只看后缀）
        for i in range(3):
            open(os.path.join(data_dir, f"img{i}.jpg"), "w").close()

        tab = PreAnnotateTab()
        tab._image_dir_edit.setText(data_dir)
        app.processEvents()

        # 预览应填充 3 条"待预标注"
        assert tab._browser._list.count() == 3, \
            f"预览项数错: {tab._browser._list.count()}"
        first_item = tab._browser._items[0]
        assert first_item["subtitle"] == "待预标注"
        print(f"✅ 选择图片目录后自动预览 {tab._browser._list.count()} 张")

        # 模拟完成一次分类 → 配置参数，再手动调用 relabel
        output_dir = tempfile.mkdtemp(prefix="pa_output_")
        try:
            tab._last_output_dir = output_dir
            tab._last_categories = ["normal", "fatigue"]
            os.makedirs(os.path.join(output_dir, "normal"))
            os.makedirs(os.path.join(output_dir, "fatigue"))
            with open(os.path.join(output_dir, "normal", "img0.jpg"), "w") as f:
                f.write("x")

            # 启用编辑
            tab._browser.set_editable_categories(tab._last_categories)
            tab._browser.set_items([{
                "image_path": os.path.join(data_dir, "img0.jpg"),
                "title": "img0.jpg",
                "subtitle": "→ normal",
                "current_label": "normal",
                "meta": {"predicted_class": "normal", "confidence": 0.8},
            }])

            # 直接调用 _on_label_changed 绕过确认对话框会失败（Question 是 blocking）
            # 用 monkey patch 替换 QMessageBox.question 的返回
            from PyQt5.QtWidgets import QMessageBox
            _orig_q = QMessageBox.question
            QMessageBox.question = staticmethod(lambda *a, **kw: QMessageBox.Yes)

            try:
                tab._on_label_changed(
                    tab._browser._items[0], "fatigue"
                )
            finally:
                QMessageBox.question = _orig_q

            assert not os.path.exists(os.path.join(output_dir, "normal", "img0.jpg"))
            assert os.path.exists(os.path.join(output_dir, "fatigue", "img0.jpg"))
            print("✅ 标签修改成功：文件已从 normal/ 移到 fatigue/")

            # 列表项 subtitle 应含 [已修改]
            new_item = tab._browser._items[0]
            assert "[已修改]" in new_item["subtitle"]
            assert new_item["current_label"] == "fatigue"
            print(f"✅ 列表项 subtitle 更新为: {new_item['subtitle']}")
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


def test_qc_label_edit():
    print("\n" + "=" * 60)
    print("测试 7: QualityCheckTab 人工标签修改")
    print("=" * 60)

    app = QApplication.instance() or QApplication(sys.argv)
    from src.gui.widgets import QualityCheckTab

    ann_dir = tempfile.mkdtemp(prefix="qc_ann_")
    try:
        os.makedirs(os.path.join(ann_dir, "normal"))
        os.makedirs(os.path.join(ann_dir, "fatigue"))
        with open(os.path.join(ann_dir, "normal", "img001.jpg"), "w") as f:
            f.write("x")

        tab = QualityCheckTab()
        tab._last_annotation_dir = ann_dir
        tab._last_categories = ["normal", "fatigue"]

        tab._browser.set_editable_categories(tab._last_categories)
        tab._browser.set_items([{
            "image_path": "/some/image_dir/img001.jpg",  # 源图片路径，不一定存在
            "title": "img001.jpg",
            "subtitle": "人工: normal → 引擎: fatigue",
            "current_label": "normal",
            "meta": {"human_label": "normal", "engine_label": "fatigue", "confidence": 0.6},
        }])

        from PyQt5.QtWidgets import QMessageBox
        _orig_q = QMessageBox.question
        QMessageBox.question = staticmethod(lambda *a, **kw: QMessageBox.Yes)
        try:
            tab._on_label_changed(tab._browser._items[0], "fatigue")
        finally:
            QMessageBox.question = _orig_q

        assert not os.path.exists(os.path.join(ann_dir, "normal", "img001.jpg"))
        assert os.path.exists(os.path.join(ann_dir, "fatigue", "img001.jpg"))
        print("✅ 标注目录下文件已从 normal/ 移到 fatigue/")

        new_item = tab._browser._items[0]
        assert new_item["current_label"] == "fatigue"
        # 修正后与引擎一致
        assert "✓一致" in new_item["subtitle"]
        print(f"✅ subtitle 更新为: {new_item['subtitle']}")
    finally:
        shutil.rmtree(ann_dir, ignore_errors=True)


def main():
    test_relabel_regular_file()
    test_relabel_collision()
    test_relabel_symlink()
    test_relabel_with_csv_update()
    test_result_browser_edit_toggle()
    test_preannotate_preview_and_edit()
    test_qc_label_edit()

    print("\n" + "=" * 60)
    print("🎉 标签修改全链路验证通过")
    print("=" * 60)


if __name__ == "__main__":
    main()
