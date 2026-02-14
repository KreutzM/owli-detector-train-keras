from pathlib import Path


def test_pyproject_has_ruff_temp_excludes():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "extend-exclude" in pyproject
    assert '"TMP"' in pyproject
    assert '"tmp"' in pyproject
    assert '".tmp"' in pyproject
    assert '"work"' in pyproject
    assert '"outputs"' in pyproject
    assert '"data"' in pyproject
    assert "norecursedirs" in pyproject


def test_gitignore_has_temp_patterns():
    gitignore = Path(".gitignore").read_text(encoding="utf-8")
    assert "TMP/" in gitignore
    assert "tmp/" in gitignore
    assert ".tmp/" in gitignore
    assert "**/TMP/**" in gitignore
    assert "**/tmp/**" in gitignore


def test_e2e_coco128_script_exists_and_contains_key_steps():
    script_path = Path("scripts/e2e_coco128_smoke.ps1")
    assert script_path.is_file()

    script = script_path.read_text(encoding="utf-8")
    assert "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip" in script
    assert '"dataset", "import", "yolo"' in script
    assert '"dataset", "split"' in script
    assert '"dataset", "export", "modelmaker-csv"' in script
    assert "train efficientdet" in script
    assert '"inspect", "tflite"' in script
    assert "requirements\\modelmaker.txt" in script
