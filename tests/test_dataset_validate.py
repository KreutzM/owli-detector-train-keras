from pathlib import Path
from typer.testing import CliRunner
from owli_train.cli import app

runner = CliRunner()

def test_validate_min_coco():
    coco = Path('tests/data/coco_min.json')
    r = runner.invoke(app, ['dataset', 'validate', '--coco', str(coco)])
    assert r.exit_code == 0
    assert 'OK' in r.stdout
