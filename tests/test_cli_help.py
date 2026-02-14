from typer.testing import CliRunner

from owli_train.cli import app

runner = CliRunner()


def test_help():
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0


def test_dataset_help_lists_normalize():
    r = runner.invoke(app, ["dataset", "--help"])
    assert r.exit_code == 0
    assert "normalize" in r.stdout
