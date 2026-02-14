from typer.testing import CliRunner
from owli_train.cli import app

runner = CliRunner()


def test_help():
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
