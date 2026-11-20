"""Model folders shared by the app and the CLI (night 4, D5).

One setting in one file (model_dir and model_dirs in the MTPLX config.toml);
a folder on an unplugged drive is listed as not connected, never dropped;
nothing is copied, moved or deleted.
"""

from __future__ import annotations

import json
import os
import tomllib
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx import model_folders as mf
from mtplx import toml_lines
from mtplx.cli import main
from mtplx.config import load_user_config, user_config_path


def _unplugged(tail: str = "models") -> str:
    # A /Volumes name that is certainly not mounted on the test Mac.
    return f"/Volumes/mtplx-test-unplugged-{uuid.uuid4().hex}/{tail}"


# --- Line-level edits keep everything else ------------------------------------


def test_set_top_level_values_keeps_comments_other_keys_and_tables():
    text = (
        "# MTPLX user configuration\n"
        'model = "Youssofal/x"\n'
        "model_dirs = [\n  \"/a\",\n  \"/b[1]\",\n]\n"
        "profile = \"sustained\"  # mine\n"
        "\n"
        "[extra]\n"
        "model_dir = \"not top level\"\n"
    )
    out = toml_lines.set_top_level_values(
        text, {"model_dirs": ["/c"], "model_dir": "/Volumes/SSD/models"}
    )
    parsed = tomllib.loads(out)
    assert parsed["model_dirs"] == ["/c"] and parsed["model_dir"] == "/Volumes/SSD/models"
    assert parsed["model"] == "Youssofal/x" and parsed["profile"] == "sustained"
    assert parsed["extra"] == {"model_dir": "not top level"}
    assert "# MTPLX user configuration" in out and "# mine" in out
    # A new key goes before the first table, never into it.
    assert out.index('model_dir = "/Volumes/SSD/models"') < out.index("[extra]")
    removed = tomllib.loads(toml_lines.set_top_level_values(out, {"model_dirs": None}))
    assert "model_dirs" not in removed and removed["model_dir"] == "/Volumes/SSD/models"
    assert toml_lines.set_top_level_values(out, {}) == out


# --- Status of a folder ----------------------------------------------------------


def test_a_folder_on_an_unplugged_drive_is_not_connected_not_missing(tmp_path):
    assert mf.folder_status(tmp_path)["available"] is True
    unplugged = mf.folder_status(_unplugged())
    assert unplugged["available"] is False and unplugged["reason"] == "not_connected"
    assert unplugged["volume"].startswith("/Volumes/mtplx-test-unplugged-")
    assert mf.folder_status(tmp_path / "nope")["reason"] == "missing"
    (tmp_path / "file").write_text("x")
    assert mf.folder_status(tmp_path / "file")["reason"] == "not_a_folder"
    # The #466 layout: a link onto an external drive that is unplugged.
    link = tmp_path / "models-link"
    os.symlink(_unplugged(), link)
    assert mf.folder_status(link)["reason"] == "not_connected"


# --- One shared setting -------------------------------------------------------------


def _pack(root: Path, name: str) -> Path:
    pack = root / name
    pack.mkdir(parents=True)
    (pack / "config.json").write_text("{}")
    return pack


def test_add_remove_and_downloads_write_the_keys_the_cli_reads(tmp_path):
    config = user_config_path()
    config.write_text('# mine\nprofile = "sustained"\n', encoding="utf-8")
    ssd = tmp_path / "ssd-models"
    _pack(ssd, "Youssofal--Pack-A")
    unplugged = _unplugged()

    mf.add_folders([str(ssd), unplugged])
    loaded = load_user_config()
    assert list(loaded.model_dirs) == [str(ssd), unplugged]
    assert loaded.profile == "sustained" and "# mine" in config.read_text()

    rows = mf.folders_report(loaded.model_dir, search_dirs=loaded.model_dirs)
    assert [row["role"] for row in rows] == ["downloads", "search", "search"]
    assert rows[1]["packs"] == 1
    assert rows[2]["available"] is False and rows[2]["reason"] == "not_connected"

    # Adding twice changes nothing; a made-up path on the startup disk is refused.
    assert mf.add_folders([str(ssd)])["added"] == []
    with pytest.raises(ValueError, match="no folder"):
        mf.add_folders([str(tmp_path / "typo")])

    # A new download folder: the previous one stays searched, nothing moves.
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    before = sorted(p.name for p in ssd.iterdir())
    mf.set_download_folder(str(downloads))
    loaded = load_user_config()
    assert loaded.model_dir == str(downloads)
    assert loaded.model_dirs[0] == os.environ["MTPLX_MODEL_DIR"]
    assert sorted(p.name for p in ssd.iterdir()) == before

    with pytest.raises(ValueError, match="download folder"):
        mf.remove_folder(str(downloads))
    mf.remove_folder(unplugged)
    assert unplugged not in load_user_config().model_dirs
    assert (ssd / "Youssofal--Pack-A" / "config.json").is_file()


def test_folders_command_lists_and_edits_the_shared_setting(tmp_path, capsys):
    ssd = tmp_path / "ssd"
    _pack(ssd, "Youssofal--Pack-B")
    assert main(["folders", "add", str(ssd), _unplugged(), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["added"]) == 2
    assert [row["reason"] for row in payload["folders"]][1:] == [None, "not_connected"]

    assert main(["folders"]) == 0
    text = capsys.readouterr().out
    assert "not connected; its packs show again when the drive is plugged in" in text
    assert "1 pack" in text

    assert main(["folders", "remove", str(tmp_path / "never-added"), "--json"]) == 1
    assert "not one of the model folders" in json.loads(capsys.readouterr().out)["error"]


def test_list_reports_an_unplugged_folder_instead_of_dropping_it(capsys):
    unplugged = _unplugged()
    mf.add_folders([unplugged])
    assert main(["list", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    rows = {row["path"]: row for row in payload["model_folders"]}
    assert rows[unplugged]["reason"] == "not_connected"
    assert unplugged in payload["model_roots"]


def test_the_swift_toml_fixture_is_current():
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "write_toml_lines_fixture", root / "scripts" / "write_toml_lines_fixture.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixture = root / "apps" / "MTPLXApp" / "Tests" / "Fixtures" / "toml_lines_cases.json"
    assert fixture.read_text(encoding="utf-8") == module.render(), (
        "stale fixture: run `python scripts/write_toml_lines_fixture.py`"
    )
    for case in json.loads(module.render())["cases"]:
        tomllib.loads(case["output"])  # every case the app is held to is valid TOML


def _complete_pack(root, repo_id):
    pack = root / repo_id.replace("/", "--")
    pack.mkdir(parents=True)
    (pack / "config.json").write_text("{}\n", encoding="utf-8")
    (pack / "model.safetensors").write_bytes(b"weights")
    return pack


def test_a_pack_in_a_shared_folder_is_used_not_downloaded_again(tmp_path, monkeypatch):
    """#573 through the shared setting: `mtplx folders add` names the folder,
    and `mtplx start` (the "same configuration" path auto-pulls a repo id)
    resolves the pack there instead of downloading a second copy."""

    from mtplx.commands import public
    from mtplx.config import apply_user_config

    config = tmp_path / "config.toml"
    monkeypatch.setenv("MTPLX_CONFIG", str(config))
    ssd = tmp_path / "ssd"
    # A complete pack under its repo folder name (a non-MTPLX repo id keeps the
    # fixture to config.json plus weights).
    pack = _complete_pack(ssd, "someone/example-pack")
    assert main(["folders", "add", str(ssd), "--config", str(config)]) == 0
    # What `mtplx start` gets from the shared file, as every command does.
    args = SimpleNamespace(command="start", cache_dir=None, model_search_dirs=None, _cli_flags=set())
    apply_user_config(args, config_path=config)
    assert args.model_search_dirs == [str(ssd)]
    pulled = []
    monkeypatch.setattr("mtplx.hf_loader.pull_model", lambda *a, **k: pulled.append(a))
    runtime, resolution = public._quickstart_resolve_model(
        "someone/example-pack",
        cache_dir=str(tmp_path / "downloads"),
        **public._model_search_kwargs(args),
        download=True,
    )
    assert runtime == str(pack.resolve()) and resolution["downloaded"] is False
    assert pulled == []


def test_an_unplugged_drive_stops_an_automatic_download(tmp_path, monkeypatch, capsys):
    """`mtplx start` pulling a model it was told to use earlier (the "same
    configuration" path) while a model folder's drive is unplugged: it says
    so and downloads nothing. Only --download (or a yes to the question)
    fetches a second copy."""

    from mtplx.commands import public

    config = tmp_path / "config.toml"
    monkeypatch.setenv("MTPLX_CONFIG", str(config))
    unplugged = f"/Volumes/mtplx-test-unplugged-{uuid.uuid4().hex}/models"
    config.write_text(f"model_dirs = [{json.dumps(unplugged)}]\n", encoding="utf-8")
    calls = []

    def fake_resolve(model, *, cache_dir, search_dirs=None, download):
        calls.append({"download": download, "search_dirs": list(search_dirs or [])})
        return None, {"model": model, "runtime_model": None, "downloaded": False,
                      "error": {"error": "model is not available locally", "model": model}}

    monkeypatch.setattr(public, "_quickstart_resolve_model", fake_resolve)
    # The wizard's auto-pull policy: a repo id picked earlier downloads
    # without --download.
    monkeypatch.setattr(
        public, "_quickstart_choose_model",
        lambda args, target="terminal": ("someone/example-pack", True),
    )
    code = main(["start", "cli", "--model", "someone/example-pack", "--yes", "--prompt", "hi"])
    out = capsys.readouterr()
    assert code == 1
    assert calls and calls[0]["download"] is False, "no automatic pull while the drive is unplugged"
    assert calls[0]["search_dirs"] == [unplugged]
    assert "not connected" in (out.out + out.err) and unplugged in (out.out + out.err)
    # An explicit --download still downloads.
    calls.clear()
    main(["start", "cli", "--model", "someone/example-pack", "--yes", "--prompt", "hi", "--download"])
    assert calls and calls[0]["download"] is True
