import inspect
from pathlib import Path


def test_openai_main_patches_handle_exit():
    """Verify main() patches uvicorn.Server.handle_exit to call os._exit for clean Ctrl-C termination."""
    openai_path = Path(__file__).resolve().parent.parent / "mtplx" / "server" / "openai.py"
    source = openai_path.read_text()

    assert "handle_exit" in source, "main() must patch uvicorn.Server.handle_exit for clean Ctrl-C termination"
    assert "os._exit" in source, "handle_exit patch must call os._exit to return control to terminal"
