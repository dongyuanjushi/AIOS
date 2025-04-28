"""
MCP server that exposes Ubuntu GUI + shell controls.
Run inside the VM:  uv run ubuntu_mcp_server.py
"""
from mcp.server.fastmcp import FastMCP          # auto-generates tool schemas :contentReference[oaicite:3]{index=3}
import subprocess, json, os, shlex
mcp = FastMCP("ubuntu-gui")

# ---------- low-level helpers ----------
def _run(cmd: list[str] | str, capture: bool = True) -> str:
    """Run shell command safely and return stdout/err."""
    out = subprocess.run(
        cmd if isinstance(cmd, list) else shlex.split(cmd),
        capture_output=capture, text=True, check=False
    )
    return out.stdout.strip() or out.stderr.strip() or "(no output)"

# ---------- MCP TOOLS ----------
@mcp.tool(description="Move mouse to absolute X/Y coordinates (pixels)")
async def move_mouse(x: int, y: int) -> str:
    return _run(["xdotool", "mousemove", str(x), str(y)])

@mcp.tool(description="Click mouse button at X/Y; button defaults to left (1)")
async def click(x: int, y: int, button: str | None = "left") -> str:
    btn = {"left": "1", "middle": "2", "right": "3"}.get(button or "left", "1")
    _run(["xdotool", "mousemove", str(x), str(y)], capture=False)
    return _run(["xdotool", "click", btn])

@mcp.tool(description="Type Unicode text at current cursor position")
async def type_text(text: str) -> str:
    return _run(["xdotool", "type", "--", text])

@mcp.tool(description="Launch a desktop application by its Exec name")
async def open_app(app_name: str) -> str:
    return _run([app_name], capture=False)

@mcp.tool(description="Execute arbitrary shell command (non-interactive)")
async def run_command(command: str) -> str:
    return _run(command)

@mcp.tool(description="Return JSON list of open window titles")
async def list_windows() -> str:
    raw = _run(["wmctrl", "-l"])
    titles = [l.split(None, 3)[-1] for l in raw.splitlines() if l]
    return json.dumps(titles)

if __name__ == "__main__":
    mcp.run(transport="stdio")   # 1-line launch as required by the spec :contentReference[oaicite:4]{index=4}
