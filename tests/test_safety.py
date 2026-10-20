import json
from pathlib import Path

import pytest

from hkgai_agent.safety import SafetyError, Workspace
from hkgai_agent.tools import ListFilesTool, ReadFileTool, ToolExecutor, ToolRegistry, WriteFileTool
from hkgai_agent.types import ToolCall


@pytest.mark.parametrize("unsafe", [
    "../outside.txt", "sub/../../outside.txt", "..\\outside.txt", "/etc/passwd", "\\rooted.txt",
    "C:\\Windows\\system.ini", "C:relative.txt", "\\\\server\\share\\file", "file.txt:stream",
    "NUL", "con.txt", "sub/COM1", "file. ", "file.", "\x00file", "",
])
def test_unsafe_paths_rejected(tmp_path, unsafe):
    with pytest.raises(SafetyError):
        Workspace(tmp_path).resolve(unsafe)


def test_valid_workspace_relative_paths(tmp_path):
    workspace = Workspace(tmp_path)
    assert workspace.resolve("sub/file.txt") == tmp_path / "sub" / "file.txt"
    assert workspace.resolve("sub\\file.txt") == tmp_path / "sub" / "file.txt"
    assert workspace.resolve(".", allow_root=True) == tmp_path
    with pytest.raises(SafetyError):
        workspace.resolve(".")


def symlink_or_skip(link: Path, target: Path):
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except (OSError, NotImplementedError):
        pytest.skip("This platform does not permit symlink creation")


def test_symlink_escape_and_new_file_through_escaping_parent(tmp_path):
    root = tmp_path / "workspace"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    symlink_or_skip(root / "link", outside)
    with pytest.raises(SafetyError):
        Workspace(root).resolve("link/new.txt")


async def test_safe_file_read_write_and_explicit_overwrite(tmp_path):
    workspace = Workspace(tmp_path)
    tools = [ReadFileTool(workspace), WriteFileTool(workspace), ListFilesTool(workspace)]
    executor = ToolExecutor(ToolRegistry(tools), allowed_tools=frozenset(tool.name for tool in tools))
    results = await executor.execute_batch([
        ToolCall("write", "write_file", '{"path":"hello.txt","text":"hello"}'),
        ToolCall("read", "read_file", '{"path":"hello.txt"}'),
        ToolCall("overwrite", "write_file", '{"path":"hello.txt","text":"replacement"}'),
        ToolCall("list", "list_files", "{}"),
    ])
    assert [result.ok for result in results] == [True, True, False, True]
    assert results[1].output["text"] == "hello"
    assert results[2].error_code == "invalid_input"
    assert results[3].output["entries"] == ["hello.txt"]
    replacement = await executor.execute_batch([
        ToolCall("replace", "write_file", '{"path":"hello.txt","text":"new","overwrite":true}')
    ])
    assert replacement[0].ok
    assert (tmp_path / "hello.txt").read_text() == "new"
    assert not list(tmp_path.glob(".agent-*"))


@pytest.mark.parametrize("name", ["read_file", "write_file", "list_files"])
async def test_every_filesystem_tool_uses_workspace_guard(tmp_path, name):
    workspace = Workspace(tmp_path)
    tools = [ReadFileTool(workspace), WriteFileTool(workspace), ListFilesTool(workspace)]
    executor = ToolExecutor(ToolRegistry(tools), allowed_tools=frozenset(tool.name for tool in tools))
    arguments = {"path": "../outside.txt"}
    if name == "write_file":
        arguments["text"] = "forbidden"
    result = (await executor.execute_batch([ToolCall("bad", name, json.dumps(arguments))]))[0]
    assert result.error_code == "unsafe_path"
    assert not (tmp_path.parent / "outside.txt").exists()


async def test_non_utf8_and_oversized_files_rejected(tmp_path):
    (tmp_path / "binary").write_bytes(b"\xff\xfe")
    (tmp_path / "large").write_bytes(b"x" * 10)
    executor = ToolExecutor(
        ToolRegistry([ReadFileTool(Workspace(tmp_path), max_bytes=4)]), allowed_tools=frozenset({"read_file"})
    )
    results = await executor.execute_batch([
        ToolCall("binary", "read_file", '{"path":"binary"}'),
        ToolCall("large", "read_file", '{"path":"large"}'),
    ])
    assert all(result.error_code == "invalid_input" for result in results)


async def test_utf8_byte_write_limit(tmp_path):
    executor = ToolExecutor(
        ToolRegistry([WriteFileTool(Workspace(tmp_path), max_bytes=4)]), allowed_tools=frozenset({"write_file"})
    )
    result = (await executor.execute_batch([
        ToolCall("large", "write_file", json.dumps({"path": "large.txt", "text": "\u4e2d\u6587"}))
    ]))[0]
    assert result.error_code == "invalid_input"
    assert not (tmp_path / "large.txt").exists()
