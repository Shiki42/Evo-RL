"""Unit tests for lerobot.utils.pedal_listener.discover_pedal_devices."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("evdev")  # PedalListener.__init__ imports evdev

from lerobot.utils import pedal_listener
from lerobot.utils.pedal_listener import PedalListener, discover_pedal_devices


def _make_fake_pedal(by_id_dir: Path, serial: str) -> Path:
    event_node = by_id_dir / f"event_{serial}"
    event_node.touch()
    link = by_id_dir / f"usb-LinTx_LinTx_Keyboard_{serial}-if01-event-kbd"
    link.symlink_to(event_node)
    return link


@pytest.fixture
def patched_glob(monkeypatch, tmp_path):
    monkeypatch.setattr(
        pedal_listener,
        "LINTX_PEDAL_GLOB",
        str(tmp_path / "usb-LinTx_LinTx_Keyboard_*-if01-event-kbd"),
    )
    return tmp_path


def test_discover_returns_empty_tuple_when_no_pedals(patched_glob):
    assert discover_pedal_devices() == ()


def test_discover_returns_sorted_tuple_for_multiple_pedals(patched_glob):
    link_b = _make_fake_pedal(patched_glob, "BBBBBBBB")
    link_a = _make_fake_pedal(patched_glob, "AAAAAAAA")
    result = discover_pedal_devices()
    assert isinstance(result, tuple)
    assert result == (str(link_a), str(link_b))


def test_pedal_listener_defaults_to_discovery(patched_glob):
    link = _make_fake_pedal(patched_glob, "CAFEBABE")
    listener = PedalListener(on_press=lambda _: None)
    assert listener._device_paths == (str(link),)


def test_pedal_listener_explicit_devices_skip_discovery(patched_glob):
    _make_fake_pedal(patched_glob, "DEADBEEF")
    listener = PedalListener(on_press=lambda _: None, devices=("/explicit/path",))
    assert listener._device_paths == ("/explicit/path",)
