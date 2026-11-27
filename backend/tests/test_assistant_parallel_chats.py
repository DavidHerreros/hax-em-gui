"""The chat socket with turns that block: a new message or a stop ends the running turn at once
(its worker is left to finish on its own and nothing it still emits is sent), two sockets stream
side by side, an answer never cancels, and a cancel or a disconnect releases what waits.

stream_events is faked; each fake turn blocks on an Event the test controls.
"""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from hax_gui import assistant as assistant_bridge
from hax_gui.app import app

WS = "ws://127.0.0.1/ws/assistant/chat"
needs_assistant = pytest.mark.skipif(not assistant_bridge.available(),
                                     reason="hax_assistant is not installed")
pytestmark = needs_assistant


def _recv(ws, timeout=5.0):
    box: dict = {}
    t = threading.Thread(target=lambda: box.setdefault("m", ws.receive_json()), daemon=True)
    t.start()
    t.join(timeout)
    assert "m" in box, "no frame from the server"
    return box["m"]


def _wait(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.01)


@pytest.fixture
def fake(monkeypatch):
    """A stream_events whose "slow" turns block in one long step until ``release`` is set."""
    st = {"release": threading.Event(), "inside": threading.Event(), "late": [], "finished": [],
          "interactions": [], "threads": []}

    def stream(messages, grounded=True, run=None, allow_actions=False, interaction=None, tools=None):
        text = messages[-1]["content"]
        st["interactions"].append(interaction)
        st["threads"].append(threading.current_thread())
        try:
            yield {"type": "status", "text": f"{text}: inspecting"}
            if text.startswith("slow"):
                st["inside"].set()
                st["release"].wait(30)          # the long blocking step (a run's inspection)
                st["late"].append(text)
                yield {"type": "content", "delta": "late"}
            yield {"type": "content", "delta": f"{text}: answer"}
        finally:
            st["finished"].append(text)
    monkeypatch.setattr(assistant_bridge, "stream_events", stream)
    yield st
    st["release"].set()


def _turn(text, turn):
    return {"messages": [{"role": "user", "content": text}], "turn": turn}


def test_new_message_interrupts_a_blocked_turn(fake):
    with TestClient(app).websocket_connect(WS) as ws:
        ws.send_json(_turn("slow", 1))
        assert _recv(ws) == {"status": "slow: inspecting"}
        assert fake["inside"].wait(5)
        ws.send_json(_turn("fast", 2))
        assert _recv(ws) == {"done": True, "turn": 1}
        assert _recv(ws) == {"status": "fast: inspecting"}       # while the old worker still blocks
        assert _recv(ws) == {"token": "fast: answer"}
        assert _recv(ws) == {"done": True, "turn": 2}
        assert fake["late"] == [] and fake["threads"][0].is_alive()

        fake["release"].set()                                     # the old step ends; its events go nowhere
        _wait(lambda: "slow" in fake["finished"])
        assert fake["late"] == ["slow"]
        ws.send_json(_turn("fast", 3))
        assert [_recv(ws) for _ in range(3)] == [{"status": "fast: inspecting"}, {"token": "fast: answer"},
                                                 {"done": True, "turn": 3}]


def test_stop_ends_a_blocked_turn_at_once(fake):
    with TestClient(app).websocket_connect(WS) as ws:
        ws.send_json(_turn("slow", 5))
        assert _recv(ws) == {"status": "slow: inspecting"}
        assert fake["inside"].wait(5)
        t0 = time.monotonic()
        ws.send_json({"stop": True})
        assert _recv(ws) == {"done": True, "turn": 5}
        assert time.monotonic() - t0 < 2 and not fake["release"].is_set() and fake["late"] == []
        assert fake["interactions"][0]._cancel.is_set()


def test_two_sockets_stream_side_by_side(monkeypatch):
    both = threading.Barrier(2, timeout=10)       # passes only while both turns are in flight

    def stream(messages, grounded=True, run=None, allow_actions=False, interaction=None, tools=None):
        who = messages[-1]["content"]
        yield {"type": "status", "text": f"{who} step"}
        both.wait()
        for i in (1, 2):
            yield {"type": "content", "delta": f"{who}-{i}"}
    monkeypatch.setattr(assistant_bridge, "stream_events", stream)
    client = TestClient(app)
    with client.websocket_connect(WS) as a, client.websocket_connect(WS) as b:
        a.send_json(_turn("A", 7))
        b.send_json(_turn("B", 9))
        got_a = [_recv(a, 15) for _ in range(4)]
        got_b = [_recv(b, 15) for _ in range(4)]
    assert got_a == [{"status": "A step"}, {"token": "A-1"}, {"token": "A-2"}, {"done": True, "turn": 7}]
    assert got_b == [{"status": "B step"}, {"token": "B-1"}, {"token": "B-2"}, {"done": True, "turn": 9}]


def test_answer_continues_and_cancel_releases_a_pending_question(monkeypatch):
    from hax_assistant.interaction import Cancelled, Question
    released = threading.Event()

    def stream(messages, grounded=True, run=None, allow_actions=False, interaction=None, tools=None):
        yield {"type": "status", "text": "asking"}
        try:
            ans = interaction.ask(Question("go", "Go on?", kind="confirm"))
        except Cancelled:
            released.set()
            return
        yield {"type": "content", "delta": f"answered {ans}"}
    monkeypatch.setattr(assistant_bridge, "stream_events", stream)
    with TestClient(app).websocket_connect(WS) as ws:
        ws.send_json(_turn("q", 1))
        assert _recv(ws) == {"status": "asking"}
        q = _recv(ws)["question"]
        ws.send_json({"answer": {"id": q["id"], "value": "yes"}})
        assert _recv(ws) == {"token": "answered yes"}
        assert _recv(ws) == {"done": True, "turn": 1}

        ws.send_json(_turn("q", 2))
        assert _recv(ws) == {"status": "asking"}
        assert "question" in _recv(ws)
        ws.send_json(_turn("q", 3))                           # interrupts the turn waiting on its question
        assert _recv(ws) == {"done": True, "turn": 2}
        assert released.wait(1)
        assert _recv(ws) == {"status": "asking"}
        q3 = _recv(ws)["question"]
        ws.send_json({"answer": {"id": q["id"], "value": "no"}})     # a stale answer: ignored, no cancel
        ws.send_json({"answer": {"id": q3["id"], "value": "no"}})
        assert _recv(ws) == {"token": "answered no"}
        assert _recv(ws) == {"done": True, "turn": 3}


def test_disconnect_mid_turn_leaves_nothing_waiting(fake):
    def session():
        with TestClient(app).websocket_connect(WS) as ws:
            ws.send_json(_turn("slow", 1))
            assert _recv(ws) == {"status": "slow: inspecting"}
            assert fake["inside"].wait(5)
    t = threading.Thread(target=session, daemon=True)
    t.start()
    t.join(10)
    assert not t.is_alive(), "the socket did not close while its turn was blocked"
    _wait(lambda: fake["interactions"][0]._cancel.is_set())
    fake["release"].set()
    fake["threads"][0].join(5)
    assert not fake["threads"][0].is_alive() and fake["finished"] == ["slow"]
