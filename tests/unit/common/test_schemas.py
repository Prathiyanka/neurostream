import json

import pytest
from pydantic import ValidationError

from neurostream_common.schemas import (
    DataPacket,
    SessionEnd,
    SessionHello,
    check_packet_against_hello,
    parse_device_message,
)

HELLO = {
    "type": "hello",
    "session_id": "s1",
    "sfreq": 160,
    "ch_names": ["C3", "Cz"],
    "chunk_size": 2,
}


def packet(**kw: object) -> dict[str, object]:
    base: dict[str, object] = {
        "type": "data",
        "session_id": "s1",
        "seq": 0,
        "timestamp": 1.0,
        "samples": [[1.0, 2.0], [3.0, 4.0]],
        "truth": {"labels": [0, 1]},
    }
    base.update(kw)
    return base


def test_parses_each_message_type() -> None:
    assert isinstance(parse_device_message(json.dumps(HELLO)), SessionHello)
    assert isinstance(parse_device_message(json.dumps(packet())), DataPacket)
    assert isinstance(parse_device_message('{"type":"end","session_id":"s1"}'), SessionEnd)


@pytest.mark.parametrize(
    "bad",
    [
        packet(seq=-1),
        packet(samples=[]),
        packet(samples=[[1.0, 2.0], [3.0]]),  # ragged
        packet(truth={"labels": [0]}),  # wrong label count
        packet(session_id="bad id!"),
        packet(extra_field=1),  # unknown fields rejected
        packet(type="nope"),
        packet(samples=[["x", 2.0], [3.0, 4.0]]),
    ],
)
def test_rejects_malformed_packets(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        parse_device_message(json.dumps(bad))


def test_rejects_nan_and_infinity() -> None:
    raw = '{"type":"data","session_id":"s1","seq":0,"timestamp":1,"samples":[[NaN,1],[1,Infinity]]}'
    with pytest.raises(ValidationError):
        parse_device_message(raw)


def test_rejects_duplicate_channel_names() -> None:
    with pytest.raises(ValidationError):
        parse_device_message(json.dumps({**HELLO, "ch_names": ["C3", "C3"]}))


def test_rejects_garbage_json() -> None:
    with pytest.raises(ValidationError):
        parse_device_message("{not json")


def test_packet_checked_against_hello() -> None:
    hello = SessionHello.model_validate(HELLO)
    assert check_packet_against_hello(DataPacket.model_validate(packet()), hello) is None
    three_ch = DataPacket.model_validate(packet(samples=[[1.0, 2.0]] * 3, truth=None))
    assert "channels" in (check_packet_against_hello(three_ch, hello) or "")
    long = DataPacket.model_validate(packet(samples=[[1.0, 2.0, 3.0]] * 2, truth=None))
    assert "chunk_size" in (check_packet_against_hello(long, hello) or "")
    other = DataPacket.model_validate(packet(session_id="s2"))
    assert "session_id" in (check_packet_against_hello(other, hello) or "")
