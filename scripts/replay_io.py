from __future__ import annotations

from enum import IntEnum
from pathlib import Path
import struct
import zlib


MAGIC = b"RUNR"
FORMAT_VERSION = 1
HEADER = struct.Struct("<4sBBII")
MAX_ACTION_COUNT = 1_000_000


class ReplayRuleset(IntEnum):
    CLASSIC_100M = 1
    HURDLE_110M = 2
    LONG_JUMP_30M = 3


RULESETS_BY_GAME_MODE = {
    "classic_100m": ReplayRuleset.CLASSIC_100M,
    "hurdle_110m": ReplayRuleset.HURDLE_110M,
    "long_jump_30m": ReplayRuleset.LONG_JUMP_30M,
}


def ruleset_for_game_mode(game_mode: str) -> ReplayRuleset:
    try:
        return RULESETS_BY_GAME_MODE[game_mode]
    except KeyError as error:
        raise ValueError(f"unknown game mode: {game_mode}") from error


def write_replay(path: Path, actions: list[int], ruleset: ReplayRuleset) -> None:
    if not 0 < len(actions) <= MAX_ACTION_COUNT:
        raise ValueError(f"invalid replay action count: {len(actions)}")

    payload = bytearray((len(actions) + 1) // 2)
    for index, action in enumerate(actions):
        if not 0 <= action <= 15:
            raise ValueError(f"action {index} is outside the 4-bit range: {action}")
        payload[index // 2] |= action if index % 2 == 0 else action << 4

    header = HEADER.pack(
        MAGIC,
        FORMAT_VERSION,
        int(ruleset),
        len(actions),
        zlib.crc32(payload),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + payload)


def read_replay(path: Path) -> tuple[ReplayRuleset, list[int]]:
    data = path.read_bytes()
    if len(data) < HEADER.size:
        raise ValueError("replay is shorter than its header")

    magic, version, raw_ruleset, action_count, expected_crc = HEADER.unpack_from(data)
    if magic != MAGIC:
        raise ValueError("invalid replay signature")
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported replay format version: {version}")
    try:
        ruleset = ReplayRuleset(raw_ruleset)
    except ValueError as error:
        raise ValueError(f"unknown replay ruleset: {raw_ruleset}") from error
    if not 0 < action_count <= MAX_ACTION_COUNT:
        raise ValueError(f"invalid replay action count: {action_count}")

    payload = data[HEADER.size :]
    if len(payload) != (action_count + 1) // 2:
        raise ValueError("replay size does not match its action count")
    if zlib.crc32(payload) != expected_crc:
        raise ValueError("replay payload checksum mismatch")
    if action_count % 2 and payload[-1] & 0xF0:
        raise ValueError("non-zero padding in replay payload")

    actions = [
        payload[index // 2] & 0x0F if index % 2 == 0 else payload[index // 2] >> 4
        for index in range(action_count)
    ]
    return ruleset, actions
