#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""JSONL 数据文件：每行一个完整 episode。"""

import json
import os
from typing import Any, Dict, Sequence


def append_episode(path: str, episode: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(list(episode), ensure_ascii=False, allow_nan=False) + "\n")
        output_file.flush()


def prepare_output_files(training_output: str, raw_output: str, clear_output: bool) -> None:
    """按开关决定清空或追加；training 与 raw 不能是同一文件。"""
    if os.path.abspath(training_output) == os.path.abspath(raw_output):
        raise ValueError("训练文件与 raw 文件不能相同")
    for path in (training_output, raw_output):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if clear_output:
            with open(path, "w", encoding="utf-8"):
                pass


def count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as source:
        return sum(1 for line in source if line.strip())


def merge_files(sources: Sequence[str], destination: str) -> int:
    """把各 worker 文件按顺序追加到目标文件，返回合并的行数，成功后删除源文件。"""
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    merged = 0
    with open(destination, "a", encoding="utf-8") as output_file:
        for source in sources:
            if not os.path.exists(source):
                continue
            with open(source, "r", encoding="utf-8") as input_file:
                for line in input_file:
                    if line.strip():
                        output_file.write(line if line.endswith("\n") else line + "\n")
                        merged += 1
    for source in sources:
        if os.path.exists(source):
            os.remove(source)
    return merged
