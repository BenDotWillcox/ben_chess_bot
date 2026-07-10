from __future__ import annotations

import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
from playwright.sync_api import expect, sync_playwright


pytestmark = pytest.mark.browser


@contextmanager
def browser_server():
    configured_url = os.getenv("CHESS_BOT_TEST_BASE_URL")
    if configured_url:
        yield configured_url
        return

    root = Path(__file__).resolve().parent.parent
    process = subprocess.Popen(
        [sys.executable, str(root / "tests" / "browser_server.py")],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base_url = "http://127.0.0.1:8765"
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"{base_url}/health", timeout=0.5).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        else:
            raise RuntimeError("browser smoke server did not start")
        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def test_browser_can_start_and_play_a_smoke_game() -> None:
    with browser_server() as base_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(base_url, wait_until="networkidle")

        expect(page.locator("#board .square")).to_have_count(64)
        expect(page.locator("#gameStatus")).to_have_text("Your move")
        expect(page.locator("#serviceStatusText")).to_contain_text("Ready")

        page.locator('[data-square="e2"]').click()
        page.locator('[data-square="e4"]').click()
        expect(page.locator("#gameStatus")).to_have_text("Your move", timeout=10_000)
        expect(page.locator("#candidateList li")).not_to_have_count(0)
        expect(page.locator("#candidateList li").first.locator("strong")).to_have_text("e5")
        browser.close()
