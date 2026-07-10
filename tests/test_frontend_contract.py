from pathlib import Path

import yaml
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
FILES_URL = "https://huggingface.co/spaces/Bwillcox/BenBot/tree/main"
SPACE_URL = "https://huggingface.co/spaces/Bwillcox/BenBot"


def _frontend() -> BeautifulSoup:
    return BeautifulSoup((ROOT / "web" / "index.html").read_text(encoding="utf-8"), "html.parser")


def test_frontend_exposes_readiness_degradation_and_public_source() -> None:
    page = _frontend()

    service_status = page.select_one("#serviceStatus[role='status']")
    degraded_notice = page.select_one("#degradedNotice[role='alert'][hidden]")
    links = {link.get("href") for link in page.select(".project-links a")}

    assert service_status is not None
    assert degraded_notice is not None
    assert SPACE_URL in links
    assert FILES_URL in links


def test_frontend_describes_count_prior_without_fine_tuning_claim() -> None:
    text = " ".join(_frontend().stripped_strings).lower()

    assert "pretrained maia2" in text
    assert "count-based" in text
    assert "not a fine-tuned neural model" in text
    assert "ai chess twin" not in text
    assert "trained to mirror" not in text


def test_frontend_script_consumes_readiness_and_safety_status() -> None:
    script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")

    assert 'fetch("/ready"' in script
    assert 'status === "degraded"' in script
    assert "config.stockfish_status" in script
    assert "config.stockfish_error" in script
    assert "60_000" in script


def test_space_readme_is_deployable_model_card() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    _, front_matter, body = readme.split("---", 2)
    metadata = yaml.safe_load(front_matter)

    assert metadata["title"] == "BenBot"
    assert metadata["sdk"] == "docker"
    assert metadata["app_port"] == 7860
    assert FILES_URL in body
    for section in (
        "## Data card",
        "### Split logic",
        "### Privacy and retention",
        "## Evaluation",
        "## Hyperparameters",
        "## Deployment and operational constraints",
        "## Known limitations and failure modes",
        "## Personal contribution and third-party components",
    ):
        assert section in body
    assert "not a personally fine-tuned neural network" in body.lower()
    for evidence in (
        "59.7581%",
        "1.227588",
        "8.7411%",
        "`strategy=fen`",
        "14/200 choices",
        "310.53 ms",
        "reports/heldout_evaluation/summary.json",
        "reports/safety_evaluation/summary.json",
    ):
        assert evidence in body
