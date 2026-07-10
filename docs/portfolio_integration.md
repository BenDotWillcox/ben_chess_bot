# Portfolio integration

The portfolio is a separate codebase and deployment. This repository owns the status contract and
accurate project wording; the portfolio must consume that contract explicitly.

## BenBot card replacement

Use this copy in place of “trained to mirror my rapid-game decision patterns”:

```html
<div class="project-header">
  <h3 class="project-title">BenBot &mdash; Personalized Maia2 Chess Policy</h3>
  <span
    id="benbot-status"
    class="status-badge status-badge--checking"
    data-ready-url="https://bwillcox-benbot.hf.space/ready"
    role="status"
    aria-live="polite"
  >Checking</span>
</div>

<div class="project-description">
  <p>
    A playable chess policy that blends pretrained <strong>Maia2</strong> move probabilities
    with an inspectable count-based prior from my historical rapid games. Leakage-free validation
    selected exact-position memory; this is not a personally fine-tuned neural model.
  </p>
  <p>
    The live app exposes candidate probabilities, memory source, policy settings, and Stockfish
    veto decisions. Its readiness badge distinguishes full operation, optional-safety degradation,
    and model unavailability.
  </p>
  <p>
    On 3,638 later moves from a game-isolated chronological test split, personalization raised
    top-1 next-move accuracy from 55.50% to 59.76% and reduced log loss from 1.3436 to 1.2276.
    Exact-memory coverage was 8.74%, so most positions still use base Maia2.
  </p>
</div>

<div class="project-links">
  <a
    rel="noreferrer"
    target="_blank"
    class="cta-btn cta-btn--hero"
    href="https://huggingface.co/spaces/Bwillcox/BenBot"
  >See Live</a>
  <a
    rel="noreferrer"
    target="_blank"
    class="cta-btn cta-btn--hero"
    href="https://huggingface.co/spaces/Bwillcox/BenBot/tree/main"
  >Public Files / Source</a>
</div>
```

This addresses both claim precision and the missing source link. Replace the existing combined-policy
explainer with: “Maia2 supplies the human-move distribution; a count prior from exact positions in the
chronological training split nudges choices toward Ben's recorded tendencies. Validation selected
FEN memory with `alpha=0.7` and `min_count=1`; prefix memory remains an evaluated alternative.” Do not
use “fine-tuned,” “trained model,” or “AI twin” as a technical description.

## Dynamic status badge

Do not hard-code `Live`. Fetch the public `/ready` endpoint on page load and refresh it periodically.
The endpoint returns `ready`, `degraded`, or `unavailable`; a network error is also unavailable.

For a no-JavaScript badge, link the service's self-contained cached SVG to the active readiness
endpoint. The image reports the last-known state without causing a heavyweight model load; following
the link performs a fresh dependency check.

```html
<a
  href="https://bwillcox-benbot.hf.space/ready"
  target="_blank"
  rel="noreferrer"
  aria-label="Check BenBot dependency readiness"
>
  <img
    src="https://bwillcox-benbot.hf.space/status.svg"
    alt="BenBot service status"
    width="132"
    height="20"
  />
</a>
```

For the portfolio's existing pill styling, use the live text version below. Unlike the cached SVG,
it calls `/ready`, so it both initializes dependencies and reports their current state.

```js
const benbotStatus = document.querySelector("#benbot-status");

async function refreshBenbotStatus() {
  const readyUrl = benbotStatus?.dataset.readyUrl;
  if (!readyUrl) return;

  try {
    const response = await fetch(readyUrl, {
      headers: { Accept: "application/json" },
      signal: AbortSignal.timeout(8000),
    });
    const payload = await response.json();
    const status = payload.status || (response.ok && payload.ready ? "ready" : "unavailable");
    const labels = { ready: "Live", degraded: "Degraded", unavailable: "Unavailable" };
    benbotStatus.textContent = labels[status] || "Unavailable";
    benbotStatus.className = `status-badge status-badge--${status}`;
  } catch {
    benbotStatus.textContent = "Unavailable";
    benbotStatus.className = "status-badge status-badge--unavailable";
  }
}

refreshBenbotStatus();
window.setInterval(refreshBenbotStatus, 60_000);
```

Add portfolio badge variants alongside its existing `--live` style:

```scss
.status-badge--checking {
  background: rgba(#6c757d, 0.12);
  color: #5f676d;
  &::before { background: #6c757d; }
}

.status-badge--degraded {
  background: rgba(#d88917, 0.12);
  color: #8a5a12;
  &::before { background: #d88917; }
}

.status-badge--unavailable {
  background: rgba(#b93632, 0.12);
  color: #92302d;
  &::before { background: #b93632; }
}
```

If the backend restricts CORS, include the production portfolio origin in
`CHESS_BOT_CORS_ORIGINS`. This badge reflects application readiness. A hosting scheduler failure
will appear as a timeout/unavailable result, but external uptime monitoring is still needed for
alerts because code inside a stopped container cannot alert on itself.
