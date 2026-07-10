const pieceGlyphs = {
  P: "♙",
  N: "♘",
  B: "♗",
  R: "♖",
  Q: "♕",
  K: "♔",
  p: "♟",
  n: "♞",
  b: "♝",
  r: "♜",
  q: "♛",
  k: "♚",
};

const state = {
  game: null,
  readiness: null,
  selected: null,
  pendingPromotion: null,
  animatingMove: null,
  busy: false,
  requestId: 0,
  readinessPromise: null,
};

const policyOptions = {
  top_k: 5,
  mode: "sample",
  temperature: 0.8,
};

const els = {
  board: document.querySelector("#board"),
  gameStatus: document.querySelector("#gameStatus"),
  newGameButton: document.querySelector("#newGameButton"),
  humanColor: document.querySelector("#humanColor"),
  identityPanel: document.querySelector("#identityPanel"),
  identityStatus: document.querySelector("#identityStatus"),
  serviceStatus: document.querySelector("#serviceStatus"),
  serviceStatusText: document.querySelector("#serviceStatusText"),
  degradedNotice: document.querySelector("#degradedNotice"),
  turnValue: document.querySelector("#turnValue"),
  botColorValue: document.querySelector("#botColorValue"),
  sourceValue: document.querySelector("#sourceValue"),
  configMode: document.querySelector("#configMode"),
  configTopK: document.querySelector("#configTopK"),
  configTemperature: document.querySelector("#configTemperature"),
  configStrategy: document.querySelector("#configStrategy"),
  configAlpha: document.querySelector("#configAlpha"),
  configMinCount: document.querySelector("#configMinCount"),
  configStockfishVeto: document.querySelector("#configStockfishVeto"),
  candidateList: document.querySelector("#candidateList"),
  promotionDialog: document.querySelector("#promotionDialog"),
  gameOverDialog: document.querySelector("#gameOverDialog"),
  gameOverTitle: document.querySelector("#gameOverTitle"),
  gameOverMessage: document.querySelector("#gameOverMessage"),
};

function setStatus(message, isError = false) {
  els.gameStatus.textContent = message;
  els.gameStatus.classList.toggle("error", isError);
}

function setBusy(isBusy) {
  state.busy = isBusy;
  els.board.classList.toggle("busy", isBusy);
  els.newGameButton.disabled = isBusy;
  els.humanColor.disabled = isBusy;
}

function optionsPayload() {
  return { ...policyOptions };
}

function delay(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function setIdentity(status, isThinking = false) {
  els.identityStatus.textContent = status;
  els.identityPanel.classList.toggle("thinking", isThinking);
}

function normalizeReadiness(payload) {
  const dependencies = payload?.dependencies || {};
  const policy = dependencies.policy || {};
  const stockfish = dependencies.stockfish || {};
  const policyReady = policy.status === "ready" || payload?.policy_loaded === true;
  const stockfishEnabled = stockfish.enabled ?? payload?.stockfish_enabled ?? false;
  const stockfishReady = !stockfishEnabled || stockfish.status === "ready" || payload?.stockfish_ready === true;
  let status = payload?.status;

  if (!status) {
    if (payload?.ready === false || (!policyReady && payload?.ok === false)) {
      status = "unavailable";
    } else if (policyReady && stockfishEnabled && !stockfishReady) {
      status = "degraded";
    } else if (payload?.ready === true || payload?.ok === true || policyReady) {
      status = "ready";
    } else {
      status = "unavailable";
    }
  }

  return {
    status,
    policyReady,
    stockfishEnabled,
    stockfishReady,
    error: policy.error || stockfish.error || payload?.detail || null,
  };
}

function renderReadiness(readiness) {
  state.readiness = readiness;
  const status = readiness?.status || "checking";
  els.serviceStatus.className = `service-status service-status--${status}`;

  if (status === "ready") {
    els.serviceStatusText.textContent = readiness.stockfishEnabled ? "Ready · Maia2 + Stockfish" : "Ready · Maia2";
  } else if (status === "degraded") {
    els.serviceStatusText.textContent = "Ready · Safety degraded";
  } else if (status === "unavailable") {
    els.serviceStatusText.textContent = "Model unavailable";
  } else {
    els.serviceStatusText.textContent = "Checking service";
  }

  const stockfishDegraded = status === "degraded" && readiness.stockfishEnabled && !readiness.stockfishReady;
  els.degradedNotice.hidden = !stockfishDegraded;
  els.serviceStatus.title = readiness?.error || "";
}

async function refreshReadiness({ quiet = false } = {}) {
  if (state.readinessPromise) {
    return state.readinessPromise;
  }
  if (!quiet) {
    renderReadiness({ status: "checking", stockfishEnabled: false, stockfishReady: false });
  }

  state.readinessPromise = (async () => {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 10_000);
    try {
      const response = await fetch("/ready", {
        headers: { Accept: "application/json" },
        signal: controller.signal,
      });
      const payload = await response.json();
      const readiness = normalizeReadiness(payload);
      if (!response.ok && !["starting", "unavailable"].includes(readiness.status)) {
        throw new Error(payload.detail || `${response.status} ${response.statusText}`);
      }
      renderReadiness(readiness);
      return readiness;
    } catch (error) {
      const readiness = {
        status: "unavailable",
        policyReady: false,
        stockfishEnabled: false,
        stockfishReady: false,
        error: error.name === "AbortError" ? "Readiness check timed out" : error.message,
      };
      renderReadiness(readiness);
      return readiness;
    } finally {
      window.clearTimeout(timeout);
      state.readinessPromise = null;
    }
  })();
  return state.readinessPromise;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      message = payload.detail || message;
    } catch {
      // Keep the HTTP status text.
    }
    throw new Error(message);
  }
  return response.json();
}

async function syncCurrentGame() {
  if (!state.game?.game_id) {
    return;
  }
  state.game = await api(`/game/${state.game.game_id}`);
  render();
}

async function recoverTurnDesync() {
  setBusy(false);
  await syncCurrentGame();
  if (!isHumanTurn() && !state.game?.game_over) {
    setStatus("BenBot thinking...");
    setIdentity("BenBot is thinking", true);
    await delay(1000);
    await syncCurrentGame();
  }
  if (!isHumanTurn() && !state.game?.game_over) {
    try {
      setBusy(true);
      state.game = await api(`/game/${state.game.game_id}/bot-move`, {
        method: "POST",
        body: JSON.stringify(optionsPayload()),
      });
      setBusy(false);
      render();
    } finally {
      setBusy(false);
    }
  }
  setStatus(isHumanTurn() ? "Your move" : "BenBot thinking...");
  setIdentity(identityForMove(state.game?.bot_move), !isHumanTurn() && !state.game?.game_over);
}

function filesForOrientation() {
  return state.game?.human_color === "black"
    ? ["h", "g", "f", "e", "d", "c", "b", "a"]
    : ["a", "b", "c", "d", "e", "f", "g", "h"];
}

function ranksForOrientation() {
  return state.game?.human_color === "black"
    ? [1, 2, 3, 4, 5, 6, 7, 8]
    : [8, 7, 6, 5, 4, 3, 2, 1];
}

function parseFenBoard(fen) {
  const boardPart = fen.split(" ")[0];
  const rows = boardPart.split("/");
  const pieces = {};
  rows.forEach((row, rowIndex) => {
    let fileIndex = 0;
    for (const char of row) {
      if (Number.isInteger(Number(char)) && char !== "0") {
        fileIndex += Number(char);
      } else {
        const file = "abcdefgh"[fileIndex];
        const rank = 8 - rowIndex;
        pieces[`${file}${rank}`] = char;
        fileIndex += 1;
      }
    }
  });
  return pieces;
}

function squarePosition(square) {
  const files = filesForOrientation();
  const ranks = ranksForOrientation();
  return {
    fileIndex: files.indexOf(square[0]),
    rankIndex: ranks.indexOf(Number(square[1])),
  };
}

function matedKingSquare(pieces) {
  if (!state.game?.checkmate) {
    return null;
  }
  const king = state.game.turn === "white" ? "K" : "k";
  return Object.entries(pieces).find(([, piece]) => piece === king)?.[0] || null;
}

function legalTargets(fromSquare) {
  return (state.game?.legal_moves || []).filter((move) => move.uci.slice(0, 2) === fromSquare);
}

function legalMoveFor(fromSquare, toSquare) {
  const matches = (state.game?.legal_moves || []).filter((move) => move.uci.slice(0, 4) === `${fromSquare}${toSquare}`);
  if (matches.length <= 1) {
    return matches[0]?.uci || null;
  }
  state.pendingPromotion = { fromSquare, toSquare, matches };
  showPromotionDialog();
  return null;
}

function isHumanTurn() {
  return state.game && !state.busy && state.game.turn === state.game.human_color && !state.game.game_over;
}

function sourceLabel(source, safety) {
  if (safety?.vetoed || source === "stockfish_veto") {
    return "Stockfish veto";
  }
  if (source === "prefix") {
    return "Prefix memory";
  }
  if (source === "fen") {
    return "Position memory";
  }
  if (source === "maia2") {
    return "Maia2";
  }
  return "Unknown";
}

function selectedMoveLabel(move) {
  return move?.san || move?.move || "-";
}

function identityForMove(botMove) {
  const selected = botMove?.selected;
  if (!selected) {
    return "Ready";
  }
  const move = selectedMoveLabel(selected);
  if (botMove.safety?.vetoed || selected.source === "stockfish_veto") {
    return `Stockfish veto selected ${move}`;
  }
  if (selected.source === "prefix") {
    return `BenBot played ${move} from prefix memory`;
  }
  if (selected.source === "fen") {
    return `BenBot played ${move} from position memory`;
  }
  if (policyOptions.mode === "sample") {
    return `BenBot sampled ${move} from Maia2`;
  }
  return `BenBot played ${move}`;
}

function renderBoard() {
  if (!state.game) {
    return;
  }

  const pieces = parseFenBoard(state.game.fen);
  const files = filesForOrientation();
  const ranks = ranksForOrientation();
  const targets = state.selected ? legalTargets(state.selected).map((move) => move.uci.slice(2, 4)) : [];
  const matedKing = matedKingSquare(pieces);

  els.board.innerHTML = "";
  ranks.forEach((rank) => {
    files.forEach((file) => {
      const squareName = `${file}${rank}`;
      const square = document.createElement("button");
      square.type = "button";
      square.className = "square";
      square.dataset.square = squareName;

      const isDark = (files.indexOf(file) + ranks.indexOf(rank)) % 2 === 1;
      square.classList.toggle("dark", isDark);
      square.classList.toggle("selected", state.selected === squareName);
      square.classList.toggle("target", targets.includes(squareName) && !pieces[squareName]);
      square.classList.toggle("capture", targets.includes(squareName) && Boolean(pieces[squareName]));
      square.classList.toggle("last-move-to", state.game.last_move?.to === squareName);
      square.classList.toggle("mated-king", matedKing === squareName);

      const piece = pieces[squareName];
      if (piece) {
        const pieceEl = document.createElement("span");
        pieceEl.className = "piece";
        pieceEl.textContent = pieceGlyphs[piece] || "";
        if (state.animatingMove && state.game.last_move?.to === squareName) {
          const from = squarePosition(state.game.last_move.from);
          const to = squarePosition(squareName);
          pieceEl.classList.add("piece-moving");
          pieceEl.style.setProperty("--move-x", `${from.fileIndex - to.fileIndex}`);
          pieceEl.style.setProperty("--move-y", `${from.rankIndex - to.rankIndex}`);
        }
        square.appendChild(pieceEl);
      }
      square.addEventListener("click", () => onSquareClick(squareName));

      if (file === files[0]) {
        const rankLabel = document.createElement("span");
        rankLabel.className = "coord rank";
        rankLabel.textContent = rank;
        square.appendChild(rankLabel);
      }
      if (rank === ranks[ranks.length - 1]) {
        const fileLabel = document.createElement("span");
        fileLabel.className = "coord file";
        fileLabel.textContent = file;
        square.appendChild(fileLabel);
      }

      els.board.appendChild(square);
    });
  });
}

function renderDetails() {
  if (!state.game) {
    return;
  }

  els.turnValue.textContent = state.game.turn;
  els.botColorValue.textContent = state.game.bot_color;

  const selected = state.game.bot_move?.selected;
  els.sourceValue.textContent = selected ? sourceLabel(selected.source, state.game.bot_move?.safety) : "-";

  const config = state.game.policy_config || policyOptions;
  els.configMode.textContent = config.mode || "-";
  els.configTopK.textContent = config.top_k ?? "-";
  els.configTemperature.textContent = config.temperature ?? "-";
  els.configStrategy.textContent = config.strategy || "-";
  els.configAlpha.textContent = config.alpha ?? "-";
  els.configMinCount.textContent = config.min_count ?? "-";
  if (!config.stockfish_enabled || config.stockfish_status === "disabled") {
    els.configStockfishVeto.textContent = "Off";
  } else if (config.stockfish_veto_cp === undefined) {
    els.configStockfishVeto.textContent = "-";
  } else {
    const suffix = config.stockfish_status && config.stockfish_status !== "ready" ? ` (${config.stockfish_status})` : "";
    els.configStockfishVeto.textContent = `${config.stockfish_veto_cp}cp${suffix}`;
  }

  if (config.stockfish_enabled && config.stockfish_status && config.stockfish_status !== "ready") {
    renderReadiness({
      status: "degraded",
      policyReady: true,
      stockfishEnabled: true,
      stockfishReady: false,
      error: config.stockfish_error || null,
    });
  }

  els.candidateList.innerHTML = "";
  const candidates = state.game.bot_move?.candidates || [];
  candidates.slice(0, 5).forEach((candidate) => {
    const item = document.createElement("li");
    item.classList.toggle("selected-candidate", selected?.move === candidate.move);
    item.style.setProperty("--probability", `${Math.max(0, Math.min(1, candidate.probability)) * 100}%`);
    const content = document.createElement("div");
    content.className = "candidate-main";
    const move = document.createElement("strong");
    move.textContent = selectedMoveLabel(candidate);
    content.appendChild(move);
    const side = document.createElement("div");
    side.className = "candidate-side";
    const probability = document.createElement("span");
    probability.className = "candidate-probability";
    probability.textContent = `${(candidate.probability * 100).toFixed(1)}%`;
    side.appendChild(probability);
    if (selected?.move === candidate.move) {
      const badge = document.createElement("span");
      badge.className = "played-badge";
      badge.textContent = "Played";
      side.appendChild(badge);
    }
    item.append(content, side);
    els.candidateList.appendChild(item);
  });

  setIdentity(identityForMove(state.game.bot_move));

  if (state.game.game_over) {
    setStatus(`Game over: ${state.game.result}`);
    showGameOverDialog();
  } else if (isHumanTurn()) {
    hideGameOverDialog();
    setStatus("Your move");
  } else {
    hideGameOverDialog();
    setStatus("Bot to move");
  }
}

function render() {
  state.animatingMove = state.game?.last_move?.move || null;
  renderBoard();
  state.animatingMove = null;
  renderDetails();
}

async function startGame() {
  if (state.busy) {
    return;
  }
  const requestId = state.requestId + 1;
  state.requestId = requestId;
  setBusy(true);
  setStatus("Loading model...");
  setIdentity("BenBot is waking up", true);
  state.selected = null;
  state.pendingPromotion = null;
  hidePromotionDialog();
  hideGameOverDialog();
  try {
    const game = await api("/new-game", {
      method: "POST",
      body: JSON.stringify({
        human_color: els.humanColor.value,
        elo_self: 1650,
        elo_oppo: 1650,
        ...optionsPayload(),
      }),
    });
    if (requestId !== state.requestId) {
      return;
    }
    state.game = game;
    setBusy(false);
    render();
    return;
  } catch (error) {
    setStatus(error.message, true);
    setIdentity("Service unavailable");
    await refreshReadiness({ quiet: true });
  } finally {
    setBusy(false);
    renderBoard();
  }
}

async function submitMove(move) {
  if (!state.game || state.busy) {
    return;
  }
  const requestId = state.requestId + 1;
  state.requestId = requestId;
  setBusy(true);
  setStatus("BenBot thinking...");
  setIdentity("BenBot is thinking", true);
  state.selected = null;
  state.pendingPromotion = null;
  hidePromotionDialog();
  try {
    const game = await api(`/game/${state.game.game_id}/move`, {
      method: "POST",
      body: JSON.stringify({ move, ...optionsPayload() }),
    });
    if (requestId !== state.requestId) {
      return;
    }
    state.game = game;
    setBusy(false);
    render();
    return;
  } catch (error) {
    if (requestId !== state.requestId) {
      return;
    }
    if (error.message.includes("not the human player's turn") || error.message.includes("Internal Server Error")) {
      try {
        await recoverTurnDesync();
      } catch {
        setStatus(error.message, true);
      }
    } else {
      setStatus(error.message, true);
      setIdentity(identityForMove(state.game?.bot_move));
      renderBoard();
    }
  } finally {
    if (requestId === state.requestId) {
      setBusy(false);
      renderBoard();
    }
  }
}

function onSquareClick(square) {
  if (state.busy || !isHumanTurn()) {
    return;
  }

  if (!state.selected) {
    if (legalTargets(square).length > 0) {
      state.pendingPromotion = null;
      hidePromotionDialog();
      state.selected = square;
      renderBoard();
    }
    return;
  }

  if (state.selected === square) {
    state.selected = null;
    state.pendingPromotion = null;
    hidePromotionDialog();
    renderBoard();
    return;
  }

  const move = legalMoveFor(state.selected, square);
  if (move) {
    submitMove(move);
    return;
  }

  if (legalTargets(square).length > 0) {
    state.pendingPromotion = null;
    hidePromotionDialog();
    state.selected = square;
    renderBoard();
  }
}

function showPromotionDialog() {
  els.promotionDialog.hidden = false;
}

function hidePromotionDialog() {
  els.promotionDialog.hidden = true;
}

function showGameOverDialog() {
  if (state.game.checkmate) {
    const winner = state.game.turn === "white" ? "Black" : "White";
    els.gameOverTitle.textContent = "Checkmate";
    els.gameOverMessage.textContent =
      winner.toLowerCase() === state.game.bot_color ? "BenBot wins by checkmate." : "You checkmated BenBot.";
  } else if (state.game.stalemate) {
    els.gameOverTitle.textContent = "Stalemate";
    els.gameOverMessage.textContent = "The game is a draw by stalemate.";
  } else {
    els.gameOverTitle.textContent = "Game Over";
    els.gameOverMessage.textContent = `Result: ${state.game.result}`;
  }
  els.gameOverDialog.hidden = false;
}

function hideGameOverDialog() {
  els.gameOverDialog.hidden = true;
}

els.promotionDialog.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-piece]");
  if (!button || !state.pendingPromotion) {
    return;
  }
  const { fromSquare, toSquare, matches } = state.pendingPromotion;
  const selected = matches.find((move) => move.uci === `${fromSquare}${toSquare}${button.dataset.piece}`);
  state.pendingPromotion = null;
  hidePromotionDialog();
  if (selected) {
    submitMove(selected.uci);
  }
});

els.newGameButton.addEventListener("click", startGame);
els.gameOverDialog.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-play-again]");
  if (!button) {
    return;
  }
  const color = button.dataset.playAgain;
  if (color === "white" || color === "black") {
    els.humanColor.value = color;
  }
  startGame();
});
els.humanColor.addEventListener("change", startGame);

async function initialize() {
  let readiness = await refreshReadiness();
  for (let attempt = 0; readiness.status === "starting" && attempt < 120; attempt += 1) {
    await delay(1_500);
    readiness = await refreshReadiness({ quiet: true });
  }
  if (readiness.status === "ready" || readiness.status === "degraded") {
    await startGame();
  } else {
    setStatus("The model could not be loaded. Try again when service status recovers.", true);
    setIdentity("Service unavailable");
  }
  window.setInterval(() => refreshReadiness({ quiet: true }), 60_000);
}

initialize();
