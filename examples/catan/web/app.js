// Entry point: wires components together.

const session = new Session();
const board = new Board(document.getElementById('board-svg'));
const mctsPanel = new MCTSPanel();
const controls = new Controls(session);

const RESOURCE_NAMES = ['lumber', 'brick', 'wool', 'grain', 'ore'];
const DEV_CARD_NAMES = ['Knight', 'VP', 'Road Building', 'Year of Plenty', 'Monopoly'];
const DEV_SHORT = ['Kn', 'VP', 'RB', 'YP', 'Mo'];

let currentState = null;
let currentBoard = null;

// ── Message handlers ─────────────────────────────────────────────────

session.on('GameState', (msg) => {
  currentState = msg;
  const state = msg.state;

  // Initialize board on first state (or after new game).
  if (state.board) {
    const boardChanged = !currentBoard ||
      JSON.stringify(state.board) !== JSON.stringify(currentBoard);
    if (boardChanged) {
      currentBoard = state.board;
      board.initBoard(currentBoard);
    }
  }

  // Update frame
  if (state.frame && currentBoard) {
    board.updateFrame(state.frame, currentBoard);
  }

  // Phase / turn
  document.getElementById('phase-label').textContent = msg.phase;
  document.getElementById('turn-label').textContent = state.turn != null ? `Turn ${state.turn}` : '';

  // Player panels
  updatePlayerPanel(0, state);
  updatePlayerPanel(1, state);
  updateBank(state);
  updateDice(state);

  // Highlight active player
  document.getElementById('player-0').classList.toggle('active', msg.current_player === 0);
  document.getElementById('player-1').classList.toggle('active', msg.current_player === 1);

  // Legal actions
  const actionList = document.getElementById('action-list');
  actionList.innerHTML = '';
  board.clearOverlays();

  if (!msg.is_terminal && !msg.is_chance) {
    // Board overlays for spatial actions
    if (currentBoard) {
      board.showLegalActions(msg.legal_actions, currentBoard);
    }

    // Button list for all actions
    for (const a of msg.legal_actions) {
      const btn = document.createElement('button');
      btn.className = 'py-1 px-2 text-[11px] bg-bg-3 border border-gray-700 text-gray-200 rounded cursor-pointer whitespace-nowrap hover:bg-accent transition-colors';
      btn.textContent = a.label;
      btn.addEventListener('click', () => {
        session.send({ type: 'PlayAction', action: a.action });
      });
      actionList.appendChild(btn);
    }
  }

  // Game log
  const logView = document.getElementById('log-view');
  logView.innerHTML = '';
  if (msg.action_log) {
    for (let i = 0; i < msg.action_log.length; i++) {
      const line = document.createElement('div');
      line.className = 'py-0.5';
      const text = msg.action_log[i];
      if (text.startsWith('P1:')) {
        line.style.color = '#4a9eff';
      } else if (text.startsWith('P2:')) {
        line.style.color = '#ff6b6b';
      } else {
        line.style.color = '#a0a0a0';
        line.style.fontStyle = 'italic';
      }
      const parts = text.split('\n');
      line.textContent = `${i + 1}. ${parts[0]}`;
      logView.appendChild(line);
      for (let p = 1; p < parts.length; p++) {
        const sub = document.createElement('div');
        sub.style.color = '#888';
        sub.style.fontSize = '0.85em';
        sub.style.paddingLeft = '1.5em';
        sub.textContent = parts[p].trim();
        logView.appendChild(sub);
      }
    }
    logView.scrollTop = logView.scrollHeight;
  }

  // Undo/Redo button states
  document.getElementById('btn-undo').disabled = !msg.can_undo;
  document.getElementById('btn-redo').disabled = !msg.can_redo;

  // Result banner
  const banner = document.getElementById('result-banner');
  if (msg.is_terminal && msg.result) {
    banner.textContent = msg.result;
    banner.classList.remove('hidden');
    controls.onGameOver();
  } else {
    banner.classList.add('hidden');
  }

  controls.onStateUpdate(msg);
});

session.on('Snapshot', (msg) => {
  mctsPanel.updateSnapshot(msg.snapshot, msg.action_labels);
  controls.onSimsDone(msg.snapshot);
});

session.on('Subtree', (msg) => {
  mctsPanel.showSubtree(msg.tree);
});

session.on('SearchProgress', (msg) => {
  mctsPanel.updateSnapshot(msg.snapshot, msg.action_labels);
  mctsPanel.showProgress(msg.snapshot.total_simulations, msg.sims_total);
});

session.on('BotAction', (msg) => {
  if (msg.snapshot) {
    mctsPanel.updateSnapshot(msg.snapshot, msg.action_labels || []);
  }
  controls.onBotDone();
});

session.on('Error', (msg) => {
  controls.setSearching(false);
  console.error('Server error:', msg.message);
});

// ── Board action clicks ──────────────────────────────────────────────

board.onActionClick = (action) => {
  session.send({ type: 'PlayAction', action });
};

// ── MCTS explore ─────────────────────────────────────────────────────

mctsPanel.onExplore = (actionPath) => {
  session.send({ type: 'ExploreSubtree', action_path: actionPath, depth: 3 });
};

// ── Player panel helpers ─────────────────────────────────────────────

function updatePlayerPanel(idx, state) {
  const frame = state.frame;
  if (!frame) return;

  const pf = frame.players[idx];
  if (!pf) return;

  // Player name
  if (state.player_names && state.player_names[idx]) {
    document.getElementById(`p${idx}-name`).textContent = state.player_names[idx];
  }

  // VP
  document.getElementById(`p${idx}-vp`).textContent = pf.vp;

  // Hand — always show all 5 resources as colored rectangles
  const handEl = document.getElementById(`p${idx}-hand`);
  handEl.innerHTML = '';
  for (let r = 0; r < 5; r++) {
    const card = document.createElement('span');
    card.className = `resource-card ${RESOURCE_NAMES[r]}`;
    card.textContent = pf.hand[r];
    handEl.appendChild(card);
  }

  // Dev cards
  const devEl = document.getElementById(`p${idx}-dev`);
  devEl.innerHTML = '';
  for (let d = 0; d < 5; d++) {
    if (pf.dev_cards[d] > 0) {
      const chip = document.createElement('span');
      const bought = pf.dev_cards_bought_this_turn ? pf.dev_cards_bought_this_turn[d] : 0;
      const playable = pf.dev_cards[d] - bought;
      if (playable > 0) {
        chip.className = 'dev-chip';
        chip.textContent = `${playable} ${DEV_CARD_NAMES[d]}`;
        devEl.appendChild(chip);
      }
      if (bought > 0) {
        const bchip = document.createElement('span');
        bchip.className = 'dev-chip bought-this-turn';
        bchip.textContent = `${bought} ${DEV_CARD_NAMES[d]}`;
        bchip.title = 'Bought this turn — cannot play yet';
        devEl.appendChild(bchip);
      }
    }
  }
  if (pf.hidden_dev_cards > 0) {
    const est = state.expected_dev && state.expected_dev[idx];
    const hasEstimate = est && est.some(v => v > 0);
    if (hasEstimate) {
      // Show hypergeometric expected distribution
      for (let d = 0; d < 5; d++) {
        if (est[d] >= 0.05) {
          const chip = document.createElement('span');
          chip.className = 'dev-chip dev-estimate';
          chip.textContent = `~${est[d].toFixed(1)} ${DEV_SHORT[d]}`;
          devEl.appendChild(chip);
        }
      }
    } else {
      const chip = document.createElement('span');
      chip.className = 'dev-chip hidden-dev';
      chip.textContent = `${pf.hidden_dev_cards} unknown`;
      devEl.appendChild(chip);
    }
  }

  // Stats: knights played + awards
  const statsEl = document.getElementById(`p${idx}-stats`);
  const parts = [];
  parts.push(`Knights: ${pf.knights}`);
  if (frame.longest_road && frame.longest_road[0] === idx) {
    parts.push(`Longest Road: ${frame.longest_road[1]}`);
  }
  if (frame.largest_army && frame.largest_army[0] === idx) {
    parts.push(`Largest Army: ${frame.largest_army[1]}`);
  }
  statsEl.textContent = parts.join(' · ');
}

function updateBank(state) {
  const bankEl = document.getElementById('bank-dev');
  if (!bankEl) return;
  bankEl.innerHTML = '';

  // Use hypergeometric estimate when available (colonist mode — bank is unknown).
  const est = state.expected_bank_dev;
  const hasEstimate = est && est.some(v => v > 0);

  if (hasEstimate) {
    for (let d = 0; d < 5; d++) {
      if (est[d] >= 0.05) {
        const chip = document.createElement('span');
        chip.className = 'dev-chip dev-estimate';
        chip.textContent = `~${est[d].toFixed(1)} ${DEV_SHORT[d]}`;
        bankEl.appendChild(chip);
      }
    }
  } else {
    // Self-play: show exact pool counts.
    const pool = state.frame && state.frame.dev_pool;
    if (!pool) return;
    for (let d = 0; d < 5; d++) {
      if (pool[d] > 0) {
        const chip = document.createElement('span');
        chip.className = 'dev-chip';
        chip.textContent = `${pool[d]} ${DEV_SHORT[d]}`;
        bankEl.appendChild(chip);
      }
    }
    if (pool.every(v => v === 0)) {
      bankEl.textContent = 'Empty';
    }
  }
}

// ── Dice probability chart ───────────────────────────────────────────

// Fair 2d6 probabilities for reference line.
const FAIR_PROBS = [1,2,3,4,5,6,5,4,3,2,1].map(v => v / 36);

function updateDice(state) {
  const panel = document.getElementById('dice-panel');
  const dice = state.dice;
  if (!dice) { panel.style.display = 'none'; return; }
  panel.style.display = '';

  document.getElementById('dice-cards').textContent = `(${dice.cards_left}/${dice.total_cards})`;

  const chart = document.getElementById('dice-chart');
  chart.innerHTML = '';

  // Fixed scale: fair 7 probability (~16.7%) maps to ~75% of chart height,
  // leaving headroom for sums that exceed fair odds.
  const scale = FAIR_PROBS[5] / 0.75; // denominator so fair-7 bar is 75% tall
  const chartH = 64;

  for (let i = 0; i < 11; i++) {
    const sum = i + 2;
    const prob = dice.probs[i];
    const fair = FAIR_PROBS[i];

    const col = document.createElement('div');
    col.className = 'dice-col';
    col.style.flex = '1';
    col.style.display = 'flex';
    col.style.flexDirection = 'column';
    col.style.alignItems = 'center';
    col.style.justifyContent = 'flex-end';
    col.style.height = chartH + 'px';
    col.style.position = 'relative';

    // Fair probability reference tick
    const ref = document.createElement('div');
    ref.style.position = 'absolute';
    const labelH = 14;
    ref.style.bottom = (labelH + Math.round(fair / scale * (chartH - labelH))) + 'px';
    ref.style.width = '100%';
    ref.style.height = '1px';
    ref.style.background = '#555';
    col.appendChild(ref);

    // Actual probability bar
    const bar = document.createElement('div');
    const h = Math.max(1, Math.min(chartH - labelH, Math.round(prob / scale * (chartH - labelH))));
    bar.style.width = '80%';
    bar.style.height = h + 'px';
    bar.style.borderRadius = '1px';
    const ratio = fair > 0 ? prob / fair : 1;
    if (ratio > 1.15) {
      bar.style.background = '#e94560'; // above fair — red/hot
    } else if (ratio < 0.85) {
      bar.style.background = '#2a6';    // below fair — green/cold
    } else {
      bar.style.background = '#4a9eff'; // near fair — blue
    }
    col.appendChild(bar);

    // Tooltip on the whole column
    col.title = `Roll ${sum}: ${(prob * 100).toFixed(1)}% (fair: ${(fair * 100).toFixed(1)}%)`;

    // Label
    const lbl = document.createElement('div');
    lbl.style.fontSize = '9px';
    lbl.style.color = sum === 7 ? '#e94560' : '#888';
    lbl.style.marginTop = '2px';
    lbl.textContent = sum;
    col.appendChild(lbl);

    chart.appendChild(col);
  }
}

// ── Keyboard shortcuts ────────────────────────────────────────────────

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === 'ArrowLeft' && currentState?.can_undo) {
    e.preventDefault();
    controls.stopAutoplay();
    session.send({ type: 'Undo' });
  } else if (e.key === 'ArrowRight' && currentState?.can_redo) {
    e.preventDefault();
    controls.stopAutoplay();
    session.send({ type: 'Redo' });
  }
});

// ── Start ────────────────────────────────────────────────────────────

session.connect();
