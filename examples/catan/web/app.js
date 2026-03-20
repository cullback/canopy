// Entry point: wires components together.

const session = new Session();
const board = new Board(document.getElementById('board-svg'));
const mctsPanel = new MCTSPanel();
const controls = new Controls(session);

const RESOURCE_NAMES = ['lumber', 'brick', 'wool', 'grain', 'ore'];
const DEV_CARD_NAMES = ['Knight', 'VP', 'Road Building', 'Year of Plenty', 'Monopoly'];

let currentState = null;
let currentBoard = null;

// ── Message handlers ─────────────────────────────────────────────────

session.on('GameState', (msg) => {
  currentState = msg;
  const state = msg.state;

  // Initialize board on first state (or after new game).
  if (state.board && (!currentBoard || !board.boardData)) {
    currentBoard = state.board;
    board.initBoard(currentBoard);
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
      line.textContent = `${i + 1}. ${text}`;
      logView.appendChild(line);
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
});

session.on('Snapshot', (msg) => {
  mctsPanel.updateSnapshot(msg.snapshot, msg.action_labels);
});

session.on('Subtree', (msg) => {
  mctsPanel.showSubtree(msg.tree);
});

session.on('BotAction', (msg) => {
  if (msg.snapshot) {
    mctsPanel.updateSnapshot(msg.snapshot, []);
  }
});

session.on('Error', (msg) => {
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

  // VP
  document.getElementById(`p${idx}-vp`).textContent = pf.vp;

  // Hand
  const handEl = document.getElementById(`p${idx}-hand`);
  handEl.innerHTML = '';
  for (let r = 0; r < 5; r++) {
    if (pf.hand[r] > 0) {
      const chip = document.createElement('span');
      chip.className = `resource-chip ${RESOURCE_NAMES[r]}`;
      chip.textContent = `${pf.hand[r]} ${RESOURCE_NAMES[r]}`;
      handEl.appendChild(chip);
    }
  }

  // Dev cards
  const devEl = document.getElementById(`p${idx}-dev`);
  devEl.innerHTML = '';
  for (let d = 0; d < 5; d++) {
    if (pf.dev_cards[d] > 0) {
      const chip = document.createElement('span');
      chip.className = 'dev-chip';
      chip.textContent = `${pf.dev_cards[d]} ${DEV_CARD_NAMES[d]}`;
      devEl.appendChild(chip);
    }
  }

  // Stats
  const statsEl = document.getElementById(`p${idx}-stats`);
  const parts = [];
  if (pf.knights > 0) parts.push(`Knights: ${pf.knights}`);
  statsEl.textContent = parts.join(' | ');
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
