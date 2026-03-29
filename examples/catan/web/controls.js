// Control bar: buttons, apply/autoplay logic, auto-search loop.

const AUTO_SEARCH_BATCH = 10;
const POLL_INTERVAL_MS = 2000;

class Controls {
  constructor(session) {
    this.session = session;
    this.searching = false;
    this.autoplay = false;
    // Initialize from checkbox state (checked by default in HTML).
    this.autoSearch = document.getElementById('autosearch-toggle').checked;
    this.autoSearchTriggered = false;
    this.lastPollTime = 0;
    this.pollTimer = null;

    this._bind();
  }

  /// Schedule the next PollState, enforcing a minimum 2s interval.
  schedulePoll() {
    if (this.pollTimer) return;
    const elapsed = Date.now() - this.lastPollTime;
    const delay = Math.max(0, POLL_INTERVAL_MS - elapsed);
    this.pollTimer = setTimeout(() => {
      this.pollTimer = null;
      if (this.autoSearch && !this.searching) {
        this.lastPollTime = Date.now();
        this.session.send({ type: 'PollState' });
      }
    }, delay);
  }

  setSearching(active) {
    this.searching = active;
    const runBtn = document.getElementById('btn-run-sims');
    const botBtn = document.getElementById('btn-bot-move');
    runBtn.disabled = active;
    botBtn.disabled = active;
    runBtn.textContent = active ? 'Searching...' : 'Run Sims';
  }

  /// Called when RunSims completes (Snapshot received).
  onSimsDone(snapshot) {
    this.setSearching(false);

    if (this.autoSearchTriggered) {
      const cap = parseInt(document.getElementById('sims-input').value);
      if (snapshot && snapshot.total_simulations > 0 && snapshot.total_simulations < cap) {
        // Under cap — poll to check for state changes, then continue.
        this.schedulePoll();
      } else {
        // Hit cap or no progress — stop searching this state.
        this.autoSearchTriggered = false;
        this.schedulePoll();
      }
      return;
    }

    if (document.getElementById('apply-toggle').checked || this.autoplay) {
      // Play the most-visited action from the completed search.
      if (snapshot && snapshot.edges && snapshot.edges.length > 0) {
        const best = snapshot.edges.reduce((a, b) => b.visits > a.visits ? b : a);
        this.pendingAutoplay = this.autoplay;
        this.session.send({ type: 'PlayAction', action: best.action });
      }
    }
  }

  /// Called when BotMove completes (BotAction received).
  onBotDone() {
    this.setSearching(false);
  }

  /// Called on every GameState update.
  onStateUpdate(msg) {
    if (this.pendingAutoplay && !msg.is_terminal) {
      this.pendingAutoplay = false;
      const count = parseInt(document.getElementById('sims-input').value);
      this.setSearching(true);
      this.session.send({ type: 'RunSims', count });
      return;
    }

    // Auto-search: after a state update, kick off search or re-poll.
    if (this.autoSearch && !this.searching) {
      const canSearch = !msg.is_terminal && !msg.is_chance;
      if (canSearch) {
        this.autoSearchTriggered = true;
        this.setSearching(true);
        this.session.send({ type: 'RunSims', count: AUTO_SEARCH_BATCH });
      } else {
        // Nothing to analyze (0-1 actions, chance, or terminal) — re-poll.
        this.schedulePoll();
      }
    }
  }

  /// Called on server Error — keep auto-search alive by re-polling after a delay.
  onSearchError() {
    this.setSearching(false);
    if (this.autoSearch) {
      this.autoSearchTriggered = false;
      this.schedulePoll();
    }
  }

  _bind() {
    document.getElementById('btn-new-game').addEventListener('click', () => {
      this.stopAutoplay();
      this.session.send({ type: 'NewGame', seed: null });
    });

    document.getElementById('btn-undo').addEventListener('click', () => {
      this.stopAutoplay();
      this.session.send({ type: 'Undo' });
    });

    document.getElementById('btn-redo').addEventListener('click', () => {
      this.stopAutoplay();
      this.session.send({ type: 'Redo' });
    });

    document.getElementById('btn-bot-move').addEventListener('click', () => {
      if (this.searching) return;
      const sims = parseInt(document.getElementById('sims-input').value);
      this.setSearching(true);
      this.session.send({ type: 'BotMove', simulations: sims });
    });

    document.getElementById('btn-run-sims').addEventListener('click', () => {
      if (this.searching) return;
      const count = parseInt(document.getElementById('sims-input').value);
      this.setSearching(true);
      this.session.send({ type: 'RunSims', count });
    });

    // Auto-play toggle: continuous run sims → play → run sims → ...
    document.getElementById('autoplay-toggle').addEventListener('change', (e) => {
      if (e.target.checked) {
        this.startAutoplay();
      } else {
        this.stopAutoplay();
      }
    });

    // Auto-search toggle: continuous PollState → RunSims → PollState → ...
    document.getElementById('autosearch-toggle').addEventListener('change', (e) => {
      if (e.target.checked) {
        this.startAutoSearch();
      } else {
        this.stopAutoSearch();
      }
    });

    // Takeover buttons
    for (const btn of document.querySelectorAll('.takeover-btn')) {
      btn.addEventListener('click', () => {
        const player = parseInt(btn.dataset.player);
        const isHuman = btn.textContent.trim() === 'Take Over';
        if (isHuman) {
          this.session.send({ type: 'TakeOver', player });
          btn.textContent = 'Release';
        } else {
          this.session.send({ type: 'ReleaseControl', player });
          btn.textContent = 'Take Over';
        }
      });
    }
  }

  startAutoplay() {
    this.autoplay = true;
    // Kick off the first search.
    const count = parseInt(document.getElementById('sims-input').value);
    this.setSearching(true);
    this.session.send({ type: 'RunSims', count });
  }

  stopAutoplay() {
    this.autoplay = false;
    this.stopAutoSearch();
    document.getElementById('autoplay-toggle').checked = false;
  }

  startAutoSearch() {
    this.autoSearch = true;
    this.schedulePoll();
  }

  stopAutoSearch() {
    this.autoSearch = false;
    this.autoSearchTriggered = false;
    if (this.pollTimer) { clearTimeout(this.pollTimer); this.pollTimer = null; }
    document.getElementById('autosearch-toggle').checked = false;
  }

  onGameOver() {
    this.stopAutoplay();
  }
}
