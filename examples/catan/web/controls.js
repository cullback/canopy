// Control bar: buttons, apply/autoplay logic.

class Controls {
  constructor(session) {
    this.session = session;
    this.searching = false;
    this.autoplay = false;

    this._bind();
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
    document.getElementById('autoplay-toggle').checked = false;
  }

  onGameOver() {
    this.stopAutoplay();
  }
}
