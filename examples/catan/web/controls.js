// Control bar: buttons, auto-search toggle, autoplay.
//
// The server drives polling and search batches via a budget model.
// RunSims adds to the budget; SetAutoSearch sets auto-refill on state
// change. The client just sends commands and renders server updates.

class Controls {
  constructor(session) {
    this.session = session;
    this.autoplay = false;
    this.autoSearch = document.getElementById('autosearch-toggle').checked;
    this._bind();
    // Tell the server our initial auto-search state.
    if (this.autoSearch) this._syncAutoSearch();
  }

  _syncAutoSearch() {
    const target = parseInt(document.getElementById('sims-input').value);
    this.session.send({ type: 'SetAutoSearch', enabled: this.autoSearch, target });
  }

  /// Called on every GameState update.
  onStateUpdate(msg) {
    if (this.pendingAutoplay && !msg.is_terminal) {
      this.pendingAutoplay = false;
      const count = parseInt(document.getElementById('sims-input').value);
      this.session.send({ type: 'RunSims', count });
      return;
    }
  }

  /// Called when a Snapshot arrives (manual RunSims completed).
  onSimsDone(snapshot) {
    if (document.getElementById('apply-toggle').checked || this.autoplay) {
      if (snapshot && snapshot.edges && snapshot.edges.length > 0) {
        const best = snapshot.edges.reduce((a, b) => b.visits > a.visits ? b : a);
        this.pendingAutoplay = this.autoplay;
        this.session.send({ type: 'PlayAction', action: best.action });
      }
    }
  }

  /// Called when BotMove completes.
  onBotDone() {}

  /// Called on server Error.
  onSearchError() {}

  onGameOver() {
    this.stopAutoplay();
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
      const sims = parseInt(document.getElementById('sims-input').value);
      this.session.send({ type: 'BotMove', simulations: sims });
    });

    document.getElementById('btn-run-sims').addEventListener('click', () => {
      const count = parseInt(document.getElementById('sims-input').value);
      this.session.send({ type: 'RunSims', count });
    });

    document.getElementById('autoplay-toggle').addEventListener('change', (e) => {
      if (e.target.checked) {
        this.startAutoplay();
      } else {
        this.stopAutoplay();
      }
    });

    document.getElementById('autosearch-toggle').addEventListener('change', (e) => {
      this.autoSearch = e.target.checked;
      this._syncAutoSearch();
    });

    // Re-sync target when the sims input changes.
    document.getElementById('sims-input').addEventListener('change', () => {
      if (this.autoSearch) this._syncAutoSearch();
    });

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
    const count = parseInt(document.getElementById('sims-input').value);
    this.session.send({ type: 'RunSims', count });
  }

  stopAutoplay() {
    this.autoplay = false;
    document.getElementById('autoplay-toggle').checked = false;
  }
}
