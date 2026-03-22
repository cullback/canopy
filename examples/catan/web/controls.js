// Control bar: buttons, sliders, autoplay logic.

class Controls {
  constructor(session) {
    this.session = session;
    this.autoplayInterval = null;
    this.autoplayDelay = 500;
    this.searching = false;

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
      const sims = parseInt(document.getElementById('sims-slider').value);
      this.setSearching(true);
      this.session.send({ type: 'BotMove', simulations: sims || null });
    });

    document.getElementById('btn-run-sims').addEventListener('click', () => {
      if (this.searching) return;
      const count = parseInt(document.getElementById('sims-slider').value);
      this.setSearching(true);
      this.session.send({ type: 'RunSims', count });
    });

    // Sims slider
    const simsSlider = document.getElementById('sims-slider');
    const simsValue = document.getElementById('sims-value');
    simsSlider.addEventListener('input', () => {
      simsValue.textContent = simsSlider.value;
    });

    // Speed slider
    const speedSlider = document.getElementById('speed-slider');
    const speedValue = document.getElementById('speed-value');
    speedSlider.addEventListener('input', () => {
      this.autoplayDelay = parseInt(speedSlider.value);
      speedValue.textContent = `${this.autoplayDelay}ms`;
      if (this.autoplayInterval) {
        this.stopAutoplay();
        this.startAutoplay();
      }
    });

    // Autoplay toggle
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
    if (this.autoplayInterval) return;
    const tick = () => {
      const sims = parseInt(document.getElementById('sims-slider').value);
      this.setSearching(true);
      this.session.send({ type: 'BotMove', simulations: sims || null });
    };
    tick();
    this.autoplayInterval = setInterval(tick, this.autoplayDelay);
  }

  stopAutoplay() {
    if (this.autoplayInterval) {
      clearInterval(this.autoplayInterval);
      this.autoplayInterval = null;
    }
    document.getElementById('autoplay-toggle').checked = false;
  }

  // Stop autoplay when game ends.
  onGameOver() {
    this.stopAutoplay();
  }
}
