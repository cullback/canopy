// Control bar: buttons, sliders, autoplay logic.

class Controls {
  constructor(session) {
    this.session = session;
    this.autoplayInterval = null;
    this.autoplayDelay = 500;

    this._bind();
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
      const sims = parseInt(document.getElementById('sims-slider').value);
      this.session.send({ type: 'BotMove', simulations: sims || null });
    });

    document.getElementById('btn-run-sims').addEventListener('click', () => {
      const count = parseInt(document.getElementById('sims-slider').value);
      this.session.send({ type: 'RunSims', count });
    });

    document.getElementById('btn-snapshot').addEventListener('click', () => {
      this.session.send({ type: 'GetSnapshot' });
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
