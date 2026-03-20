// WebSocket client for the analysis board.
//
// Dispatches incoming messages to registered handlers.

class Session {
  constructor() {
    this.ws = null;
    this.handlers = {};
    this.connected = false;
    this.queue = [];
  }

  connect() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${location.host}/ws`;
    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      this.connected = true;
      for (const msg of this.queue) {
        this.ws.send(msg);
      }
      this.queue = [];
    };

    this.ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      const handler = this.handlers[msg.type];
      if (handler) handler(msg);
    };

    this.ws.onclose = () => {
      this.connected = false;
      setTimeout(() => this.connect(), 2000);
    };

    this.ws.onerror = () => {
      this.ws.close();
    };
  }

  send(msg) {
    const json = JSON.stringify(msg);
    if (this.connected) {
      this.ws.send(json);
    } else {
      this.queue.push(json);
    }
  }

  on(type, handler) {
    this.handlers[type] = handler;
  }
}
