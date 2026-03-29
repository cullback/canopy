// SVG hex board rendering for Catan.
//
// Renders tiles, buildings, roads, robber, ports, and legal-action overlays.

const TERRAIN_COLORS = {
  forest: '#2d5a27',
  hills: '#b85c38',
  pasture: '#7ec850',
  fields: '#e8b430',
  mountains: '#7a7a7a',
  desert: '#d4c088',
};

const HEX_SIZE = 50;
const SQRT3 = Math.sqrt(3);

class Board {
  constructor(svgEl) {
    this.svg = svgEl;
    this.boardData = null;
    this.onActionClick = null;
  }

  // Initial render from board topology (tiles, nodes, edges, ports).
  initBoard(board) {
    this.boardData = board;
    this._centroid = null;
    this.svg.innerHTML = '';

    // Compute viewBox from node positions
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const [x, y] of board.nodes) {
      minX = Math.min(minX, x); minY = Math.min(minY, y);
      maxX = Math.max(maxX, x); maxY = Math.max(maxY, y);
    }
    const pad = 40;
    this.svg.setAttribute('viewBox',
      `${minX - pad} ${minY - pad} ${maxX - minX + 2 * pad} ${maxY - minY + 2 * pad}`);

    // Ocean background
    this.svg.appendChild(this._el('rect', {
      x: minX - pad, y: minY - pad,
      width: maxX - minX + 2 * pad, height: maxY - minY + 2 * pad,
      fill: '#1a5276', rx: 10
    }));

    // Draw tiles
    const tilesG = this._g('tiles');
    for (const tile of board.tiles) {
      this._drawTile(tilesG, tile);
    }

    // Draw edges (roads layer - empty initially)
    this._g('roads');

    // Draw ports
    const portsG = this._g('ports');
    for (const port of board.ports) {
      this._drawPort(portsG, port, board.nodes);
    }

    // Draw nodes (buildings layer - empty initially)
    this._g('buildings');

    // Robber layer
    this._g('robber');

    // Persistent hover targets for tiles, edges, and nodes
    const hitsG = this._g('hit-targets');
    for (let tid = 0; tid < board.tiles.length; tid++) {
      const tile = board.tiles[tid];
      if (!tile) continue;
      const el = this._el('circle', {
        cx: tile.cx, cy: tile.cy, r: 18,
        fill: 'transparent', 'pointer-events': 'all'
      });
      this._attachTooltip(el, `T${tid}`);
      hitsG.appendChild(el);
    }
    for (let eid = 0; eid < board.edges.length; eid++) {
      const edge = board.edges[eid];
      if (!edge) continue;
      const [n0, n1] = edge;
      const [x0, y0] = board.nodes[n0];
      const [x1, y1] = board.nodes[n1];
      const el = this._el('line', {
        x1: x0, y1: y0, x2: x1, y2: y1,
        stroke: 'transparent', 'stroke-width': 6,
        'stroke-linecap': 'round', 'pointer-events': 'stroke'
      });
      this._attachTooltip(el, `E${eid}`);
      hitsG.appendChild(el);
    }
    for (let nid = 0; nid < board.nodes.length; nid++) {
      const [x, y] = board.nodes[nid];
      const el = this._el('circle', {
        cx: x, cy: y, r: 7,
        fill: 'transparent', 'pointer-events': 'all'
      });
      this._attachTooltip(el, `N${nid}`);
      hitsG.appendChild(el);
    }

    // Overlay for legal actions
    this._g('overlays');
  }

  // Update dynamic elements from a frame.
  updateFrame(frame, board) {
    if (!this.boardData) return;
    const nodes = board.nodes;

    // Roads
    const roadsG = this.svg.querySelector('.roads');
    roadsG.innerHTML = '';
    for (let p = 0; p < 2; p++) {
      const color = p === 0 ? '#4a9eff' : '#ff6b6b';
      for (const eid of frame.buildings[p].roads) {
        const edge = board.edges[eid];
        if (!edge) continue;
        const [n0, n1] = edge;
        const [x0, y0] = nodes[n0];
        const [x1, y1] = nodes[n1];
        const line = this._el('line', {
          x1: x0, y1: y0, x2: x1, y2: y1,
          stroke: color, 'stroke-width': 4, 'stroke-linecap': 'round'
        });
        roadsG.appendChild(line);
      }
    }

    // Buildings
    const buildG = this.svg.querySelector('.buildings');
    buildG.innerHTML = '';
    for (let p = 0; p < 2; p++) {
      const color = p === 0 ? '#4a9eff' : '#ff6b6b';
      for (const nid of frame.buildings[p].settlements) {
        const [x, y] = nodes[nid];
        buildG.appendChild(this._el('rect', {
          x: x - 5, y: y - 5, width: 10, height: 10,
          fill: color, stroke: '#111', 'stroke-width': 1
        }));
      }
      for (const nid of frame.buildings[p].cities) {
        const [x, y] = nodes[nid];
        // House shape: pointed roof on a wider base
        const pts = `${x},${y-9} ${x+7},${y-3} ${x+7},${y+7} ${x-7},${y+7} ${x-7},${y-3}`;
        buildG.appendChild(this._el('polygon', {
          points: pts,
          fill: color, stroke: '#111', 'stroke-width': 1
        }));
      }
    }

    // Robber
    const robberG = this.svg.querySelector('.robber');
    robberG.innerHTML = '';
    if (board.tiles[frame.robber]) {
      const tile = board.tiles[frame.robber];
      robberG.appendChild(this._el('circle', {
        cx: tile.cx, cy: tile.cy - 18, r: 8,
        fill: '#111', stroke: '#e94560', 'stroke-width': 2
      }));
    }
  }

  // Show legal action overlays on the board.
  showLegalActions(actions, board) {
    const overlayG = this.svg.querySelector('.overlays');
    overlayG.innerHTML = '';
    if (!board) return;

    for (const { action, label } of actions) {
      const overlay = this._actionOverlay(action, label, board);
      if (overlay) overlayG.appendChild(overlay);
    }
  }

  clearOverlays() {
    const overlayG = this.svg.querySelector('.overlays');
    if (overlayG) overlayG.innerHTML = '';
  }

  // Attach instant tooltip (replaces slow browser-native <title>).
  _attachTooltip(el, label) {
    const tip = document.getElementById('svg-tooltip');
    el.addEventListener('mouseenter', (e) => {
      tip.textContent = label;
      tip.style.display = 'block';
      tip.style.left = e.clientX + 10 + 'px';
      tip.style.top = e.clientY + 10 + 'px';
    });
    el.addEventListener('mousemove', (e) => {
      tip.style.left = e.clientX + 10 + 'px';
      tip.style.top = e.clientY + 10 + 'px';
    });
    el.addEventListener('mouseleave', () => {
      tip.style.display = 'none';
    });
  }

  // Create a clickable overlay element for an action.
  _actionOverlay(action, label, board) {
    const nodes = board.nodes;
    // Settlement: action 0..54 -> node
    if (action < 54) {
      const [x, y] = nodes[action];
      const el = this._el('circle', {
        cx: x, cy: y, r: 8,
        fill: 'rgba(255,255,255,0.3)', stroke: '#fff',
        'stroke-width': 1.5, cursor: 'pointer', class: 'action-overlay'
      });
      el.dataset.action = action;
      el.addEventListener('click', () => this.onActionClick?.(action));
      this._attachTooltip(el, label);
      return el;
    }
    // Road: 54..126 -> edge
    if (action >= 54 && action < 126) {
      const eid = action - 54;
      const edge = board.edges[eid];
      if (!edge) return null;
      const [n0, n1] = edge;
      const [x0, y0] = nodes[n0];
      const [x1, y1] = nodes[n1];
      const el = this._el('line', {
        x1: x0, y1: y0, x2: x1, y2: y1,
        stroke: 'rgba(255,255,255,0.5)', 'stroke-width': 6,
        'stroke-linecap': 'round', cursor: 'pointer', class: 'action-overlay'
      });
      el.dataset.action = action;
      el.addEventListener('click', () => this.onActionClick?.(action));
      this._attachTooltip(el, label);
      return el;
    }
    // City: 126..180 -> node
    if (action >= 126 && action < 180) {
      const nid = action - 126;
      const [x, y] = nodes[nid];
      const el = this._el('circle', {
        cx: x, cy: y, r: 10,
        fill: 'rgba(255,255,100,0.3)', stroke: '#ff0',
        'stroke-width': 1.5, cursor: 'pointer', class: 'action-overlay'
      });
      el.dataset.action = action;
      el.addEventListener('click', () => this.onActionClick?.(action));
      this._attachTooltip(el, label);
      return el;
    }
    // Robber: 205..224 -> tile
    if (action >= 205 && action < 224) {
      const tid = action - 205;
      const tile = board.tiles[tid];
      if (!tile) return null;
      const el = this._el('circle', {
        cx: tile.cx, cy: tile.cy, r: 15,
        fill: 'rgba(233,69,96,0.3)', stroke: '#e94560',
        'stroke-width': 2, cursor: 'pointer', class: 'action-overlay'
      });
      el.dataset.action = action;
      el.addEventListener('click', () => this.onActionClick?.(action));
      this._attachTooltip(el, label);
      return el;
    }
    return null;
  }

  _drawTile(parent, tile) {
    const { cx, cy, terrain, number, nodes: tileNodes } = tile;
    const color = TERRAIN_COLORS[terrain] || '#444';

    // Hex path
    let points = '';
    for (let i = 0; i < 6; i++) {
      const angle = (Math.PI / 3) * i - Math.PI / 6;
      const px = cx + HEX_SIZE * Math.cos(angle);
      const py = cy + HEX_SIZE * Math.sin(angle);
      points += `${px},${py} `;
    }
    parent.appendChild(this._el('polygon', {
      points: points.trim(),
      fill: color, stroke: '#111', 'stroke-width': 1
    }));

    // Number token (shifted down so robber doesn't fully obscure it)
    if (number) {
      const isRed = number === 6 || number === 8;
      parent.appendChild(this._el('circle', {
        cx, cy, r: 12,
        fill: '#f5f0e1', stroke: '#333', 'stroke-width': 0.5
      }));
      const txt = this._el('text', {
        x: cx, y: cy + 4.5,
        'text-anchor': 'middle', 'font-size': '12',
        'font-weight': isRed ? 'bold' : 'normal',
        fill: isRed ? '#c00' : '#333'
      });
      txt.textContent = number;
      parent.appendChild(txt);
    }
  }

  _drawPort(parent, port, nodes) {
    const PORT_COLORS = {
      lumber: '#2d5a27', brick: '#b85c38', wool: '#7ec850',
      grain: '#e8b430', ore: '#7a7a7a', generic: '#ffffff',
    };
    const [n0, n1] = port.nodes;
    const [x0, y0] = nodes[n0];
    const [x1, y1] = nodes[n1];
    const mx = (x0 + x1) / 2;
    const my = (y0 + y1) / 2;
    const color = PORT_COLORS[port.kind] || PORT_COLORS.generic;
    const ratio = port.kind === 'generic' ? '3:1' : '2:1';
    const darkText = port.kind === 'grain' || port.kind === 'wool' || port.kind === 'generic';

    // Normal perpendicular to the port edge, pointing outward
    if (!this._centroid) {
      let sx = 0, sy = 0;
      for (const [x, y] of nodes) { sx += x; sy += y; }
      this._centroid = [sx / nodes.length, sy / nodes.length];
    }
    const [cx, cy] = this._centroid;
    // Edge direction and its perpendicular
    const ex = x1 - x0, ey = y1 - y0;
    let nx = -ey, ny = ex;
    // Pick the normal pointing away from board center
    const toCenterX = cx - mx, toCenterY = cy - my;
    if (nx * toCenterX + ny * toCenterY > 0) { nx = -nx; ny = -ny; }
    const nlen = Math.sqrt(nx * nx + ny * ny) || 1;
    const offset = 18;
    const lx = mx + nx / nlen * offset;
    const ly = my + ny / nlen * offset;

    // Circle label with ratio
    const isGeneric = port.kind === 'generic';
    parent.appendChild(this._el('circle', {
      cx: lx, cy: ly, r: 10,
      fill: isGeneric ? '#334' : color,
      stroke: isGeneric ? color : 'none', 'stroke-width': 1.5,
      opacity: 0.85
    }));
    const txt = this._el('text', {
      x: lx, y: ly + 3,
      'text-anchor': 'middle', 'font-size': '8',
      fill: isGeneric ? '#ddd' : (darkText ? '#222' : '#fff'),
      'font-weight': '600'
    });
    txt.textContent = ratio;
    parent.appendChild(txt);
  }

  _g(cls) {
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttribute('class', cls);
    this.svg.appendChild(g);
    return g;
  }

  _el(tag, attrs) {
    const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [k, v] of Object.entries(attrs)) {
      el.setAttribute(k, v);
    }
    return el;
  }
}
