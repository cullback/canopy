// MCTS analysis panel: policy bar chart, search info, tree explorer.

class MCTSPanel {
  constructor() {
    this.barsEl = document.getElementById('policy-bars');
    this.simsEl = document.getElementById('sims-count');
    this.rootQEl = document.getElementById('root-q');
    this.netValEl = document.getElementById('net-value');
    this.treeViewEl = document.getElementById('tree-view');
    this.onExplore = null;
    this.expandedPaths = new Set();
  }

  // Update the analysis panel with a search snapshot.
  // Reuses existing DOM rows when the edge set hasn't changed to keep
  // click handlers stable during live search updates.
  updateSnapshot(snapshot, labels, currentPlayer = 0) {
    const pvDepth = snapshot.pv_depth ?? 0;
    this.simsEl.textContent = `${snapshot.total_simulations} sims · depth ${pvDepth}`;
    const [w, d, l] = snapshot.root_wdl;
    this.rootQEl.textContent = `W ${(w*100).toFixed(0)}% D ${(d*100).toFixed(0)}% L ${(l*100).toFixed(0)}%`;
    const nv = snapshot.network_value;
    const nw = (nv + 1) / 2;
    this.netValEl.textContent = `Net: ${(nw*100).toFixed(0)}%`;

    // Build sorted edge data
    const edges = snapshot.edges.map((e, i) => ({
      ...e,
      label: labels[i] || `Action ${e.action}`,
    }));

    // Sort: visited first by visits descending, then Q as tiebreaker (flipped for P2),
    // then unvisited by policy
    const qSign = currentPlayer === 0 ? 1 : -1;
    edges.sort((a, b) => {
      const av = a.visits > 0 ? 1 : 0;
      const bv = b.visits > 0 ? 1 : 0;
      if (av !== bv) return bv - av;
      if (av && bv) {
        if (a.visits !== b.visits) return b.visits - a.visits;
        const aq = (a.q ?? 0) * qSign;
        const bq = (b.q ?? 0) * qSign;
        return bq - aq;
      }
      const ap = a.improved_policy ?? 0;
      const bp = b.improved_policy ?? 0;
      return bp - ap;
    });

    // Check if the edge set changed (different actions or count).
    const newKey = edges.map(e => e.action).join(',');
    const rebuild = newKey !== this._lastEdgeKey;
    this._lastEdgeKey = newKey;

    if (rebuild) {
      this.barsEl.innerHTML = '';
    }

    const maxPolicy = Math.max(...edges.map(e => e.improved_policy ?? 0), 0.001);

    for (let idx = 0; idx < edges.length; idx++) {
      const edge = edges[idx];
      const policy = edge.improved_policy ?? 0;
      const pct = (policy / maxPolicy) * 100;
      const q = edge.q;
      const qColor = q != null ? this._qColor(q) : '#666';

      if (rebuild) {
        const row = document.createElement('div');
        row.className = 'flex items-center gap-1 px-1.5 py-0.5 text-[11px] cursor-pointer rounded hover:bg-bg';
        row.dataset.action = edge.action;
        row.addEventListener('click', () => {
          this.onExplore?.([edge.action]);
        });

        row.innerHTML = `
          <span class="w-36 shrink-0 overflow-hidden text-ellipsis whitespace-nowrap" title="${edge.label}">${edge.label}</span>
          <div class="flex-1 h-3.5 bg-bar rounded-sm relative">
            <div class="h-full rounded-sm transition-[width] duration-150" style="width:${pct}%;background:${qColor}"></div>
          </div>
          <span class="w-10 text-right text-gray-500 shrink-0 text-[10px]">${edge.visits}</span>
          <span class="w-11 text-right shrink-0 text-[10px]" style="color:${qColor}">${q != null ? ((q + 1) / 2 * 100).toFixed(0) + '%' : '—'}</span>
          <span class="w-6 text-right text-gray-500 shrink-0 text-[10px]">${edge.depth || ''}</span>
        `;

        this.barsEl.appendChild(row);
      } else {
        // In-place update: just patch the changing values.
        const row = this.barsEl.children[idx];
        if (!row) continue;
        const bar = row.querySelector('.bg-bar > div');
        if (bar) {
          bar.style.width = `${pct}%`;
          bar.style.background = qColor;
        }
        const spans = row.querySelectorAll('span');
        if (spans[1]) spans[1].textContent = `${edge.visits}`;
        if (spans[2]) {
          spans[2].textContent = q != null ? ((q + 1) / 2 * 100).toFixed(0) + '%' : '—';
          spans[2].style.color = qColor;
        }
        if (spans[3]) spans[3].textContent = `${edge.depth || ''}`;
      }
    }
  }

  // Display a subtree in the tree explorer.
  // If the tree structure matches the current DOM, update values in-place
  // to keep click handlers and expanded state stable.
  showSubtree(tree) {
    if (this.treeViewEl.firstChild && this._updateNodeInPlace(this.treeViewEl.firstChild, tree)) {
      return; // in-place update succeeded
    }
    this.treeViewEl.innerHTML = '';
    this.treeViewEl.appendChild(this._renderNode(tree, 0, true, []));
  }

  // Try to patch an existing tree DOM node with new data.
  // Returns true if structure matched and values were updated in-place.
  _updateNodeInPlace(domNode, dataNode) {
    const header = domNode.querySelector(':scope > div:not(.tree-children)');
    if (!header) return false;

    // Check structure match: same action
    const existingAction = header.dataset.nodeAction;
    const newAction = dataNode.action != null ? String(dataNode.action) : 'root';
    if (existingAction !== newAction) return false;

    // Update header text
    const playerPrefix = dataNode.player != null ? `P${dataNode.player + 1}: ` : '';
    const actionLabel = dataNode.label || (dataNode.action != null ? `Action ${dataNode.action}` : 'Root');
    const q = (dataNode.wdl[0] - dataNode.wdl[2]).toFixed(3);
    const branch = header.dataset.branch || '';
    const prefix = header.dataset.prefix || '';
    header.textContent = `${prefix}${branch}${playerPrefix}${actionLabel} [${dataNode.kind}] V:${dataNode.visits} Q:${q}`;

    // Update children if present — match by action, not position
    const childrenDiv = domNode.querySelector(':scope > .tree-children');
    if (childrenDiv && dataNode.children) {
      const childDomMap = {};
      for (const child of childrenDiv.children) {
        const ch = child.querySelector(':scope > div:not(.tree-children)');
        if (ch?.dataset.nodeAction) childDomMap[ch.dataset.nodeAction] = child;
      }
      for (const childData of dataNode.children) {
        const key = childData.action != null ? String(childData.action) : 'root';
        const childDom = childDomMap[key];
        if (childDom) {
          this._updateNodeInPlace(childDom, childData);
        }
        // New children that don't exist in DOM are ignored (no rebuild)
      }
    }
    return true;
  }

  showProgress(done, total) {
    this.simsEl.textContent = `${done} / ${total} sims`;
  }

  clear() {
    this.barsEl.innerHTML = '';
    this.simsEl.textContent = '0 sims';
    this.rootQEl.textContent = '';
    this.netValEl.textContent = '';
    this.treeViewEl.innerHTML = '';
  }

  _renderNode(node, depth, isLast = true, path = [], prefix = '') {
    const div = document.createElement('div');
    const pathKey = path.join(',');

    const header = document.createElement('div');
    header.className = 'cursor-pointer py-0.5 whitespace-nowrap hover:text-accent';
    header.dataset.nodeAction = node.action != null ? String(node.action) : 'root';

    const playerPrefix = node.player != null ? `P${node.player + 1}: ` : '';
    const actionLabel = node.label || (node.action != null ? `Action ${node.action}` : 'Root');
    const visits = node.visits;
    const q = (node.wdl[0] - node.wdl[2]).toFixed(3);
    const kind = node.kind;

    const branch = depth === 0 ? '' : (isLast ? '└ ' : '├ ');
    header.dataset.branch = branch;
    header.dataset.prefix = prefix;
    header.textContent = `${prefix}${branch}${playerPrefix}${actionLabel} [${kind}] V:${visits} Q:${q}`;
    div.appendChild(header);

    if (node.children && node.children.length > 0) {
      const childrenDiv = document.createElement('div');
      childrenDiv.className = 'tree-children';
      if (this.expandedPaths.has(pathKey)) {
        childrenDiv.classList.add('open');
      }

      header.addEventListener('click', () => {
        childrenDiv.classList.toggle('open');
        if (childrenDiv.classList.contains('open')) {
          this.expandedPaths.add(pathKey);
        } else {
          this.expandedPaths.delete(pathKey);
        }
      });

      const sorted = [...node.children].sort((a, b) => b.visits - a.visits);
      for (let i = 0; i < sorted.length; i++) {
        const childPath = [...path, sorted[i].action];
        const childPrefix = prefix + (depth === 0 ? '' : (isLast ? '  ' : '│ '));
        childrenDiv.appendChild(this._renderNode(sorted[i], depth + 1, i === sorted.length - 1, childPath, childPrefix));
      }
      div.appendChild(childrenDiv);
    }

    return div;
  }

  // Color Q values: green for positive (good for P1), red for negative.
  _qColor(q) {
    if (q > 0.1) return '#4caf50';
    if (q > 0) return '#8bc34a';
    if (q > -0.1) return '#ff9800';
    return '#f44336';
  }
}
