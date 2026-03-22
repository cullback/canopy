// MCTS analysis panel: policy bar chart, search info, tree explorer.

class MCTSPanel {
  constructor() {
    this.barsEl = document.getElementById('policy-bars');
    this.simsEl = document.getElementById('sims-count');
    this.rootQEl = document.getElementById('root-q');
    this.netValEl = document.getElementById('net-value');
    this.treeViewEl = document.getElementById('tree-view');
    this.onExplore = null;
  }

  // Update the analysis panel with a search snapshot.
  updateSnapshot(snapshot, labels) {
    this.simsEl.textContent = `${snapshot.total_simulations} sims`;
    this.rootQEl.textContent = `Q: ${snapshot.root_q.toFixed(3)}`;
    this.netValEl.textContent = `Net: ${snapshot.network_value.toFixed(3)}`;

    // Build sorted edge data
    const edges = snapshot.edges.map((e, i) => ({
      ...e,
      label: labels[i] || `Action ${e.action}`,
    }));

    // Sort by improved policy (descending), fallback to visits
    edges.sort((a, b) => {
      const ap = a.improved_policy ?? 0;
      const bp = b.improved_policy ?? 0;
      if (ap !== bp) return bp - ap;
      return b.visits - a.visits;
    });

    this.barsEl.innerHTML = '';
    const maxPolicy = Math.max(...edges.map(e => e.improved_policy ?? 0), 0.001);

    for (const edge of edges) {
      const policy = edge.improved_policy ?? 0;
      const pct = (policy / maxPolicy) * 100;
      const q = edge.q;
      const qColor = q != null ? this._qColor(q) : '#666';

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
        <span class="w-11 text-right shrink-0 text-[10px]" style="color:${qColor}">${q != null ? q.toFixed(3) : '—'}</span>
      `;

      this.barsEl.appendChild(row);
    }
  }

  // Display a subtree in the tree explorer.
  showSubtree(tree) {
    this.treeViewEl.innerHTML = '';
    this.treeViewEl.appendChild(this._renderNode(tree, 0));
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

  _renderNode(node, depth, prefix = '', isLast = true) {
    const div = document.createElement('div');

    const header = document.createElement('div');
    header.className = 'cursor-pointer py-0.5 whitespace-nowrap hover:text-accent';

    const playerPrefix = node.player != null ? `P${node.player + 1}: ` : '';
    const actionLabel = node.label || (node.action != null ? `Action ${node.action}` : 'Root');
    const visits = node.visits;
    const q = node.q.toFixed(3);
    const kind = node.kind;

    const branch = depth === 0 ? '' : (isLast ? '└─' : '├─');
    header.textContent = `${prefix}${branch}${playerPrefix}${actionLabel} [${kind}] V:${visits} Q:${q}`;
    div.appendChild(header);

    if (node.children && node.children.length > 0) {
      const childrenDiv = document.createElement('div');
      childrenDiv.className = 'tree-children';

      header.addEventListener('click', () => {
        childrenDiv.classList.toggle('open');
      });

      const childPrefix = depth === 0 ? '' : (prefix + (isLast ? '  ' : '│ '));
      const sorted = [...node.children].sort((a, b) => b.visits - a.visits);
      for (let i = 0; i < sorted.length; i++) {
        childrenDiv.appendChild(this._renderNode(sorted[i], depth + 1, childPrefix, i === sorted.length - 1));
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
