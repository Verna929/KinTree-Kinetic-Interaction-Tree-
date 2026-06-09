from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

def parse_pos2resid(pdb_path: Path) -> list[int]:
    """Read a PDB and return an ordered list mapping 0-based positional index
    -> real residue number, based on the order residues first appear."""
    pos2resid: list[int] = []
    seen: set[tuple[str, int, str]] = set()
    with open(pdb_path, "r", encoding="utf-8") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if len(line) < 27:
                continue
            chain = line[21]
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            icode = line[26]
            key = (chain, resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            pos2resid.append(resseq)
    return pos2resid


def _remap_rule_name(name: str, pos2resid: list[int]) -> str:
    """Rewrite each `N-M` pair in a rule name by mapping 0-based positional
    indices through pos2resid. Non-numeric suffixes are preserved."""
    if not name or not pos2resid:
        return name

    def sub(m: re.Match) -> str:
        a, b = int(m.group(1)), int(m.group(2))
        if 0 <= a < len(pos2resid) and 0 <= b < len(pos2resid):
            return f"{pos2resid[a]}-{pos2resid[b]}"
        print(f"[warn] index out of range in rule '{name}': {a}-{b} (pos2resid len={len(pos2resid)})")
        return m.group(0)

    return re.sub(r'(\d+)-(\d+)', sub, name)


def assign_ids_and_collect(tree, pos2resid: list[int] | None = None):
    """Convert nested tree dict into flat node/edge list."""
    nodes = []
    edges = []
    counter = 0
    pos2resid = pos2resid or []

    def walk(node, parent_id=None, branch_label=None):
        nonlocal counter
        node_id = f"n{counter}"
        counter += 1

        is_leaf = bool(node.get("leaf", False))
        raw_rule_name = node.get("rule", {}).get("name", "") if node.get("rule") else ""
        rule_name = _remap_rule_name(raw_rule_name, pos2resid)
        # Decision nodes show the rule; Leaf nodes show the Microstate ID
        label = f"Microstate {node.get('leaf_id', '')}" if is_leaf else (rule_name or "Split")

        out = {
            "id": node_id,
            "depth": int(node.get("depth", 0)),
            "n_frames": int(node.get("n_frames", 0)),
            "leaf": is_leaf,
            "leaf_id": node.get("leaf_id"),
            "rule_name": rule_name,
            "display_label": label,
        }
        nodes.append(out)
        
        if parent_id is not None:
            # Semantic labels for the decision flow
            edge_label = "Present" if branch_label == "left" else "Absent"
            edges.append({
                "id": f"e_{parent_id}_{node_id}",
                "source": parent_id,
                "target": node_id,
                "label": edge_label
            })

        if not is_leaf:
            if node.get("left") is not None:
                walk(node.get("left"), node_id, "left")
            if node.get("right") is not None:
                walk(node.get("right"), node_id, "right")

    walk(tree)
    return nodes, edges

HTML_TEMPLATE = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Kinetic Decision Tree Viewer</title>
  <script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
  <script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
  <script src="https://unpkg.com/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>
  <script src="https://unpkg.com/ngl@2.0.0-dev.39/dist/ngl.js"></script>
  <style>
    :root { --accent: #2563eb; --bg: #ffffff; --panel: #f8fafc; --border: #e2e8f0; }
    html, body { height: 100%; margin: 0; font-family: 'Inter', system-ui, sans-serif; overflow: hidden; }
    #app { display: grid; grid-template-columns: 50% 50%; height: 100vh; }
    #viewerPane { border-right: 1px solid var(--border); position: relative; }
    #treePane { display: grid; grid-template-rows: auto 1fr auto auto; background: var(--panel); }
    #viewport { width: 100%; height: 100%; }
    #toolbar { padding: 12px 20px; background: #fff; border-bottom: 1px solid var(--border); display: flex; gap: 12px; align-items: center; }
    #cy { width: 100%; height: 100%; }
    #legend { padding: 10px 20px; background: #fff; border-top: 1px solid var(--border); display: flex; flex-wrap: wrap; gap: 10px; align-items: center; font-size: 12px; color: #475569; }
    #legend.hidden { display: none; }
    .legend-swatch { width: 13px; height: 13px; border-radius: 50%; flex-shrink: 0; }
    .legend-item { display: inline-flex; align-items: center; gap: 5px; }
    #infoBox { padding: 16px 20px; background: #fff; border-top: 1px solid var(--border); font-size: 13px; height: 60px; }
    button { 
        background: #fff; border: 1px solid #cbd5e1; padding: 6px 14px; border-radius: 6px; 
        cursor: pointer; font-size: 12px; font-weight: 600; color: #475569; transition: all 0.15s;
    }
    button:hover { border-color: var(--accent); color: var(--accent); box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .label-pill { background: #f1f5f9; padding: 2px 8px; border-radius: 12px; font-family: monospace; font-weight: bold; }
  </style>
</head>
<body>
<div id="app">
  <div id="viewerPane"><div id="viewport"></div></div>
  <div id="treePane">
    <div id="toolbar">
      <button id="resetTree">Fit Tree</button>
      <button id="resetView">Reset Structure</button>
      <span style="font-size: 11px; color: #94a3b8">Click interaction boxes to highlight 3D structure</span>
    </div>
    <div id="cy"></div>
    <div id="legend" class="hidden"></div>
    <div id="infoBox"><div id="details">Select a <b>Decision Box</b> to see the structural split.</div></div>
  </div>
</div>

<script>
const TREE_NODES = __TREE_NODES__;
const TREE_EDGES = __TREE_EDGES__;
const PDB_TEXT = __PDB_TEXT__;
const MACROSTATE_MAP = __MACROSTATE_MAP__; // {leaf_id (str) -> macrostate_id (str)}
const RESIDUE_OFFSET = 0; // Rule names are already remapped to real PDB resids in Python

// --- MACROSTATE COLORS (Tableau-10, publication-ready) ---
const PALETTE = ['#4e79a7','#f28e2b','#e15759','#76b7b2','#59a14f','#edc948','#b07aa1','#ff9da7','#9c755f','#bab0ac'];
const _macroIds = [...new Set(Object.values(MACROSTATE_MAP))].sort((a, b) => String(a).localeCompare(String(b), undefined, {numeric: true}));
const MACRO_COLOR = Object.fromEntries(_macroIds.map((id, i) => [String(id), PALETTE[i % PALETTE.length]]));

function macrostateColor(leafId) {
  const macroId = MACROSTATE_MAP[String(leafId)];
  return macroId !== undefined ? MACRO_COLOR[String(macroId)] : '#cbd5e1';
}

function buildLegend() {
  const el = document.getElementById('legend');
  if (!_macroIds.length) return;
  el.classList.remove('hidden');
  el.innerHTML = '<span style="font-weight:600;margin-right:4px;">Macrostates:</span>' +
    _macroIds.map(id =>
      `<span class="legend-item">
         <span class="legend-swatch" style="background:${MACRO_COLOR[String(id)]}"></span>
         <span>${id}</span>
       </span>`
    ).join('');
}
buildLegend();

cytoscape.use(cytoscapeDagre);

// --- NGL ENGINE (PyMOL Style) ---
const stage = new NGL.Stage('viewport', { backgroundColor: 'white', antialias: true, sampleLevel: 4 });
let structureComponent = null;

const pdbBlob = new Blob([PDB_TEXT], { type: 'text/plain' });
stage.loadFile(pdbBlob, { ext: 'pdb' }).then(comp => {
  structureComponent = comp;
  applyHighQualityStyle();
  comp.autoView();
});

function applyHighQualityStyle() {
  if (!structureComponent) return;
  structureComponent.removeAllRepresentations();
  
  // Professional Ribbon Rendering
  structureComponent.addRepresentation('cartoon', { 
    colorScheme: 'chainname', 
    opacity: 0.8,
    quality: 'high',
    aspectRatio: 4.0,      // Thicker "ribbon" look
    scale: 0.8,
    subdivision: 12,       // Maximum smoothness
    radialSegments: 20
  });
}

function highlightRule(ruleName) {
  if (!structureComponent || !ruleName) return;
  applyHighQualityStyle();

  const residues = [];
  const matches = [...ruleName.matchAll(/(\d+)\s*-\s*(\d+)/g)];
  matches.forEach(m => {
    residues.push(parseInt(m[1]) + RESIDUE_OFFSET, parseInt(m[2]) + RESIDUE_OFFSET);
  });

  if (residues.length > 0) {
    const sele = residues.join(' or ');
    structureComponent.addRepresentation('licorice', {
      sele: sele,
      color: 'element',
      radius: 0.45,
      multipleBond: 'offset'
    });
    
    // Residue labels (CA only) — name + number with legible background
    structureComponent.addRepresentation('label', {
        sele: `(${sele}) and .CA`,
        labelType: 'residue',
        labelGrouping: 'residue',
        color: '#1e293b',
        fontFamily: 'sans-serif',
        fontSize: 3.0,
        fontWeight: 'bold',
        showBackground: true,
        backgroundColor: 'white',
        backgroundOpacity: 0.85
    });
    
    stage.animationControls.zoomMove(structureComponent.getCenter(sele), structureComponent.getZoom(sele) - 10, 800);
  }
}

// --- CYTOSCAPE TREE ---
const _leafFrames = TREE_NODES.filter(n => n.leaf).map(n => n.n_frames);
const _minFrames = Math.min(..._leafFrames);
const _maxFrames = Math.max(..._leafFrames);
const LEAF_MIN_PX = 28, LEAF_MAX_PX = 88;
function framesToSize(n) {
  if (_maxFrames === _minFrames) return (LEAF_MIN_PX + LEAF_MAX_PX) / 2;
  return LEAF_MIN_PX + (n - _minFrames) / (_maxFrames - _minFrames) * (LEAF_MAX_PX - LEAF_MIN_PX);
}

const cy = cytoscape({
  container: document.getElementById('cy'),
  elements: { nodes: TREE_NODES.map(n => ({ data: n })), edges: TREE_EDGES.map(e => ({ data: e })) },
  layout: { name: 'dagre', rankDir: 'TB', nodeSep: 100, rankSep: 140 },
  style: [
    {
      selector: 'node', // Decision Boxes
      style: {
        'shape': 'round-rectangle',
        'width': 'label',
        'height': 'label',
        'padding': '12px',
        'background-color': '#ffffff',
        'border-width': 2,
        'border-color': '#475569',
        'label': 'data(display_label)',
        'font-size': '15px',
        'font-weight': '600',
        'text-valign': 'center',
        'text-halign': 'center',
        'color': '#1e293b'
      }
    },
    {
      selector: 'node[?leaf]', // Microstate Circles — sized by frame count, colored by macrostate
      style: {
        'shape': 'ellipse',
        'width': ele => framesToSize(ele.data('n_frames')),
        'height': ele => framesToSize(ele.data('n_frames')),
        'background-color': ele => macrostateColor(ele.data('leaf_id')),
        'background-opacity': 0.85,
        'border-width': ele => _macroIds.length ? 2 : 0,
        'border-color': ele => macrostateColor(ele.data('leaf_id')),
        'label': 'data(display_label)',
        'font-size': '13px',
        'text-valign': 'bottom',
        'text-margin-y': 8,
        'color': '#64748b'
      }
    },
    {
      selector: 'node:selected',
      style: { 'border-color': '#2563eb', 'border-width': 4 }
    },
    {
      selector: 'edge',
      style: {
        'width': 2,
        'line-color': '#94a3b8',
        'curve-style': 'bezier',
        'target-arrow-shape': 'triangle',
        'target-arrow-color': '#94a3b8',
        'label': 'data(label)',
        'font-size': '12px',
        'text-background-color': '#f8fafc',
        'text-background-opacity': 1,
        'text-margin-y': -10
      }
    }
  ]
});

cy.on('tap', 'node', function(evt){
  const node = evt.target;
  const rule = node.data('rule_name');
  if (!node.data('leaf')) {
    document.getElementById('details').innerHTML = `Highlighted Interaction: <span class="label-pill">${rule}</span>`;
    highlightRule(rule);
  } else {
    const lid = node.data('leaf_id');
    const mid = MACROSTATE_MAP[String(lid)];
    const macroTag = mid !== undefined
      ? ` &nbsp;<span class="label-pill" style="background:${MACRO_COLOR[String(mid)]};color:#fff;">Macrostate ${mid}</span>`
      : '';
    document.getElementById('details').innerHTML =
      `<b>Microstate ${lid}</b> &mdash; ${node.data('n_frames')} frames${macroTag}`;
  }
});

document.getElementById('resetTree').addEventListener('click', () => cy.fit());
document.getElementById('resetView').addEventListener('click', () => {
    applyHighQualityStyle();
    structureComponent.autoView();
});
window.addEventListener('resize', () => stage.handleResize());
</script>
</body>
</html>
"""

def build_html(tree_json_path: Path, pdb_path: Path, out_html: Path, macrostate_path: Path | None = None):
    with open(tree_json_path, "r", encoding="utf-8") as f: tree = json.load(f)
    with open(pdb_path, "r", encoding="utf-8") as f: pdb_text = f.read()
    pos2resid = parse_pos2resid(pdb_path)
    nodes, edges = assign_ids_and_collect(tree, pos2resid)

    macrostate_map: dict[str, str] = {}
    if macrostate_path is not None:
        with open(macrostate_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        macrostate_map = {str(k): str(v) for k, v in raw.items()}

    html_text = HTML_TEMPLATE.replace("__TREE_NODES__", json.dumps(nodes))
    html_text = html_text.replace("__TREE_EDGES__", json.dumps(edges))
    html_text = html_text.replace("__PDB_TEXT__", json.dumps(pdb_text))
    html_text = html_text.replace("__PDB_NAME__", html.escape(pdb_path.name))
    html_text = html_text.replace("__MACROSTATE_MAP__", json.dumps(macrostate_map))
    out_html.write_text(html_text, encoding="utf-8")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree_json", required=True)
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--out", default="temporal_tree_viewer.html")
    parser.add_argument("--macrostates", default=None,
                        help="JSON file mapping leaf_id -> macrostate_id, e.g. {\"0\": 0, \"1\": 0, \"2\": 1}")
    args = parser.parse_args()
    build_html(Path(args.tree_json), Path(args.pdb), Path(args.out),
               Path(args.macrostates) if args.macrostates else None)
