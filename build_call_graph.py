# /home/claude/build_call_graph.py
"""
Static call graph generator using AST analysis.
Produces a Mermaid diagram of function calls with arguments.
No imports required — pure static analysis.
"""
import ast
import sys
from pathlib import Path
from collections import defaultdict

def parse_arg(node):
    """Convert an AST argument node to a readable string."""
    if isinstance(node, ast.Constant):
        return repr(node.value)
    elif isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return f"{parse_arg(node.value)}.{node.attr}"
    elif isinstance(node, ast.Call):
        return f"{parse_arg(node.func)}(...)"
    elif isinstance(node, ast.Subscript):
        return f"{parse_arg(node.value)}[...]"
    elif isinstance(node, ast.List):
        return "[...]"
    elif isinstance(node, ast.Dict):
        return "{...}"
    elif isinstance(node, ast.UnaryOp):
        return "..."
    else:
        return "..."

def parse_call_args(call_node):
    """Extract positional and keyword args from a Call node."""
    parts = []
    for arg in call_node.args:
        parts.append(parse_arg(arg))
    for kw in call_node.keywords:
        parts.append(f"{kw.arg}={parse_arg(kw.value)}")
    return parts

class CallGraphVisitor(ast.NodeVisitor):
    def __init__(self, module_name):
        self.module_name = module_name
        self.current_function = None
        self.functions = {}      # func_name -> {args, calls: [(callee, [args])]
        self.top_level_calls = []  # calls made outside any function

    def visit_FunctionDef(self, node):
        parent = self.current_function
        self.current_function = node.name
        if node.name not in self.functions:
            func_args = [a.arg for a in node.args.args]
            self.functions[node.name] = {"args": func_args, "calls": []}
        self.generic_visit(node)
        self.current_function = parent

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node):
        callee = None
        if isinstance(node.func, ast.Name):
            callee = node.func.id
        elif isinstance(node.func, ast.Attribute):
            callee = f"{parse_arg(node.func.value)}.{node.func.attr}"

        if callee:
            call_args = parse_call_args(node)
            if self.current_function:
                self.functions[self.current_function]["calls"].append((callee, call_args))
            else:
                self.top_level_calls.append((callee, call_args))

        self.generic_visit(node)


def analyze_file(filepath):
    try:
        src = Path(filepath).read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None, None
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None, None
    module_name = Path(filepath).stem
    visitor = CallGraphVisitor(module_name)
    visitor.visit(tree)
    return module_name, visitor


def build_graph(files):
    modules = {}
    all_functions = {}  # func_name -> module_name (for cross-module lookup)

    for f in files:
        module_name, visitor = analyze_file(f)
        if module_name is None:
            continue
        modules[module_name] = visitor
        for func_name in visitor.functions:
            all_functions[func_name] = module_name

    return modules, all_functions


def to_mermaid(modules, all_functions, focus_public=True):
    lines = ["graph TD"]
    lines.append("    %% Auto-generated call graph")
    lines.append("")

    seen_edges = set()
    node_ids = {}

    def node_id(module, func):
        key = f"{module}__{func}"
        if key not in node_ids:
            node_ids[key] = f"{module}__{func}".replace("-", "_")
        return node_ids[key]

    # Subgraph per module
    for module_name, visitor in modules.items():
        lines.append(f"    subgraph {module_name}")
        for func_name, info in visitor.functions.items():
            if focus_public and func_name.startswith("__"):
                continue
            nid = node_id(module_name, func_name)
            arg_str = ", ".join(info["args"][:4])  # cap at 4 args for readability
            if len(info["args"]) > 4:
                arg_str += ", ..."
            label = f"{func_name}({arg_str})"
            lines.append(f'        {nid}["{label}"]')
        lines.append("    end")
        lines.append("")

    # Edges — only draw cross-module or intra-module calls to known functions
    for module_name, visitor in modules.items():
        for caller_func, info in visitor.functions.items():
            if focus_public and caller_func.startswith("__"):
                continue
            caller_nid = node_id(module_name, caller_func)

            for callee, call_args in info["calls"]:
                # Only draw edges to functions we know about
                callee_module = all_functions.get(callee)
                if not callee_module:
                    continue
                if focus_public and callee.startswith("__"):
                    continue

                callee_nid = node_id(callee_module, callee)
                edge_key = (caller_nid, callee_nid)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)

                # Truncate args for label
                arg_label = ", ".join(call_args[:3])
                if len(call_args) > 3:
                    arg_label += ", ..."
                if arg_label:
                    lines.append(f'    {caller_nid} -->|"{arg_label}"| {callee_nid}')
                else:
                    lines.append(f"    {caller_nid} --> {callee_nid}")

    return "\n".join(lines)


EXCLUDE_DIRS = {
    "venv", ".venv", "env", "__pycache__", "migrations",
    "node_modules", ".git", "dist", "build", ".eggs",
}

# Directories relative to the project root that contain source code.
# Edit this list to match your project layout.
SOURCE_DIRS = ["services", "routes", "models", "config"]


def find_python_files(root="."):
    """
    Discover .py files under SOURCE_DIRS, excluding EXCLUDE_DIRS.
    Stops at the project root — never climbs into site-packages.
    """
    root_path = Path(root).resolve()
    seen = set()
    found = []

    for src in SOURCE_DIRS:
        search_root = root_path / src
        if not search_root.exists():
            continue
        for path in search_root.rglob("*.py"):
            if path in seen:
                continue
            seen.add(path)
            if any(part in EXCLUDE_DIRS for part in path.parts):
                continue
            found.append(str(path))

    return sorted(found)


def to_html(mermaid: str, title: str = "Backend Call Graph") -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/mermaid/10.6.1/mermaid.min.js"></script>
  <style>
    body {{ font-family: sans-serif; margin: 0; background: #f8f9fa; }}
    header {{ background: #1a1a2e; color: white; padding: 16px 24px; }}
    header h1 {{ margin: 0; font-size: 1.2rem; font-weight: 600; }}
    header p {{ margin: 4px 0 0; font-size: 0.8rem; opacity: 0.7; }}
    .legend {{ padding: 10px 24px; background: white; border-bottom: 1px solid #dee2e6; display: flex; gap: 16px; font-size: 0.8rem; align-items: center; }}
    .dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 4px; }}
    .hint {{ margin-left: auto; color: #888; }}
    .diagram-container {{ padding: 24px; overflow: auto; }}
    .mermaid {{ background: white; border-radius: 8px; padding: 24px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
  </style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <p>Auto-generated via AST analysis &mdash; edge labels show call arguments</p>
</header>
<div class="legend">
  <span><span class="dot" style="background:#dbeafe;border:1px solid #93c5fd"></span>clustering</span>
  <span><span class="dot" style="background:#dcfce7;border:1px solid #86efac"></span>role_builder</span>
  <span><span class="dot" style="background:#fef3c7;border:1px solid #fcd34d"></span>mining</span>
  <span><span class="dot" style="background:#f3e8ff;border:1px solid #d8b4fe"></span>session</span>
  <span class="hint">Scroll to zoom &middot; Edge labels show arguments</span>
</div>
<div class="diagram-container">
  <div class="mermaid">
{mermaid}
  </div>
</div>
<script>
  mermaid.initialize({{ startOnLoad: true, theme: "default", flowchart: {{ useMaxWidth: false }}, securityLevel: "loose" }});
</script>
</body>
</html>"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate a call graph from Python source files.")
    parser.add_argument("files", nargs="*", help="Python files to analyze. Defaults to auto-discovery via SOURCE_DIRS.")
    parser.add_argument("--root", default=".", help="Project root directory (default: current directory)")
    parser.add_argument("--html", action="store_true", help="Output a self-contained HTML file instead of raw Mermaid")
    parser.add_argument("--title", default="Backend Call Graph", help="Title for the HTML page")
    args = parser.parse_args()

    files = args.files if args.files else find_python_files(args.root)
    files = [f for f in files if Path(f).exists()]

    if not files:
        print("No Python files found. Run from your project root, or pass files explicitly.", file=sys.stderr)
        sys.exit(1)

    modules, all_functions = build_graph(files)
    print(f"Analyzed {len(modules)} modules, {len(all_functions)} functions", file=sys.stderr)
    mermaid = to_mermaid(modules, all_functions)

    if args.html:
        print(to_html(mermaid, title=args.title))
    else:
        print(mermaid)