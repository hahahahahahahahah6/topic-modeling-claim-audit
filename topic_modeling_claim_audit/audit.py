"""Deterministic documentation and Python syntax audit."""
import argparse
import ast
import json
import re
import sys
from pathlib import Path
from . import __version__
from .phrases import TOPIC, NOTOPIC, GENERIC_NAMES

ROUTES = {
    'gensim.models.LdaModel': {'get_topics', 'show_topics', 'print_topics', 'get_document_topics'},
    'gensim.models.LdaMulticore': {'get_topics', 'show_topics', 'print_topics', 'get_document_topics'},
    'sklearn.decomposition.LatentDirichletAllocation': {'fit', 'fit_transform'},
    'sklearn.decomposition.NMF': {'fit', 'fit_transform'},
    'bertopic.BERTopic': {'fit', 'fit_transform'},
    'gensim.models.wrappers.LdaMallet': {'show_topics', 'get_topics', 'print_topics'},
    'gensim.models.Nmf': {'get_topics', 'show_topics'},
}
EXCLUDED = {'.git', '__pycache__', '.venv', 'venv', 'env', '.env', 'virtualenv'}


def prose(text):
    lines = []
    fenced = False
    for line in text.splitlines():
        if re.match(r'^\s*```', line):
            fenced = not fenced
        elif not fenced:
            lines.append(line.replace('`', ''))
    return '\n'.join(lines).lower()


def consume(text, phrases):
    matches = []
    for phrase in sorted(phrases, key=len, reverse=True):
        start = text.find(phrase)
        while start >= 0:
            matches.append(phrase)
            text = text[:start] + ' ' * len(phrase) + text[start + len(phrase):]
            start = text.find(phrase)
    return text, matches


def match_claims(text):
    remaining, denials = consume(prose(text), NOTOPIC)
    _, claims = consume(remaining, TOPIC)
    return claims, denials


def has_evidence(source):
    tree = ast.parse(source)
    nodes = list(ast.walk(tree))
    shadowed = {n.name for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    imports = {}
    for n in nodes:
        if isinstance(n, ast.Import):
            for a in n.names:
                imports[a.asname or a.name.split('.')[0]] = a.name if a.asname else a.name.split('.')[0]
        elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
            for a in n.names:
                imports[a.asname or a.name] = n.module + '.' + a.name
    for name in shadowed:
        imports.pop(name, None)

    def resolve(node):
        if isinstance(node, ast.Name):
            return imports.get(node.id, '')
        if isinstance(node, ast.Attribute):
            base = resolve(node.value)
            return base + '.' + node.attr if base else ''
        return ''

    instances = {}
    def constructed(node):
        if isinstance(node, ast.Call):
            name = resolve(node.func)
            if name in ROUTES:
                return name
        if isinstance(node, ast.Name):
            return instances.get(node.id)
        return None

    # Fixed point allows instance aliases without depending on AST walk order.
    for _ in range(len(nodes) + 1):
        changed = False
        for n in nodes:
            if isinstance(n, (ast.Assign, ast.AnnAssign)):
                kind = constructed(n.value)
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for target in targets:
                    if isinstance(target, ast.Name) and kind and instances.get(target.id) != kind:
                        instances[target.id] = kind
                        changed = True
        if not changed:
            break
    for n in nodes:
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if isinstance(f, ast.Attribute):
            kind = constructed(f.value)
            if kind and f.attr in ROUTES[kind]:
                return True
            # A locally defined route name cannot supply evidence.
            if f.attr in GENERIC_NAMES and f.attr not in shadowed:
                if isinstance(f.value, ast.Call) and isinstance(f.value.func, ast.Name) and f.value.func.id in shadowed:
                    continue
                if isinstance(f.value, ast.Name) and f.value.id in shadowed:
                    continue
                # Shadowed constructors assigned to a receiver also defeat pairs.
                blocked = any(isinstance(a, ast.Assign) and any(isinstance(t, ast.Name) and isinstance(f.value, ast.Name) and t.id == f.value.id for t in a.targets) and isinstance(a.value, ast.Call) and isinstance(a.value.func, ast.Name) and a.value.func.id in shadowed for a in nodes)
                if not blocked:
                    return True
        elif isinstance(f, ast.Name) and f.id not in shadowed:
            resolved = resolve(f)
            if resolved and resolved.rsplit('.', 1)[-1] in GENERIC_NAMES:
                return True
    return False


def audit(repo, ignore=(), verbose=False):
    root = Path(repo)
    if not root.is_dir():
        raise OSError('Repository is not a directory: ' + str(root))
    evidence = False
    documents = []
    decisions = []
    def skipped(path):
        rel = path.relative_to(root)
        return any(part in EXCLUDED for part in rel.parts) or any((root.joinpath(*rel.parts[:i]) / 'pyvenv.cfg').is_file() for i in range(1, len(rel.parts))) or any(rel.as_posix().startswith(prefix) for prefix in ignore)
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if skipped(path):
            if verbose:
                decisions.append(str(rel) + ': skipped')
            continue
        if path.suffix == '.py':
            try:
                found = has_evidence(path.read_text(encoding='utf-8'))
            except SyntaxError:
                found = False
            evidence = evidence or found
            decisions.append(str(rel) + ': ' + ('evidence' if found else 'no evidence'))
        if (rel.as_posix() == 'README.md' or (rel.parts[0] == 'docs' and path.suffix == '.md') or ('tests' in rel.parts[:-1] and path.suffix == '.md')):
            claims, denials = match_claims(path.read_text(encoding='utf-8'))
            documents.append((rel, claims, denials))
            decisions.append(str(rel) + ': %d claims, %d denials' % (len(claims), len(denials)))
    findings = []
    for rel, claims, denials in documents:
        for identifier, phrases, applies in [('documented-topic-modeling-claim-false', claims, not evidence), ('documented-no-topic-modeling-claim-false', denials, evidence)]:
            if not applies:
                continue
            for phrase in phrases:
                severity = 'MEDIUM' if 'tests' in rel.parts[:-1] else 'HIGH'
                findings.append(dict(id=identifier, verdict=severity, message=f'{rel}: {phrase!r} contradicts source evidence', path=rel.as_posix(), phrase=phrase))
    return findings, decisions


class Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(1, 'error: ' + message + '\n')


def main(argv=None):
    parser = Parser(description='Audit documentation claims against Python source.')
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command', required=True)
    cmd = sub.add_parser('audit')
    cmd.add_argument('repo', nargs='?', default='.')
    cmd.add_argument('--json', action='store_true')
    cmd.add_argument('--verbose', action='store_true')
    cmd.add_argument('--ignore', action='append', nargs='+', default=[])
    cmd.add_argument('--fail-on', choices=['high', 'medium'], default='high')
    args = parser.parse_args(argv)
    try:
        findings, decisions = audit(args.repo, [p for group in args.ignore for p in group], args.verbose)
    except (OSError, UnicodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.verbose:
        print('Scope: README.md, docs/**/*.md, tests/**/*.md; Python sources under ' + str(args.repo), file=sys.stderr)
        for decision in decisions:
            print(decision, file=sys.stderr)
    if args.json:
        print(json.dumps(findings))
    elif findings:
        for finding in findings:
            print('{verdict} {id}: {message}'.format(**finding))
    else:
        print('PASS')
    return 2 if any(f['verdict'] == 'HIGH' or args.fail_on == 'medium' for f in findings) else 0
