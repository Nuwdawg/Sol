"""Split Maths 9709 P2, P3 and P4 (Mechanics) questions into subtopics.

Pure 3 and Mechanics use the subtopic list of Past Paper Penguin's 9709 topical pages
(pastpaperpenguin.com, fetched 2026-11-16). Past Paper Penguin has no Pure 2 page, so Pure 2 uses
the same names where the content is shared with Pure 3, plus two integration subtopics that only
Pure 2 has (exponentials/reciprocals and the trapezium rule).

Each question gets exactly one subtopic, decided from the question text and its mark scheme.
Manual corrections (id -> subtopic code) in subtopic_overrides.json win over the rules.

usage: subtopics.py SCRATCH          writes work/subtopics.json
"""
import sys, os, re, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import topics

# code, name, parent syllabus topic
SUB = {
    2: [
        ("2.1.1", "Division & Factor/Remainder Theorems", "2.1"),
        ("2.1.2", "Modulus Equations and Inequalities", "2.1"),
        ("2.2.1", "Exps and Logs: Equations and Inequalities", "2.2"),
        ("2.2.2", "Transforming Log Graphs to Linear", "2.2"),
        ("2.3.1", "Trig: Compound and Double Angles", "2.3"),
        ("2.3.2", "Rsin(x+a) and Rcos(x+a)", "2.3"),
        ("2.4.1", "Product and Quotient Rule", "2.4"),
        ("2.4.2", "Implicit Differentiation", "2.4"),
        ("2.4.3", "Parametric Differentiation", "2.4"),
        ("2.5.1", "Integration of Trig Functions", "2.5"),
        ("2.5.2", "Integration of Exponentials and 1/(ax+b)", "2.5"),
        ("2.5.3", "Trapezium Rule", "2.5"),
        ("2.6.1", "Iteration with Calculus", "2.6"),
        ("2.6.2", "Iteration with Graph Sketching or Radians", "2.6"),
    ],
    3: [
        ("3.1.1", "Division & Factor/Remainder Theorems", "3.1"),
        ("3.1.2", "Modulus Equations and Inequalities", "3.1"),
        ("3.1.3", "Binomial Expansion", "3.1"),
        ("3.1.4", "Partial Fractions and Binomial Expansion", "3.1"),
        ("3.2.1", "Exps and Logs: Equations and Inequalities", "3.2"),
        ("3.2.2", "Transforming Log Graphs to Linear", "3.2"),
        ("3.3.1", "Trig: Compound and Double Angles", "3.3"),
        ("3.3.2", "Rsin(x+a) and Rcos(x+a)", "3.3"),
        ("3.4.1", "Product and Quotient Rule", "3.4"),
        ("3.4.2", "Implicit Differentiation", "3.4"),
        ("3.4.3", "Parametric Differentiation", "3.4"),
        ("3.5.1", "Integration of Trig Functions", "3.5"),
        ("3.5.2", "Integration by Parts", "3.5"),
        ("3.5.3", "Integration by Substitution", "3.5"),
        ("3.5.4", "Partial Fractions and Integration", "3.5"),
        ("3.5.5", "Trapezium Rule", "3.5"),
        ("3.6.1", "Iteration with Calculus", "3.6"),
        ("3.6.2", "Iteration with Graph Sketching or Radians", "3.6"),
        ("3.7.1", "Vectors", "3.7"),
        ("3.8.1", "Differential Equations", "3.8"),
        ("3.8.2", "Differential Equations & Modelling", "3.8"),
        ("3.9.1", "Complex Numbers: Equations and Roots", "3.9"),
        ("3.9.2", "Complex Numbers: Equations and Loci", "3.9"),
    ],
    4: [
        ("4.1.1", "Resolving Coplanar Forces", "4.1"),
        ("4.1.2", "Friction: Horizontal", "4.1"),
        ("4.1.3", "Friction: On an Incline", "4.1"),
        ("4.2.1", "Kinematic Graphs", "4.2"),
        ("4.2.2", "Suvat: Horizontal", "4.2"),
        ("4.2.3", "Suvat: Vertical", "4.2"),
        ("4.2.4", "General Motion in a Straight Line", "4.2"),
        ("4.3.1", "Momentum & Collisions", "4.3"),
        ("4.3.2", "Collisions with Friction or Energy", "4.3"),
        ("4.4.1", "Connected Particles: Horizontal or Inclined", "4.4"),
        ("4.4.2", "Connected Particles: Vertical", "4.4"),
        ("4.4.3", "Pulleys", "4.4"),
        ("4.5.1", "Work and Energy", "4.5"),
        ("4.5.2", "Power and F=ma", "4.5"),
        ("4.5.3", "Power and F=ma with Connected Particles", "4.5"),
        ("4.5.4", "Pulleys and Work/Energy", "4.5"),
    ],
}
NAME = {c: n for comp in SUB.values() for c, n, _ in comp}

def has(rx, s):
    return re.search(rx, s, re.I) is not None

INTEGRAL = r"Ó|∫|integral|integrat|\barea\b"
TRIGF = r"\b(sin|cos|tan|sec|cosec|cot)"
DERIV = r"d\s*y\s*/?\s*d\s*x|d\s*y\s+d\s*x|differentiat|stationary|gradient|tangent|normal to|maximum|minimum|turning point"

def implicit_curve(q):
    """'The equation of a curve is x3 + 3xy2 - y3 = 5': x and y mixed, not y = f(x)."""
    m = re.search(r"(equation of (a|the) curve is|curve with equation|gradient of the curve|variables x and y satisfy the equation)\s*(.{0,70})", q, re.I)
    if not m:
        return False
    e = m.group(3)
    if re.match(r"\s*y\s*=", e):
        return False
    return re.search(r"x\s*y|y\s*x|y\s*[23]\b|\by\s*e|\be\s*y|ln\s*\(?\s*`?\s*x\s*\+\s*\d*\s*y|y\s*ln|ln\s*\d*\s*y", e) is not None

def pure(comp, r, q, m):
    p = f"{comp}."
    T = r["topic"]
    # ---- topics that only exist in P3
    if comp == 3 and (T == "3.9" or has(r"complex number|argand|\barg\b", q)):
        return "3.9.2" if has(r"shade|locus|loci|region|satisfying the inequalit|satisf(y|ies) the inequalit", q) else "3.9.1"
    if comp == 3 and T == "3.7":
        return "3.7.1"
    if comp == 3 and (T == "3.8" or has(r"differential equation", q)):
        ctx = r"proportional|modell?ed|\bmodel\b|population|tank|liquid|water|bacteria|insect|cells|chemical|substance|reaction|organism|spread|leak|\boil\b|temperature|forest|fish|virus|weed|algae|disease|rumour|crystal|drug|bird|animal|rabbit|plant|tree|money|price|company|salt|sugar|snow|ice|balloon|bowl|vessel|container|cylinder"
        return "3.8.2" if has(ctx, q) else "3.8.1"
    # ---- numerical solution
    if T == f"{comp}.6" or has(r"iterati|x\s*n\s*\+\s*1|x\s*\n?\s*n\s*\+\s*1", q):
        if has(r"\brad(ians?)?\b|sector|\barc\b|semicircle|segment", q) and not has(DERIV, q):
            return p + "6.2"                                  # geometry in radians
        return p + ("6.1" if has(DERIV + r"|Ó|∫|integral|integrat", q) else "6.2")
    pf = has(r"partial fraction", q)
    # ---- binomial / partial fractions (P3)
    if comp == 3 and has(r"expansion|ascending powers|\bexpand\b", q):
        return "3.1.4" if pf else "3.1.3"
    if comp == 3 and pf:
        return "3.5.4" if (has(INTEGRAL, q) or T == "3.5") else "3.1.4"
    # ---- integration
    if T == f"{comp}.5" or (has(INTEGRAL, q) and not has(DERIV, q)):
        if comp == 2:
            if has(r"trapezium", q):
                return "2.5.3"
            return "2.5.1" if has(TRIGF, q) else "2.5.2"
        if has(r"trapezium", q):
            return "3.5.5"
        if has(r"substitution", q):
            return "3.5.3"
        if has(r"by parts", q) or has(r"by parts", m):
            return "3.5.2"
        if has(r"substitut", m):
            return "3.5.3"
        return "3.5.1" if has(TRIGF, q) else "3.5.3"
    # ---- differentiation
    if T == f"{comp}.4":
        if has(r"parametric|parameter", q) or has(r"\bx\s*=[^.]*\bt\b[^.]*\by\s*=|\bx\s*=[^.]*θ[^.]*\by\s*=", q):
            return p + "4.3"
        if has(r"implicit", m) or has(r"implicit", q) or implicit_curve(q):
            return p + "4.2"
        return p + "4.1"
    # ---- trigonometry
    if T == f"{comp}.3":
        return p + ("3.2" if has(r"\bR\s*(sin|cos)|form\s+R|R\s*>\s*0|R\s*\(?\s*(sin|cos)", q) else "3.1")
    # ---- logs and exponentials
    if T == f"{comp}.2":
        return p + ("2.2" if has(r"straight line|against|linear|graph of (ln|lg|log)", q) else "2.1")
    # ---- algebra
    if T == f"{comp}.1":
        if has(r"factor|remainder|polynomial|divided by|quotient|\bp\s*\(\s*x\s*\)", q):
            return p + "1.1"
        return p + "1.2"                      # modulus signs are lost in extraction: the rest of 2.1/3.1 is modulus work
    raise ValueError(r["id"] + " " + T)

def mech(r, q, m):
    """Mechanics. Many 2018-2023 texts were extracted without spaces, so keywords are matched
    against the question with all whitespace removed."""
    T = r["topic"]
    z = re.sub(r"\s+", "", q).lower()
    def k(rx):
        return re.search(rx, z) is not None
    pulley = k(r"pulley|peg|passesover")
    energy = k(r"workdone|work-energy|kineticenergy|potentialenergy|energy")
    power = k(r"power|\dkw|rateofworking|workingataconstantrate|worksata(constant)?rate") or has(r"\d\s*W\b", q)
    connected = k(r"towing|towed|tow-?bar|trailer|caravan|coupling|connectedby|joinedby|areconnected|attachedtoeachother|truck|coach|carriage")
    vertical = k(r"lift|elevator|verticalstring|verticalrope|verticalcable|hangs|hanging|suspended|verticallyupwards|verticallydownwards|raised|lowered")
    rough = k(r"friction|(?<!th)rough")
    incline = k(r"inclined|slope|hill|descends")
    momentum = k(r"momentum|collid|collision|coalesc|impulse")
    # variable acceleration: the mark scheme integrates or differentiates
    if T == "4.2" and has(r"integrat|differentiat|∫|Ó|d\s*v\s*/?\s*d\s*t|d\s*s\s*/?\s*d\s*t", m):
        return "4.2.4"
    if momentum:
        return "4.3.2" if (rough or energy) else "4.3.1"
    if pulley:
        return "4.5.4" if energy else "4.4.3"
    if power:
        return "4.5.3" if connected else "4.5.2"
    if T == "4.2":
        if k(r"coefficientoffriction"):
            return "4.1.3" if incline else "4.1.2"
        if energy:
            return "4.5.1"
        if k(r"timegraph|sketch|graph"):
            return "4.2.1"
        if k(r"vertically|dropped|releasedfromrest|thrown|projected|falls|heightof"):
            return "4.2.3"
        return "4.2.2"
    if connected:
        if energy:
            return "4.5.1"
        return "4.4.2" if vertical and not incline else "4.4.1"
    if energy:
        return "4.5.1"
    if rough:
        return "4.1.3" if incline else "4.1.2"
    if T == "4.3":
        return "4.3.1"
    if T in ("4.1", "4.4"):
        if T == "4.1" or k(r"equilibrium|atrest|stationary"):
            return "4.1.1"
        if vertical:
            return "4.4.2"
        return "4.1.3" if incline else "4.2.2"
    return "4.5.1"

def classify(r):
    q, m = topics.norm(r["text"]), topics.norm(r["ms_text"])
    return mech(r, q, m) if r["comp"] == 4 else pure(r["comp"], r, q, m)

if __name__ == "__main__":
    SP = sys.argv[1]
    S = json.load(open(os.path.join(SP, "work/selection.json")))
    Q = S["questions"]
    dups = {d for r in Q.values() for d in r.get("dup_of_this", [])}
    here = os.path.dirname(os.path.abspath(__file__))
    ovf = os.path.join(here, "subtopic_overrides.json")
    OV = json.load(open(ovf)) if os.path.exists(ovf) else {}
    out = {}
    for r in Q.values():
        if r["subj"] != "9709" or r["comp"] not in (2, 3, 4) or r["id"] in dups or r.get("off_syllabus"):
            continue
        out[r["id"]] = OV.get(r["id"]) or classify(r)
    json.dump(out, open(os.path.join(SP, "work/subtopics.json"), "w"), indent=0, sort_keys=True)
    c = collections.Counter(out.values())
    for code in sorted(c):
        print(f"{code:6s} {NAME[code]:45s} {c[code]}")
    print(len(out), "questions")
