def forward(question, call_llm_fn):
    import re, json
    from collections import Counter

    q = question.strip()

    def ask(prompt, system="", temperature=0.2):
        return call_llm_fn(prompt, system=system, temperature=temperature).strip()

    def jget(txt, default):
        m = re.search(r'\{.*\}', txt, re.S)
        if not m:
            return default
        try:
            return json.loads(m.group(0))
        except:
            return default

    def extract_mc_options(text):
        pats = [
            r'([A-D])[\)\.\:]\s*(.+?)(?=(?:\n[A-D][\)\.\:])|\Z)',
            r'\(([A-D])\)\s*(.+?)(?=(?:\n\([A-D]\))|\Z)',
        ]
        for p in pats:
            ms = re.findall(p, text, re.S)
            if len(ms) >= 3:
                return {k: v.strip() for k, v in ms}
        return {}

    def parse_final(text, mode):
        if mode == "mc":
            m = re.search(r'####\s*([A-D])\b', text)
            if m:
                return m.group(1)
            m = re.search(r'\b(?:answer|final)\s*[:\-]?\s*([A-D])\b', text, re.I)
            return m.group(1).upper() if m else None
        m = re.search(r'####\s*([-+]?\d[\d,]*(?:\.\d+)?)', text)
        if m:
            return m.group(1).replace(",", "")
        nums = re.findall(r'[-+]?\d[\d,]*(?:\.\d+)?', text)
        return nums[-1].replace(",", "") if nums else None

    cls = jget(ask(
        'Classify the question. Return JSON only with keys '
        '{"mode":"math|mc|open","brief":"...","difficulty":"easy|med|hard"}\n'
        f'Question:\n{q}',
        system="You are a strict classifier. Output only JSON.",
        temperature=0.0), {"mode": "open", "brief": "", "difficulty": "med"})
    mode = cls.get("mode", "open")
    if mode == "open" and extract_mc_options(q):
        mode = "mc"

    if mode == "mc":
        opts = extract_mc_options(q)
        analyses = []
        for t in (0.1, 0.4, 0.7):
            analyses.append(ask(
                f"Solve the multiple-choice question carefully.\nQuestion:\n{q}\n"
                "Give concise reasoning, then end with '#### <letter>'.",
                system="You are an expert test solver. Be careful and avoid guessing.",
                temperature=t))
        cands = [parse_final(a, "mc") for a in analyses if parse_final(a, "mc")]
        if not cands and opts:
            elim = ask(
                f"Eliminate wrong options first, then choose the best answer.\nQuestion:\n{q}\n"
                f"Options parsed: {json.dumps(opts)}\nEnd with #### <letter>.",
                system="Use process of elimination and factual checking.",
                temperature=0.2)
            x = parse_final(elim, "mc")
            if x:
                cands.append(x)
        choice = Counter(cands).most_common(1)[0][0] if cands else "A"
        judge = ask(
            f"Question:\n{q}\nCandidate answer: {choice}\n"
            "Is this likely correct? Briefly verify against the question and options. "
            "If another option is clearly better, output it. End with #### <letter>.",
            system="You are a skeptical verifier.",
            temperature=0.0)
        final = parse_final(judge, "mc") or choice
        return f"#### {final}"

    if mode == "math":
        plans = ask(
            f"Create a short plan to solve this math word problem.\nQuestion:\n{q}",
            system="Be concise and identify equations/units.", temperature=0.0)
        sols = []
        for t in (0.0, 0.3, 0.6):
            sols.append(ask(
                f"Use this plan if helpful:\n{plans}\n\nSolve carefully:\n{q}\n"
                "Show minimal steps, compute exactly, and end with '#### <number>'.",
                system="You are a meticulous math solver.", temperature=t))
        nums = [parse_final(s, "math") for s in sols if parse_final(s, "math") is not None]
        guess = Counter(nums).most_common(1)[0][0] if nums else None
        verify = ask(
            f"Question:\n{q}\nProposed answer: {guess}\n"
            "Check the arithmetic and logic independently. If wrong, correct it. "
            "End with #### <number>.",
            system="You are a strict math verifier.", temperature=0.0)
        final = parse_final(verify, "math") or guess or "0"
        return f"#### {final}"

    ans = ask(
        f"Answer the question accurately and concisely:\n{q}",
        system="Provide the best possible answer.", temperature=0.2)
    mc = parse_final(ans, "mc")
    if mc:
        return f"#### {mc}"
    num = parse_final(ans, "math")
    if num is not None:
        return f"#### {num}"
    return ans