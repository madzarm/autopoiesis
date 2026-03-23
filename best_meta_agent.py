def forward(question, call_llm_fn):
    import re, json
    from collections import Counter

    q = question.strip()

    def ask(prompt, system="", temperature=0.2):
        return call_llm_fn(prompt, system=system, temperature=temperature).strip()

    def jtxt(s, default):
        m = re.search(r'\{.*\}', s, re.S)
        if not m:
            return default
        try:
            return json.loads(m.group(0))
        except:
            return default

    def classify():
        out = ask(
            'Return only JSON: {"type":"math|mc|open","answer_format":"number|letter|text","brief":"..."}\n'
            f"Question:\n{q}",
            system="You are a precise classifier.",
            temperature=0.0,
        )
        d = jtxt(out, {})
        t = d.get("type", "")
        if t in ("math", "mc", "open"):
            return d
        if re.search(r'(^|\n)\s*([A-D])[\).\s]', q) or re.search(r'\b[A-D][\)\.]\s', q):
            return {"type": "mc", "answer_format": "letter", "brief": ""}
        if any(x in q.lower() for x in ["how many", "what is", "total", "left", "remain", "sum"]):
            return {"type": "math", "answer_format": "number", "brief": ""}
        return {"type": "open", "answer_format": "text", "brief": ""}

    def extract_num(s):
        ms = re.findall(r'####\s*([-+]?\d[\d,]*(?:\.\d+)?)', s)
        if ms:
            return ms[-1].replace(",", "")
        ms = re.findall(r'[-+]?\d[\d,]*(?:\.\d+)?', s)
        return ms[-1].replace(",", "") if ms else None

    def extract_letter(s):
        m = re.findall(r'####\s*([A-D])', s.upper())
        if m:
            return m[-1]
        m = re.findall(r'\b([A-D])\b', s.upper())
        return m[-1] if m else None

    def solve_math():
        sols = []
        for t in (0.2, 0.5, 0.8):
            out = ask(
                "Solve carefully but concisely. Check arithmetic. End with exactly: #### <number>\n"
                f"Question:\n{q}",
                system="You are a careful math solver.",
                temperature=t,
            )
            n = extract_num(out)
            if n is not None:
                sols.append((n, out))
        if not sols:
            out = ask(f"Solve and end with #### <number>.\n{q}", temperature=0.2)
            n = extract_num(out)
            return out if n is None else re.sub(r'.*', f"#### {n}", out.splitlines()[-1])
        cnt = Counter(n for n, _ in sols)
        best = cnt.most_common(1)[0][0]
        judge = ask(
            "Choose the most likely correct final numeric answer among these candidate solutions. "
            "Return only JSON {\"answer\":\"<number>\",\"why\":\"...\"}.\n\n" +
            "\n\n".join([f"Candidate {i+1}:\n{s}" for i, (_, s) in enumerate(sols)]),
            system="You evaluate consistency and arithmetic correctness.",
            temperature=0.0,
        )
        jd = jtxt(judge, {})
        ans = jd.get("answer", best)
        return f"#### {ans}"

    def solve_mc():
        drafts = []
        for t in (0.0, 0.3, 0.7):
            out = ask(
                "Solve the multiple-choice question. Briefly reason by eliminating wrong options. "
                "End with exactly: #### <letter>\n"
                f"Question:\n{q}",
                system="You are an expert test taker.",
                temperature=t,
            )
            a = extract_letter(out)
            if a:
                drafts.append((a, out))
        if not drafts:
            out = ask(f"Answer with only the correct option letter.\n{q}", temperature=0.0)
            a = extract_letter(out)
            return f"#### {a}" if a else out
        cnt = Counter(a for a, _ in drafts)
        top = [k for k, v in cnt.items() if v == cnt.most_common(1)[0][1]]
        if len(top) == 1:
            return f"#### {top[0]}"
        judge = ask(
            "You are given several candidate analyses for the same multiple-choice question. "
            "Select the best-supported option. Return only JSON {\"answer\":\"A|B|C|D\"}.\n\n"
            f"Question:\n{q}\n\n" + "\n\n".join([f"Candidate {i+1}:\n{s}" for i, (_, s) in enumerate(drafts)]),
            system="Prefer direct textual evidence and valid reasoning.",
            temperature=0.0,
        )
        jd = jtxt(judge, {})
        a = jd.get("answer", top[0])
        return f"#### {a}"

    c = classify()
    if c["type"] == "math":
        return solve_math()
    if c["type"] == "mc":
        return solve_mc()
    out = ask(f"Answer concisely.\nQuestion:\n{q}", system="You are a helpful expert.", temperature=0.2)
    return out