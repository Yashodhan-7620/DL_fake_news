"""Generate a SYNTHETIC ISOT-shaped dataset (Fake.csv / True.csv) for smoke-testing the code.

It is NOT real news and the scores it produces mean nothing. Use the real ISOT dataset for results.
"""
import argparse
import random
from pathlib import Path

import pandas as pd

REAL_T = ["{p} announces new {t} policy after talks with {c}", "{c} parliament passes {t} bill by narrow margin",
          "Officials say {t} spending will rise next year, {p} tells reporters",
          "{p} meets {c} delegation to discuss {t} agreement"]
FAKE_T = ["SHOCKING: {p} secretly plans {t} takeover, insiders EXPOSE the truth!!!",
          "You won't BELIEVE what {p} said about {t} - media is hiding it",
          "BREAKING: {c} elites caught faking {t} numbers, wake up people",
          "{p} DESTROYS reporter over {t} lie, patriots cheering"]
PEOPLE = ["the president", "the senator", "the finance minister", "the governor", "the prime minister"]
TOPICS = ["trade", "health", "tax", "energy", "immigration", "defence", "education", "housing"]
COUNTRIES = ["Germany", "India", "Canada", "Brazil", "Japan", "France", "Kenya"]
REAL_B = ["According to a statement released on {d}, the government said the measures would take effect "
          "next quarter and that officials would review the results.",
          "Analysts said the figures were in line with expectations, while opposition lawmakers called "
          "for further debate before the final vote.",
          "The ministry said it would publish a detailed report and consult industry groups over the "
          "coming months before implementing any changes."]
FAKE_B = ["Sources close to the story, who totally cannot be named, say the whole thing is a giant cover up "
          "and nobody in the mainstream press will tell you. Share this before they delete it!",
          "They do not want you to know the real truth. Everything you have been told is a lie and the "
          "so called experts are paid to keep quiet. Wake up and spread the word everywhere.",
          "This is the biggest scandal in history and the corrupt elites are panicking. Thousands are "
          "already furious and demanding answers right now, so share it with everyone you know."]


def make(n, real, rng):
    rows = []
    for i in range(n):
        p, t, c = rng.choice(PEOPLE), rng.choice(TOPICS), rng.choice(COUNTRIES)
        title = rng.choice(REAL_T if real else FAKE_T).format(p=p, t=t, c=c)
        body = " ".join(rng.choice(REAL_B if real else FAKE_B).format(d=f"day {rng.randint(1, 28)}")
                        for _ in range(rng.randint(2, 5)))
        body += f" Case {rng.randint(0, 10**6)} concerns {t} in {c}."          # makes rows unique
        if real:
            body = f"{c.upper()} (Reuters) - " + body
        rows.append({"title": title, "text": body, "subject": "news" if real else "politics",
                     "date": f"2017-0{rng.randint(1, 9)}-1{rng.randint(0, 9)}"})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600, help="rows per class")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "raw"))
    a = ap.parse_args()
    rng = random.Random(0)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    make(a.n, False, rng).to_csv(out / "Fake.csv", index=False)
    make(a.n, True, rng).to_csv(out / "True.csv", index=False)
    print(f"Wrote synthetic Fake.csv / True.csv ({a.n} rows each) to {out}")
