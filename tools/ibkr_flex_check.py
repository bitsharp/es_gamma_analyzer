#!/usr/bin/env python3
"""Prova le credenziali del Flex Web Service e dice cosa risponde IBKR.

Serve a distinguere le cause dell'errore 1020, che IBKR restituisce identico
per situazioni diverse. La discriminante utile è *da dove* parte la richiesta:
se da qui funziona e dal cron su Vercel no, la causa è una restrizione per
indirizzo IP sul token (Vercel esce da IP variabili e non la si può soddisfare).

Uso:
    python tools/ibkr_flex_check.py --token XXXX --query 123456
    python tools/ibkr_flex_check.py            # legge da .env o dall'ambiente
"""

import argparse
import os
import re
import sys
import urllib.parse
import urllib.request

BASE = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService"

HINTS = {
    "1003": "statement non disponibile per il periodo scelto nella query",
    "1012": "token scaduto: rigenerane uno",
    "1014": "query id inesistente — hai copiato il NUMERO della query, non il nome?",
    "1015": "token non valido: ricontrolla di averlo copiato per intero",
    "1016": "account non valido per questa query",
    "1019": "statement in generazione (in realtà è un buon segno: le credenziali vanno)",
    "1020": "richiesta non validata. Da qui le cause possibili sono due:\n"
            "        - il Flex Web Service non è abilitato (il token da solo non basta:\n"
            "          serve l'interruttore acceso nell'ingranaggio accanto alla voce)\n"
            "        - il token è appena stato creato e non è ancora propagato\n"
            "        Se invece da qui funziona ed è solo Vercel a fallire, allora sul\n"
            "        token c'è una restrizione per indirizzo IP: va rimossa.",
}


def read_env(name):
    from_env = (os.getenv(name) or "").strip()
    if from_env:
        return from_env
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if line.strip().startswith(name + "="):
                return line.split("=", 1)[1].strip()
    return ""


def tag(xml, name):
    match = re.search(rf"<{name}>(.*?)</{name}>", xml, re.S | re.I)
    return match.group(1).strip() if match else None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--token", default=None)
    parser.add_argument("--query", default=None)
    args = parser.parse_args()

    token = args.token or read_env("IBKR_FLEX_TOKEN")
    query = args.query or read_env("IBKR_FLEX_QUERY_ID")
    if not token or not query:
        sys.exit("servono token e query id: --token XXXX --query 123456")

    print(f"token: {token[:4]}…{token[-4:]} ({len(token)} caratteri)")
    print(f"query: {query}\n")
    if not query.isdigit():
        print("  ATTENZIONE: il query id non è numerico. Nel Client Portal la colonna")
        print("  da copiare è il numero della query, non il nome che le hai dato.\n")

    url = f"{BASE}/SendRequest?" + urllib.parse.urlencode({"t": token, "q": query, "v": "3"})
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "polaris/1.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8", "replace")
    except Exception as error:
        sys.exit(f"IBKR irraggiungibile: {error}")

    status = (tag(body, "Status") or "?").lower()
    if status == "success":
        print(f"OK — le credenziali funzionano da questa rete.")
        print(f"     reference code: {tag(body, 'ReferenceCode')}")
        print("\nSe il cron su Vercel continua a dare 1020, allora la causa è")
        print("una restrizione per indirizzo IP sul token: rimuovila dal portale.")
        return

    code = tag(body, "ErrorCode") or "?"
    print(f"FALLITO — codice {code}: {tag(body, 'ErrorMessage') or body[:200]}")
    hint = HINTS.get(code)
    if hint:
        print(f"\n  {hint}")


if __name__ == "__main__":
    main()
