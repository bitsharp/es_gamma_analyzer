#!/usr/bin/env python3
"""Manda a Polaris gli ordini pendenti letti dal Client Portal Gateway di IBKR.

È la metà locale dell'ibrido: il Flex Web Service gira sempre sul cron di
Vercel ma le posizioni sono tutto quello che sa dare, mentre gli ordini di
lavoro esistono solo nella sessione di brokeraggio. Questo script li prende dal
gateway in esecuzione sulla tua macchina e li deposita su /api/ibkr/sync, che
li fonde con le posizioni senza sovrascriverle.

Prerequisiti
------------
1. Scarica il Client Portal Gateway da IBKR e avvialo:
       bin/run.sh root/conf.yaml          (Linux/macOS)
       bin\\run.bat root\\conf.yaml         (Windows)
2. Apri https://localhost:5000 nel browser e fai login. La sessione dura circa
   24h, quindi va rifatto ogni tanto: se lo script dice che non sei
   autenticato, è questo che manca.
3. Esporta il token di sync (lo stesso che sta su Vercel):
       export IBKR_SYNC_TOKEN=...          # oppure lo legge da .env

Uso
---
    python tools/ibkr_gateway_sync.py
    python tools/ibkr_gateway_sync.py --no-positions   # solo ordini
    python tools/ibkr_gateway_sync.py --notify         # e fa notificare l'alert

Manda posizioni e ordini, entrambi live. Il Flex resta la rete di sicurezza per
i giorni a PC spento, ma fotografa la chiusura precedente: se durante la
giornata apri o chiudi qualcosa, solo il gateway se ne accorge. Il server
confronta le date del dato e tiene il più recente, quindi il giro serale del
Flex non riporta indietro quello che il gateway ha appena scritto.

Per farlo girare da solo: Utilità di pianificazione di Windows, azione
"Avvia programma" su pythonw.exe con questo script come argomento, ogni giorno
alle 19:45 — prima del cron di Vercel, così gli ordini arrivano già freschi.
"""

import argparse
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_GATEWAY = "https://localhost:5000/v1/api"
DEFAULT_POLARIS = "https://es-gamma-analyzer.vercel.app"

# Il gateway usa un certificato autofirmato su localhost. Disattivare la
# verifica qui è accettabile — e solo qui: la connessione non esce dalla
# macchina. Verso Polaris la verifica resta quella di sistema.
_LOCAL_CTX = ssl.create_default_context()
_LOCAL_CTX.check_hostname = False
_LOCAL_CTX.verify_mode = ssl.CERT_NONE


def http_json(url, method="GET", body=None, headers=None, context=None, timeout=25):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(url, data=data, method=method,
                                     headers={"User-Agent": "polaris-gateway-sync/1.0",
                                              **({"Content-Type": "application/json"} if data else {}),
                                              **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            raw = response.read().decode("utf-8", "replace")
            return response.status, (json.loads(raw) if raw.strip() else None)
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8", "replace")
        try:
            return error.code, json.loads(raw)
        except ValueError:
            return error.code, raw
    except Exception as error:  # gateway spento, DNS, timeout…
        return 0, str(error)


def read_sync_token(explicit):
    if explicit:
        return explicit
    from_env = (os.getenv("IBKR_SYNC_TOKEN") or "").strip()
    if from_env:
        return from_env
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        for line in open(env_path, encoding="utf-8"):
            if line.strip().startswith("IBKR_SYNC_TOKEN="):
                return line.split("=", 1)[1].strip()
    return ""


def auth_status(gateway):
    status, payload = http_json(f"{gateway}/iserver/auth/status", method="POST",
                                context=_LOCAL_CTX)
    if status == 0:
        sys.exit(f"gateway non raggiungibile su {gateway}: è avviato?\n  ({payload})")
    return payload if isinstance(payload, dict) else {}


def ensure_authenticated(gateway):
    """Verifica la sessione e, se è scaduta, tenta di riaprirla.

    La sessione di brokeraggio cade da sola dopo qualche ora di inattività, e
    `/iserver/reauthenticate` risponde subito "triggered" ma impiega una
    decina di secondi a ristabilirla: controllare lo stato immediatamente dopo
    dà sempre un falso negativo, quindi si aspetta.
    """
    if auth_status(gateway).get("authenticated"):
        return
    print("sessione scaduta, provo a riaprirla…")
    http_json(f"{gateway}/iserver/reauthenticate", method="POST", context=_LOCAL_CTX)
    for _ in range(10):
        time.sleep(3)
        if auth_status(gateway).get("authenticated"):
            print("sessione ristabilita")
            return
    sys.exit("gateway non autenticato: apri https://localhost:5000 e rifai login.\n"
             "Se dice 'competing', un'altra sessione IBKR (TWS, app, connettore) "
             "ha preso il posto: IBKR ne ammette una sola per utenza.")


_ORDER_SYMBOL_RE = re.compile(r"^\s*(?:buy|sell)\s+[\d.,]+\s+(\S+)", re.IGNORECASE)


def prime_session(gateway):
    """Innesca la sessione con /iserver/accounts.

    Senza questa chiamata gli endpoint /iserver rispondono 200 con una lista
    vuota invece di un errore — che è il modo peggiore di fallire, perché
    sembra "nessun ordine aperto" e svuoterebbe la lista in pagina.
    """
    status, payload = http_json(f"{gateway}/iserver/accounts", context=_LOCAL_CTX)
    if status != 200 or not isinstance(payload, dict) or not payload.get("accounts"):
        sys.exit(f"inizializzazione della sessione fallita (HTTP {status}): {str(payload)[:200]}")
    return payload["accounts"][0]


def fetch_orders(gateway):
    prime_session(gateway)
    payload = None
    # La prima lettura può tornare vuota: l'endpoint avvia uno snapshot e
    # risponde subito, senza aspettare di averlo.
    for attempt in range(4):
        status, payload = http_json(f"{gateway}/iserver/account/orders", context=_LOCAL_CTX)
        if status != 200 or not isinstance(payload, dict):
            sys.exit(f"lettura ordini fallita (HTTP {status}): {str(payload)[:200]}")
        if payload.get("orders"):
            break
        if attempt < 3:
            time.sleep(2)
    out = []
    for row in payload.get("orders") or []:
        if not isinstance(row, dict):
            continue
        symbol = row.get("ticker")
        if not symbol:
            match = _ORDER_SYMBOL_RE.match(str(row.get("orderDesc") or ""))
            symbol = match.group(1) if match else None
        if not symbol:
            continue
        out.append({
            "order_id": row.get("orderId"),
            "symbol": symbol,
            "name": row.get("companyName"),
            "side": row.get("side"),
            "order_type": row.get("orderType"),
            "status": row.get("status"),
            "quantity": row.get("totalSize"),
            "remaining": row.get("remainingQuantity"),
            "limit_price": row.get("price"),
            "stop_price": row.get("auxPrice") or row.get("stop_price"),
            "tif": row.get("timeInForce"),
            "description": row.get("orderDesc"),
            "exchange": row.get("listingExchange"),
        })
    return out


def fetch_account(gateway, account_id):
    """Net liquidation e liquidità: il denominatore dell'esposizione.

    Senza, la pagina non può dire quanto pesa il portafoglio sul capitale — e
    preferisce dichiararlo mancante piuttosto che stimarlo.
    """
    status, payload = http_json(f"{gateway}/portfolio/{account_id}/summary", context=_LOCAL_CTX)
    if status != 200 or not isinstance(payload, dict):
        return None

    def amount(key):
        node = payload.get(key)
        if isinstance(node, dict):
            return node.get("amount")
        return node if isinstance(node, (int, float)) else None

    currency = None
    node = payload.get("netliquidation")
    if isinstance(node, dict):
        currency = node.get("currency")
    return {
        "net_liquidation": amount("netliquidation"),
        "cash": amount("totalcashvalue") or amount("availablefunds"),
        "currency": currency,
    }


def portfolio_account_id(gateway):
    status, accounts = http_json(f"{gateway}/portfolio/accounts", context=_LOCAL_CTX)
    if status != 200 or not isinstance(accounts, list) or not accounts:
        sys.exit(f"nessun account dal gateway (HTTP {status})")
    return accounts[0].get("accountId") or accounts[0].get("id")


def fetch_positions(gateway, account_id):
    out, page = [], 0
    while page < 10:
        status, rows = http_json(f"{gateway}/portfolio/{account_id}/positions/{page}",
                                 context=_LOCAL_CTX)
        if status != 200 or not isinstance(rows, list) or not rows:
            break
        for row in rows:
            symbol = row.get("ticker") or row.get("contractDesc")
            if not symbol:
                continue
            out.append({
                "symbol": symbol,
                "name": row.get("name") or row.get("contractDesc"),
                "quantity": row.get("position"),
                "avg_price": row.get("avgPrice"),
                "market_price": row.get("mktPrice"),
                "market_value": row.get("mktValue"),
                "unrealized_pnl": row.get("unrealizedPnl"),
                "currency": row.get("currency"),
                "asset_class": row.get("assetClass"),
                "exchange": row.get("listingExchange"),
            })
        if len(rows) < 30:
            break
        page += 1
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gateway", default=os.getenv("IBKR_GATEWAY_URL") or DEFAULT_GATEWAY)
    parser.add_argument("--polaris", default=os.getenv("POLARIS_BASE_URL") or DEFAULT_POLARIS)
    parser.add_argument("--token", default=None, help="IBKR_SYNC_TOKEN (default: env o .env)")
    parser.add_argument("--no-positions", action="store_true",
                        help="manda solo gli ordini, lasciando le posizioni al Flex")
    parser.add_argument("--notify", action="store_true",
                        help="fai calcolare e notificare subito l'alert del giorno dopo")
    args = parser.parse_args()

    token = read_sync_token(args.token)
    if not token:
        sys.exit("IBKR_SYNC_TOKEN non trovato: passalo con --token o mettilo in .env")

    ensure_authenticated(args.gateway)
    payload = {"orders": fetch_orders(args.gateway), "source": "gateway"}
    if not args.no_positions:
        # Le posizioni del gateway sono live, quelle del Flex sono della
        # chiusura precedente: qui si mandano per impostazione predefinita, e il
        # server tiene comunque il dato più recente confrontando le date.
        account_id = portfolio_account_id(args.gateway)
        payload["positions"] = fetch_positions(args.gateway, account_id)
        payload["positions_as_of"] = time.time()
        account = fetch_account(args.gateway, account_id)
        if account and account.get("net_liquidation"):
            payload["account"] = account
    if args.notify:
        payload["notify"] = True

    status, response = http_json(
        f"{args.polaris.rstrip('/')}/api/ibkr/sync", method="POST", body=payload,
        headers={"Authorization": f"Bearer {token}"}, timeout=60)
    if status != 200:
        sys.exit(f"sync fallita (HTTP {status}): {str(response)[:300]}")

    alert = response.get("alert") or {}
    print(f"aggiornato: {', '.join(response.get('updated') or [])}")
    print(f"ordini vivi: {response.get('live_orders')} su {response.get('orders')}")
    print(f"posizioni in archivio: {response.get('positions')}")
    if payload.get("account"):
        print(f"capitale (net liq)   : {payload['account']['net_liquidation']:,.0f} "
              f"{payload['account'].get('currency') or ''}")
    print(f"earnings {alert.get('target_date')}: {alert.get('count')} "
          f"{', '.join(i['symbol'] for i in alert.get('items') or []) or '—'}")
    if args.notify:
        print(f"telegram: {response.get('telegram')}")


if __name__ == "__main__":
    main()
