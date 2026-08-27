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


def start_logging(path, mode):
    """Dirotta stdout e stderr su file e tronca il log quando cresce troppo.

    Il logging sta qui e non in un .bat perche' l'attivita' pianificata deve
    girare con pythonw, che non apre nessuna finestra: passando da cmd, Windows
    faceva comparire una console ogni mezz'ora.
    """
    try:
        if os.path.exists(path) and os.path.getsize(path) > 200_000:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                coda = handle.readlines()[-200:]
            with open(path, "w", encoding="utf-8") as handle:
                handle.writelines(coda)
        stream = open(path, "a", encoding="utf-8", buffering=1)
    except Exception:
        return None
    sys.stdout = stream
    sys.stderr = stream
    stamp = __import__("datetime").datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"\n===== {stamp} [{mode}] =====")
    return stream


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


class GatewayUnavailable(Exception):
    """Il gateway non c'è o non è autenticato. Non è fatale: le posizioni si
    aggiornano lo stesso dal Flex, lato server."""


def auth_status(gateway):
    status, payload = http_json(f"{gateway}/iserver/auth/status", method="POST",
                                context=_LOCAL_CTX)
    if status == 0:
        raise GatewayUnavailable(f"gateway non raggiungibile su {gateway}: è avviato?")
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
    raise GatewayUnavailable(
        "gateway non autenticato: apri https://localhost:5000 e rifai login.\n"
        "  Se dice 'competing', un'altra sessione IBKR (TWS, app, connettore) "
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
            # Serve al server per riconoscere le borse che quotano in
            # sottomultipli: senza la valuta, un prezzo LSE in penny verrebbe
            # letto come sterline e il capitale impegnato sarebbe centuplicato.
            "currency": row.get("cashCcy") or row.get("currency"),
        })
    return out


def fetch_pnl(gateway, account_id):
    """P&L del giorno e non realizzato, dal conto.

    È l'unica fonte autorevole del giornaliero: comprende anche il realizzato
    delle posizioni chiuse in giornata, che ricostruito dalle sole posizioni
    aperte mancherebbe. Come altri endpoint /iserver, la prima chiamata avvia
    la sottoscrizione e può tornare vuota.
    """
    row = None
    for attempt in range(4):
        status, payload = http_json(f"{gateway}/iserver/account/pnl/partitioned",
                                    context=_LOCAL_CTX)
        if status == 200 and isinstance(payload, dict):
            entries = payload.get("upnl")
            if isinstance(entries, dict) and entries:
                # Le chiavi sono tipo "U1234567.Core": si prende quella del
                # conto, o l'unica se il nome non combacia.
                for key, value in entries.items():
                    if isinstance(value, dict) and str(key).startswith(str(account_id)):
                        row = value
                        break
                if row is None:
                    row = next((v for v in entries.values() if isinstance(v, dict)), None)
                if row:
                    break
        if attempt < 3:
            time.sleep(2)
    if not row:
        return None
    return {
        "daily_pnl": row.get("dpl"),
        "unrealized_pnl": row.get("upl"),
        "net_liquidation": row.get("nl"),
    }


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
    parser.add_argument("--flex-only", action="store_true",
                        help="salta del tutto il gateway e chiedi solo il giro Flex "
                             "lato server: e' la modalita' dei richiami ogni mezz'ora, "
                             "quando il gateway si sa gia' che non e' autenticato")
    parser.add_argument("--log", default=None,
                        help="scrivi l'esito su questo file invece che a schermo "
                             "(usato dalle attivita' pianificate)")
    args = parser.parse_args()

    if args.log:
        start_logging(args.log, "solo flex" if args.flex_only else "gateway+flex")

    token = read_sync_token(args.token)
    if not token:
        sys.exit("IBKR_SYNC_TOKEN non trovato: passalo con --token o mettilo in .env")
    base = args.polaris.rstrip("/")

    if args.flex_only:
        refresh_positions(base, token)
        return

    try:
        payload = gateway_payload(args)
    except GatewayUnavailable as motivo:
        # Il gateway serve solo per gli ordini e il P&L del giorno. Le posizioni
        # le sa aggiornare il server da solo col Flex, quindi non ci si ferma:
        # si salta la parte locale e si chiede comunque il giro remoto.
        print(f"gateway non disponibile: {motivo}")
        print("procedo col solo aggiornamento posizioni lato server (Flex)")
        refresh_positions(base, token)
        return

    status, response = http_json(
        f"{base}/api/ibkr/sync", method="POST", body=payload,
        headers={"Authorization": f"Bearer {token}"}, timeout=60)
    if status != 200:
        sys.exit(f"sync fallita (HTTP {status}): {str(response)[:300]}")
    report(payload, response, args.notify)
    # Anche col gateway vivo conviene il giro Flex: porta gli eseguiti che il
    # gateway non ha ancora regolato e tiene le posizioni allineate.
    refresh_positions(base, token)


def refresh_positions(base, token):
    """Chiede al server di rifare il giro Flex.

    `notify=0` perché durante la giornata questo gira ogni mezz'ora e l'alert
    earnings deve restare attaccato al giro serale, non ripetersi.
    """
    status, response = http_json(f"{base}/api/ibkr/cron?notify=0", method="GET",
                                 headers={"Authorization": f"Bearer {token}"}, timeout=120)
    if status != 200 or not isinstance(response, dict):
        print(f"aggiornamento posizioni lato server fallito (HTTP {status}): "
              f"{str(response)[:200]}")
        return
    simboli = ", ".join(response.get("position_symbols") or []) or "nessuna"
    print(f"posizioni dal server ({response.get('source')}): {simboli}"
          f"  [{response.get('trades_applied')} eseguiti applicati]")
    if (response.get("skipped") or {}).get("reason"):
        print(f"  nota: {response['skipped']['reason']}")


def report(payload, response, notify):
    alert = response.get("alert") or {}
    print(f"aggiornato: {', '.join(response.get('updated') or [])}")
    print(f"ordini vivi: {response.get('live_orders')} su {response.get('orders')}")
    if payload.get("account"):
        acc = payload["account"]
        riga = f"capitale (net liq)   : {acc['net_liquidation']:,.0f} {acc.get('currency') or ''}"
        if acc.get("daily_pnl") is not None:
            riga += f"   P&L oggi {acc['daily_pnl']:+,.2f}"
        print(riga)
    print(f"earnings {alert.get('target_date')}: {alert.get('count')} "
          f"{', '.join(i['symbol'] for i in alert.get('items') or []) or '-'}")
    if notify:
        print(f"telegram: {response.get('telegram')}")


def gateway_payload(args):
    ensure_authenticated(args.gateway)
    payload = {"orders": fetch_orders(args.gateway), "source": "gateway"}
    if not args.no_positions:
        # Le posizioni del gateway sono live, quelle del Flex sono della
        # chiusura precedente: qui si mandano per impostazione predefinita, e il
        # server tiene comunque il dato più recente confrontando le date.
        account_id = portfolio_account_id(args.gateway)
        payload["positions"] = fetch_positions(args.gateway, account_id)
        payload["positions_as_of"] = time.time()
        account = fetch_account(args.gateway, account_id) or {}
        pnl = fetch_pnl(args.gateway, account_id) or {}
        # Il net liquidation del P&L partizionato è lo stesso del riepilogo:
        # si tiene quello che c'è, senza pretendere che entrambi rispondano.
        merged = {
            "net_liquidation": account.get("net_liquidation") or pnl.get("net_liquidation"),
            "cash": account.get("cash"),
            "currency": account.get("currency"),
            "daily_pnl": pnl.get("daily_pnl"),
            "unrealized_pnl": pnl.get("unrealized_pnl"),
        }
        if merged["net_liquidation"]:
            payload["account"] = merged
    if args.notify:
        payload["notify"] = True
    return payload


if __name__ == "__main__":
    try:
        main()
    except SystemExit as uscita:
        # sys.exit() con messaggio: sotto pythonw non lo vedrebbe nessuno,
        # quindi finisce nel log come tutto il resto.
        if uscita.code not in (0, None):
            print(f"interrotto: {uscita.code}")
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        raise
