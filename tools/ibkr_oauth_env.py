#!/usr/bin/env python3
"""Prepara le variabili d'ambiente per l'OAuth 1.0a di IBKR.

Legge i file prodotti da OpenSSL e stampa i valori pronti da incollare tra le
Environment Variables di Vercel. Esiste per due passaggi che a mano si sbagliano
quasi sempre:

  - il primo Diffie-Hellman va dato in esadecimale continuo, mentre `openssl dh`
    lo stampa a blocchi separati da due punti e da a-capo;
  - le chiavi private sono PEM multilinea, e molti pannelli web li accettano
    male: qui vengono emesse anche in base64, forma che l'app riconosce.

Uso:
    python tools/ibkr_oauth_env.py --dir ./ibkr-keys --out ./ibkr-keys/vercel-env.txt

Si aspetta nella cartella: private_signature.pem, private_encryption.pem,
dhparam.pem. Per generarli:

    openssl genrsa -out private_signature.pem 2048
    openssl rsa -in private_signature.pem -outform PEM -pubout -out public_signature.pem
    openssl genrsa -out private_encryption.pem 2048
    openssl rsa -in private_encryption.pem -outform PEM -pubout -out public_encryption.pem
    openssl dhparam -outform PEM -out dhparam.pem 2048

Attenzione all'ultima riga: da OpenSSL 3.x il numero di bit va per ultimo,
dopo tutte le opzioni, altrimenti risponde 'Extra option: "2048"'.

Le tre pubbliche (public_signature.pem, public_encryption.pem, dhparam.pem)
vanno caricate nel Self-Service Portal. Le private restano da te: non
committarle da nessuna parte.
"""

import argparse
import base64
import os
import sys


def read_text(path):
    if not os.path.exists(path):
        sys.exit(f"manca il file {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def dh_prime_hex(dhparam_path):
    """Estrae il primo dal dhparam.pem come esadecimale continuo.

    Si parsifica il DER invece di chiamare `openssl dh -text`: non tutte le
    build di OpenSSL stampano quel testo allo stesso modo, e su Windows spesso
    l'eseguibile non c'è nemmeno.
    """
    pem = read_text(dhparam_path)
    body = "".join(line for line in pem.splitlines() if "-----" not in line)
    der = base64.b64decode(body)

    # DHParameter ::= SEQUENCE { prime INTEGER, generator INTEGER }
    def read_len(data, i):
        first = data[i]
        i += 1
        if first < 0x80:
            return first, i
        count = first & 0x7F
        return int.from_bytes(data[i:i + count], "big"), i + count

    idx = 0
    if der[idx] != 0x30:
        sys.exit("dhparam.pem non contiene una SEQUENCE: file inatteso")
    idx += 1
    _, idx = read_len(der, idx)
    if der[idx] != 0x02:
        sys.exit("dhparam.pem: primo elemento non è un INTEGER")
    idx += 1
    length, idx = read_len(der, idx)
    prime_bytes = der[idx:idx + length].lstrip(b"\x00")
    return prime_bytes.hex()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default=".", help="cartella con i file .pem")
    parser.add_argument("--out", help="scrivi su file invece che a schermo. "
                                      "Consigliato: sono chiavi private, meglio "
                                      "non lasciarle nello scrollback del terminale")
    args = parser.parse_args()

    signature = read_text(os.path.join(args.dir, "private_signature.pem"))
    encryption = read_text(os.path.join(args.dir, "private_encryption.pem"))
    prime = dh_prime_hex(os.path.join(args.dir, "dhparam.pem"))

    def b64(text):
        return base64.b64encode(text.encode("utf-8")).decode("ascii")

    lines = [
        "# Incolla questi valori tra le Environment Variables del progetto Vercel.",
        "# Le due chiavi sono in base64: l'app le riconosce e così nessun pannello",
        "# web può rovinarle mangiando gli a-capo.",
        "",
        f"IBKR_SIGNATURE_KEY={b64(signature)}",
        f"IBKR_ENCRYPTION_KEY={b64(encryption)}",
        f"IBKR_DH_PRIME={prime}",
        "",
        f"# primo DH: {len(prime) * 4} bit — atteso 2048",
        "# Restano da compilare a mano, dal Self-Service Portal:",
        "#   IBKR_CONSUMER_KEY, IBKR_ACCESS_TOKEN, IBKR_ACCESS_TOKEN_SECRET",
    ]
    text = "\n".join(lines) + "\n"

    if args.out:
        # encoding esplicito: su Windows il default è cp1252 e le lettere
        # accentate dei commenti renderebbero illeggibile il file.
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
        print(f"scritto in {args.out}")
    else:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
