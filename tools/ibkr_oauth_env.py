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
    python tools/ibkr_oauth_env.py --dir ./ibkr-keys

Si aspetta nella cartella: private_signature.pem, private_encryption.pem,
dhparam.pem. Per generarli:

    openssl genrsa -out private_signature.pem 2048
    openssl rsa -in private_signature.pem -outform PEM -pubout -out public_signature.pem
    openssl genrsa -out private_encryption.pem 2048
    openssl rsa -in private_encryption.pem -outform PEM -pubout -out public_encryption.pem
    openssl dhparam -outform PEM 2048 -out dhparam.pem

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
    args = parser.parse_args()

    signature = read_text(os.path.join(args.dir, "private_signature.pem"))
    encryption = read_text(os.path.join(args.dir, "private_encryption.pem"))
    prime = dh_prime_hex(os.path.join(args.dir, "dhparam.pem"))

    b64 = lambda text: base64.b64encode(text.encode("utf-8")).decode("ascii")

    print("# Incolla questi valori tra le Environment Variables del progetto Vercel.")
    print("# Le due chiavi sono in base64: l'app le riconosce e così nessun pannello")
    print("# web può rovinarle mangiando gli a-capo.\n")
    print(f"IBKR_SIGNATURE_KEY={b64(signature)}")
    print(f"IBKR_ENCRYPTION_KEY={b64(encryption)}")
    print(f"IBKR_DH_PRIME={prime}")
    print()
    print(f"# primo DH: {len(prime) * 4} bit — atteso 2048")
    print("# Restano da compilare a mano, dal Self-Service Portal:")
    print("#   IBKR_CONSUMER_KEY, IBKR_ACCESS_TOKEN, IBKR_ACCESS_TOKEN_SECRET")


if __name__ == "__main__":
    main()
