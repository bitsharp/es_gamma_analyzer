// Polaris · livelli gamma ZeroGEX su Tradovate (indicatore custom, ES e NQ).
//
// Scarica i livelli da Polaris ogni 5 minuti e li disegna come linee orizzontali
// con etichetta "prezzo-nome" (7763.5-CallW). I livelli hanno almeno 15 minuti di
// ritardo. ES o NQ si riconosce da solo dal prezzo del grafico.
//
// Installazione: in Tradovate apri Code Explorer, File > New, incolla tutto questo
// file, salva, poi aggiungi l'indicatore "polarisZeroGexLevels" al grafico.
//
// Se accanto al prezzo compare la scritta "ZeroGEX: ..." arancione, e' la diagnosi:
// dice se la richiesta e' partita, con quale trasporto e come e' finita. Il log
// completo e' nella console di Code Explorer (righe che cominciano con "[PZG]").

const predef = require("./tools/predef");
const meta = require("./tools/meta");
const { px, du, op } = require("./tools/graphics");

const BASE_URL = "{{ base_url }}";
const KEY = "{{ token }}";
const REFRESH_MS = 5 * 60 * 1000;
const TIMEOUT_MS = 15 * 1000;

// Lo stato sta a livello di modulo e non nell'istanza: Tradovate ricrea
// l'indicatore (init) a ogni ricalcolo, e uno stato nell'istanza ripartirebbe
// da "caricamento" a ogni tick senza mai arrivare a mostrare la risposta.
const STATE = {};

function log(msg) {
    try { console.log("[PZG] " + msg); } catch (e) { /* nessuna console */ }
}

function stateFor(sym) {
    if (!STATE[sym]) {
        STATE[sym] = { levels: [], status: "in attesa", at: 0, busy: false, transport: "", ageLabel: "" };
    }
    return STATE[sym];
}

function finish(st, levels, ageLabel, status) {
    st.levels = levels;
    st.ageLabel = ageLabel;
    st.status = status;
    st.busy = false;
}

function load(sym) {
    const st = stateFor(sym);
    const url = BASE_URL + "/tradovate/levels/" + sym + ".json?key=" + KEY;
    st.busy = true;
    st.at = Date.now();
    st.status = "richiesta inviata";

    const onData = function (d) {
        const levels = (d && d.levels) || [];
        log(sym + ": ricevuti " + levels.length + " livelli (" + ((d && d.age_label) || "") + ")");
        finish(st, levels, (d && d.age_label) || "", levels.length ? "" : "nessun livello disponibile");
    };
    const onFail = function (why) {
        log(sym + ": errore " + why);
        finish(st, st.levels, st.ageLabel, "errore: " + why);
    };

    try {
        if (typeof XMLHttpRequest === "function") {
            st.transport = "XHR";
            const x = new XMLHttpRequest();
            x.open("GET", url);
            x.onload = function () {
                try { onData(JSON.parse(x.responseText)); } catch (e) { onFail("risposta non valida (" + x.status + ")"); }
            };
            x.onerror = function () { onFail("XHR bloccata (CORS/CSP?)"); };
            x.ontimeout = function () { onFail("XHR scaduta"); };
            x.send();
        } else if (typeof fetch === "function") {
            st.transport = "fetch";
            fetch(url).then(function (r) { return r.json(); }).then(onData, function (e) {
                onFail("fetch: " + (e && e.message ? e.message : e));
            });
        } else {
            st.transport = "nessuno";
            onFail("rete non disponibile (ne' XMLHttpRequest ne' fetch)");
        }
        log(sym + ": richiesta partita con " + st.transport);
    } catch (e) {
        onFail("eccezione: " + (e && e.message ? e.message : e));
    }
}

class PolarisZeroGexLevels {
    init() {
        // niente stato qui: vedi STATE
    }

    map(d) {
        // Le linee sono infinite: basta disegnarle una volta, sull'ultima barra.
        if (!d.isLast()) {
            return {};
        }
        const price = d.value();
        const sym = price > 15000 ? "NQ" : "ES";
        const st = stateFor(sym);
        const now = Date.now();

        if (st.busy && now - st.at > TIMEOUT_MS) {
            // La richiesta non e' mai tornata: si sblocca e si dice come.
            st.busy = false;
            st.status = "nessuna risposta dopo " + Math.round(TIMEOUT_MS / 1000) + " s (" + st.transport + ")";
            log(sym + ": " + st.status);
        }
        if (!st.busy && now - st.at > REFRESH_MS) {
            load(sym);
        }

        const items = [];
        const x = du(d.index());
        st.levels.forEach(function (lv) {
            items.push({
                tag: "LineSegments",
                key: "pzg_line_" + lv.key,
                lines: [{
                    tag: "Line",
                    a: { x: du(0), y: du(lv.price) },
                    b: { x: du(1), y: du(lv.price) },
                    infiniteStart: true,
                    infiniteEnd: true
                }],
                lineStyle: { lineWidth: 1, color: lv.color, lineStyle: 3 },
                global: true
            });
            items.push({
                tag: "Text",
                key: "pzg_txt_" + lv.key,
                point: { x: op(x, "+", px(8)), y: du(lv.price) },
                text: lv.label,
                style: { fontSize: 11, fontWeight: "bold", fill: lv.color },
                textAlignment: "leftMiddle",
                global: true
            });
        });
        if (st.status) {
            items.push({
                tag: "Text",
                key: "pzg_status",
                point: { x: op(x, "+", px(8)), y: op(du(price), "-", px(20)) },
                text: "ZeroGEX " + sym + ": " + st.status,
                style: { fontSize: 11, fill: "#f59e0b" },
                textAlignment: "leftMiddle",
                global: true
            });
        }
        return { graphics: { items: items } };
    }
}

module.exports = {
    name: "polarisZeroGexLevels",
    description: "Polaris - livelli gamma ZeroGEX (ES/NQ)",
    calculator: PolarisZeroGexLevels,
    params: {},
    tags: ["Polaris"],
    areaChoice: meta.AreaChoice.SAME
};
