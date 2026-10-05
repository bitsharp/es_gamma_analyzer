// Polaris · livelli gamma ZeroGEX su Tradovate (indicatore custom, ES e NQ).
//
// Scarica i livelli da Polaris ogni 5 minuti e li disegna come linee orizzontali
// con etichetta "prezzo-nome" (7763.5-CallW). I livelli hanno almeno 15 minuti di
// ritardo. ES o NQ si riconosce da solo dal prezzo del grafico.
//
// Installazione: in Tradovate apri Code Explorer, File > New, incolla tutto questo
// file, salva, poi aggiungi l'indicatore "polarisZeroGexLevels" al grafico.
//
// Se accanto al prezzo compare la scritta "ZeroGEX: rete non disponibile" o
// "errore", Tradovate blocca le richieste verso domini esterni: in quel caso
// questa strada non e' percorribile.

const predef = require("./tools/predef");
const meta = require("./tools/meta");
const { px, du, op } = require("./tools/graphics");

const BASE_URL = "{{ base_url }}";
const KEY = "{{ token }}";
const REFRESH_MS = 5 * 60 * 1000;

function getJson(url) {
    return new Promise(function (resolve, reject) {
        if (typeof fetch === "function") {
            fetch(url).then(function (r) { return r.json(); }).then(resolve, reject);
        } else if (typeof XMLHttpRequest === "function") {
            const x = new XMLHttpRequest();
            x.open("GET", url);
            x.onload = function () {
                try { resolve(JSON.parse(x.responseText)); } catch (e) { reject(e); }
            };
            x.onerror = function () { reject(new Error("richiesta bloccata")); };
            x.send();
        } else {
            reject(new Error("rete non disponibile"));
        }
    });
}

class PolarisZeroGexLevels {
    init() {
        this.st = { levels: [], status: "caricamento...", at: 0, busy: false, sym: null, asOf: "" };
    }

    load(sym) {
        const st = this.st;
        st.busy = true;
        st.at = Date.now();
        st.sym = sym;
        try {
            getJson(BASE_URL + "/tradovate/levels/" + sym + ".json?key=" + KEY).then(function (d) {
                st.levels = (d && d.levels) || [];
                st.asOf = (d && d.age_label) || "";
                st.status = st.levels.length ? "" : "nessun livello disponibile";
                st.busy = false;
            }).catch(function (e) {
                st.status = "errore: " + (e && e.message ? e.message : e);
                st.busy = false;
            });
        } catch (e) {
            st.status = "errore: " + (e && e.message ? e.message : e);
            st.busy = false;
        }
    }

    map(d) {
        // Le linee sono infinite: basta disegnarle una volta, sull'ultima barra.
        if (!d.isLast()) {
            return {};
        }
        const st = this.st;
        const price = d.value();
        const sym = price > 15000 ? "NQ" : "ES";
        if (st.sym !== sym) {
            st.levels = [];
            st.at = 0;
        }
        if (!st.busy && Date.now() - st.at > REFRESH_MS) {
            this.load(sym);
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
                text: "ZeroGEX: " + st.status,
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
