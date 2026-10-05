// Polaris · livelli gamma ZeroGEX su Tradovate (indicatore custom, ES e NQ).
//
// File generato da Polaris: i livelli sono scritti qui dentro, nessuna richiesta
// di rete (Tradovate le blocca). Ogni mattina si genera di nuovo da Polaris
// ("Copia indicatore Tradovate"), si apre questo file in Code Explorer, si
// seleziona tutto, si incolla e si salva.
//
// ES o NQ si riconosce da solo dal prezzo del grafico. I livelli hanno almeno
// 15 minuti di ritardo rispetto al mercato.

const meta = require("./tools/meta");
const { px, du, op } = require("./tools/graphics");

const LEVELS = {{ levels_json|safe }};

class PolarisZeroGexLevels {
    init() {
        // niente stato: i livelli sono costanti
    }

    map(d) {
        // Le linee sono infinite: basta disegnarle una volta, sull'ultima barra.
        if (!d.isLast()) {
            return {};
        }
        const price = d.value();
        const sym = price > 15000 ? "NQ" : "ES";
        const set = LEVELS[sym] || { levels: [], asOf: "" };
        const x = du(d.index());
        const items = [];

        set.levels.forEach(function (lv) {
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
