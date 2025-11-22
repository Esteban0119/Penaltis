let model;
let meta;
let mean, scale;
let inputOrder;

async function cargarModelo() {
    model = await tf.loadLayersModel("model.json");
    const metaResp = await fetch("penaltis_meta.json");
    meta = await metaResp.json();

    mean = meta.scaler_mean;
    scale = meta.scaler_scale;
    inputOrder = meta.input_order;
}

cargarModelo();

function normalizar(valor, media, desviacion) {
    return (valor - media) / desviacion;
}

async function predecir() {

    let velocidad = parseFloat(document.getElementById("velocidad").value);
    let angulo = parseFloat(document.getElementById("angulo").value);
    let distancia = parseFloat(document.getElementById("distancia").value);
    let presion = document.getElementById("presion").value;
    let pie = document.getElementById("pie").value;

    // VALIDACIÓN DE RANGOS
    if (velocidad < 60 || velocidad > 140) return alert("Velocidad fuera del rango (60–140)");
    if (angulo < 0 || angulo > 60) return alert("Ángulo fuera del rango (0–60)");
    if (distancia < 0.3 || distancia > 2) return alert("Distancia fuera del rango (0.3–2)");

    // One-Hot Encoding manual
    let Presion_Alta = presion === "Alta" ? 1 : 0;
    let Presion_Media = presion === "Media" ? 1 : 0;
    let Presion_Baja = presion === "Baja" ? 1 : 0;

    let Pie_Derecho = pie === "Derecho" ? 1 : 0;
    let Pie_Izquierdo = pie === "Izquierdo" ? 1 : 0;

    let input = [
        normalizar(velocidad, mean[0], scale[0]),
        normalizar(angulo, mean[1], scale[1]),
        normalizar(distancia, mean[2], scale[2]),

        Presion_Alta,
        Presion_Media,
        Presion_Baja,

        Pie_Derecho,
        Pie_Izquierdo
    ];

    const tensor = tf.tensor([input]);
    const pred = await model.predict(tensor).data();
    const prob = pred[0];

    document.getElementById("resultado").innerHTML =
        prob > 0.5 ? "🔥 Gol" : "❌ Fallado";
}
