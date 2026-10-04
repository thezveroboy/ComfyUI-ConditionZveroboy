import { app } from "/scripts/app.js";

const PREFIX = "cond_";
const DYNAMIC_CLASSES = new Set([
  "CondAddZveroboy",
  "CondSubtractZveroboy",
  "CondAndZveroboy",
  "CondOrZveroboy",
  "CondXorZveroboy",
  "CondNandZveroboy",
  "CondNorZveroboy",
  "CondXnorZveroboy",
  "CondRemoveZveroboy",
]);

function isCondName(name) {
  return typeof name === "string" && name.startsWith(PREFIX);
}

function getCondInputs(node) {
  return (node.inputs || []).filter((i) => isCondName(i.name));
}

function ensureFreeSlot(node) {
  const inputs = getCondInputs(node);
  if (!inputs.length) return;
  const allConnected = inputs.every((inp) => inp.link != null);
  if (!allConnected) return;
  node.addInput(`${PREFIX}${inputs.length + 1}`, "CONDITIONING");
}

app.registerExtension({
  name: "zveroboy.condition.dynamic_inputs",
  async nodeCreated(node) {
    if (!DYNAMIC_CLASSES.has(node.comfyClass)) return;
    const existing = new Set((node.inputs || []).map((i) => i.name));
    for (let i = 1; i <= 2; i++) {
      const nm = `${PREFIX}${i}`;
      if (!existing.has(nm)) node.addInput(nm, "CONDITIONING");
    }
    const old = node.onConnectionsChange;
    node.onConnectionsChange = function () {
      if (old) old.apply(this, arguments);
      ensureFreeSlot(this);
    };
    ensureFreeSlot(node);
  },
});
