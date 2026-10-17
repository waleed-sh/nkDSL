/* Exercise the copy control against nbconvert's input markup without a browser. */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const script = readFileSync(new URL("../../docs/javascripts/notebook-copy.js", import.meta.url), "utf8");

async function runCase({ secure, denied = false }) {
  const source = "print('ψ = 2')\nprint(3)\n";
  let copied = null;
  let nativeRemoved = false;
  let selectionUsed = false;
  const children = [];
  const makeElement = () => ({
    listeners: {}, attributes: {},
    style: {},
    setAttribute(name, value) { this.attributes[name] = value; },
    addEventListener(name, handler) { this.listeners[name] = handler; },
    select() { copied = this.value; },
    remove() {},
  });
  const area = {
    querySelector(selector) {
      if (selector === "pre") return { textContent: source };
      return children.find((child) => child.className === "notebook-copy");
    },
    append(...items) { children.push(...items); },
    closest() { return { querySelector() { return { remove() { nativeRemoved = true; } }; } }; },
  };
  const document = {
    readyState: "complete",
    querySelectorAll(selector) {
      assert.ok(selector.includes(".jp-CodeCell .jp-InputArea .highlight-ipynb"));
      return [area];
    },
    createElement: makeElement,
    body: { append() {} },
    execCommand(command) { assert.equal(command, "copy"); return !denied; },
    createRange() { return { selectNodeContents() { selectionUsed = true; } }; },
  };
  const context = {
    document,
    navigator: { clipboard: { async writeText(text) {
      if (denied) throw new Error("Denied");
      copied = text;
    } } },
    window: {
      isSecureContext: secure,
      setTimeout() {},
      getSelection() { return { removeAllRanges() {}, addRange() {} }; },
    },
  };
  vm.runInNewContext(script, context);
  const button = children.find((child) => child.className === "notebook-copy");
  assert.equal(button.attributes["aria-label"], "Copy code cell");
  assert.equal(button.type, "button");
  assert.equal(nativeRemoved, true);
  await button.listeners.click();
  if (denied) {
    assert.equal(button.textContent, "Select code");
    assert.equal(selectionUsed, true);
  } else {
    assert.equal(copied, source);
    assert.equal(button.textContent, "Copied");
  }
  // Running initialization again must leave exactly one button on the cell.
  vm.runInNewContext(script, context);
  assert.equal(children.filter((child) => child.className === "notebook-copy").length, 1);
}

await runCase({ secure: true });
await runCase({ secure: false });
await runCase({ secure: true, denied: true });
console.log("Notebook copy controls pass clipboard, fallback, accessibility, and repeated-init checks.");
