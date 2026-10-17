/* Notebook inputs use nbconvert markup, so Material's code buttons do not apply. */
(() => {
  async function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return;
    }
    const input = document.createElement("textarea");
    input.value = text;
    input.style.position = "fixed";
    input.style.opacity = "0";
    document.body.append(input);
    input.select();
    const copied = document.execCommand("copy");
    input.remove();
    if (!copied) throw new Error("Clipboard access was denied");
  }

  function addCopyButtons() {
    document.querySelectorAll(".jp-CodeCell .jp-InputArea .highlight-ipynb, .jp-CodeCell .jp-InputArea .highlight").forEach((area) => {
      const source = area.querySelector("pre");
      if (!source || area.querySelector(".notebook-copy")) return;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "notebook-copy";
      button.textContent = "Copy";
      button.setAttribute("aria-label", "Copy code cell");
      const status = document.createElement("span");
      status.className = "notebook-copy-status";
      status.setAttribute("role", "status");
      button.addEventListener("click", async () => {
        try {
          await copyText(source.textContent);
          button.textContent = "Copied";
          status.textContent = "Code copied to clipboard";
        } catch {
          button.textContent = "Select code";
          const selection = window.getSelection();
          const range = document.createRange();
          range.selectNodeContents(source);
          selection.removeAllRanges();
          selection.addRange(range);
          status.textContent = "Select and copy the highlighted code";
        }
        window.setTimeout(() => { button.textContent = "Copy"; }, 2000);
      });
      area.append(button, status);
      area.closest(".jp-InputArea-editor")?.querySelector(".zeroclipboard-container")?.remove();
    });
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", addCopyButtons);
  } else {
    addCopyButtons();
  }
})();
