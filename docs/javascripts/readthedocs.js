/* Use the supported RTD Addons event to populate a keyboard-accessible selector. */
document.addEventListener("readthedocs-addons-data-ready", (event) => {
  const data = event.detail.data();
  if (!data.versions || !data.versions.active) return;
  document.querySelector(".rtd-version")?.remove();
  const label = document.createElement("label");
  label.className = "rtd-version";
  label.textContent = "Version ";
  const select = document.createElement("select");
  select.setAttribute("aria-label", "Documentation version");
  data.versions.active.forEach((version) => {
    const option = document.createElement("option");
    option.textContent = version.slug;
    option.value = version.urls.documentation;
    option.selected = version.slug === data.versions.current.slug;
    select.append(option);
  });
  select.addEventListener("change", () => { window.location.href = select.value; });
  label.append(select);
  document.querySelector(".md-header__inner")?.append(label);
});
