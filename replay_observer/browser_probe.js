async () => {
  const lowerHit = (value) => {
    const s = String(value || "").toLowerCase();
    return (
      s.includes("episode") ||
      s.includes("replay") ||
      s.includes("orbit") ||
      s.includes("observation") ||
      s.includes("planets") ||
      s.includes("fleets")
    );
  };

  const storage = {};
  for (const storageName of ["localStorage", "sessionStorage"]) {
    storage[storageName] = [];
    try {
      const store = window[storageName];
      for (let i = 0; i < store.length; i += 1) {
        const key = store.key(i);
        const value = store.getItem(key);
        if (lowerHit(key) || lowerHit(value)) {
          storage[storageName].push({
            key,
            preview: String(value).slice(0, 2000),
            length: String(value).length,
          });
        }
      }
    } catch (err) {
      storage[storageName].push({ error: String(err) });
    }
  }

  const performanceEntries = performance
    .getEntries()
    .map((entry) => ({
      name: entry.name,
      initiatorType: entry.initiatorType,
      duration: entry.duration,
      transferSize: entry.transferSize,
    }))
    .filter((entry) => lowerHit(entry.name));

  const canvases = Array.from(document.querySelectorAll("canvas")).map((canvas, idx) => {
    const rect = canvas.getBoundingClientRect();
    return {
      index: idx,
      width: canvas.width,
      height: canvas.height,
      clientWidth: rect.width,
      clientHeight: rect.height,
      x: rect.x,
      y: rect.y,
      className: canvas.className || "",
      id: canvas.id || "",
    };
  });

  const dataElements = Array.from(document.querySelectorAll("*"))
    .slice(0, 5000)
    .map((el) => {
      const attrs = {};
      for (const attr of Array.from(el.attributes || [])) {
        if (attr.name.startsWith("data-") || lowerHit(attr.name) || lowerHit(attr.value)) {
          attrs[attr.name] = attr.value;
        }
      }
      const text = (el.textContent || "").trim();
      if (!Object.keys(attrs).length && !lowerHit(text)) {
        return null;
      }
      return {
        tag: el.tagName,
        attrs,
        text: text.slice(0, 500),
      };
    })
    .filter(Boolean)
    .slice(0, 300);

  const windowKeys = Object.keys(window)
    .filter((key) => lowerHit(key))
    .slice(0, 200);

  return {
    url: location.href,
    title: document.title,
    storage,
    performanceEntries,
    canvases,
    dataElements,
    windowKeys,
  };
}
