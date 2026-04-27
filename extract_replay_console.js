// Paste this in your browser console (F12 → Console) while viewing the replay
// It will extract game state data and save it to a file

(async function extractReplayData() {
  console.log("🎮 Extracting Orbit Wars replay data...");

  // Try to find the game state in the page
  const gameStates = [];
  let currentTurn = 0;

  // Method 1: Look for React/Vue component state
  const allElements = document.querySelectorAll('[data-*], [class*="game"], [class*="state"]');
  console.log(`Found ${allElements.length} potential game state elements`);

  // Method 2: Monitor window for exposed objects
  const checkWindow = () => {
    const suspectObjects = [
      window.__REACT_DEVTOOLS_GLOBAL_HOOK__,
      window.__data,
      window.__state,
      window.__game,
      window.gameState,
    ];

    for (let obj of suspectObjects) {
      if (obj) {
        console.log("Found window object:", obj);
        return obj;
      }
    }
    return null;
  };

  const gameState = checkWindow();
  if (gameState) {
    console.log("Game state found:", gameState);
    return gameState;
  }

  // Method 3: Try to extract from iframe communication
  console.log("Looking for iframe communication...");

  // Listen for messages from iframe
  const messages = [];
  window.addEventListener('message', (event) => {
    messages.push({
      timestamp: new Date().toISOString(),
      data: event.data,
      source: event.origin
    });
    console.log("Captured message:", event.data);
  });

  // Method 4: Check localStorage/sessionStorage
  console.log("\n📦 Checking browser storage...");
  try {
    const localData = Object.keys(localStorage);
    const sessionData = Object.keys(sessionStorage);
    console.log("LocalStorage keys:", localData.filter(k => k.includes('game') || k.includes('replay') || k.includes('orbit')));
    console.log("SessionStorage keys:", sessionData.filter(k => k.includes('game') || k.includes('replay') || k.includes('orbit')));

    // Try to extract data
    for (let key of localData) {
      if (key.includes('game') || key.includes('orbit') || key.includes('episode')) {
        const value = localStorage.getItem(key);
        console.log(`\n✅ Found data in localStorage[${key}]:`);
        console.log(value.substring(0, 500));
      }
    }
  } catch (e) {
    console.log("Storage access failed:", e.message);
  }

  // Method 5: Try to extract from network tab data
  console.log("\n🌐 Trying to access performance entries...");
  const entries = performance.getEntries();
  const gameEndpoints = entries.filter(e =>
    e.name.includes('episode') ||
    e.name.includes('orbit') ||
    e.name.includes('api')
  );
  console.log("Found game-related endpoints:", gameEndpoints.map(e => e.name));

  return {
    status: "extraction_attempted",
    methods_tried: [
      "window_objects",
      "element_scanning",
      "iframe_messaging",
      "storage_inspection",
      "network_monitoring"
    ],
    messages_captured: messages
  };
})();

// Alternative simple method:
// If you can see the game board, try this to get visual state:
console.log("\n📸 Taking visual snapshots of current game state...");
console.log("Current URL parameters:", new URLSearchParams(window.location.search));
console.log("All data attributes on page:",
  Array.from(document.querySelectorAll('[data-turn], [data-score], [data-ships], [data-player]'))
    .map(el => ({
      tag: el.tagName,
      attributes: Object.fromEntries(
        Array.from(el.attributes)
          .filter(a => a.name.startsWith('data-'))
          .map(a => [a.name, a.value])
      ),
      text: el.textContent?.substring(0, 50)
    }))
);
