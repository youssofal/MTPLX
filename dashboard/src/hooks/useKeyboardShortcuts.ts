import { useEffect } from "react";
import { useDashboardStore } from "../state/store";
import type { TabId } from "../components/Shell";

const TAB_ORDER: TabId[] = [
  "overview",
  "speculative",
  "cache",
  "memory",
  "thermal",
  "requests",
  "systems",
  "settings",
];

export function useKeyboardShortcuts(setActive: (tab: TabId) => void): void {
  const cycleTheme = useDashboardStore((s) => s.cycleTheme);
  const togglePause = useDashboardStore((s) => s.togglePauseStream);
  const toggleSound = useDashboardStore((s) => s.toggleSound);

  useEffect(() => {
    function handler(e: KeyboardEvent) {
      // Avoid hijacking when the user is typing in an input/textarea.
      const target = e.target as HTMLElement | null;
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      switch (e.key) {
        case "t":
          cycleTheme();
          break;
        case " ":
          e.preventDefault();
          togglePause();
          break;
        case "s":
          toggleSound();
          break;
        case "g": {
          const idx = TAB_ORDER.findIndex(
            (id) => id === (document.body.dataset.activeTab as TabId | undefined),
          );
          const next = TAB_ORDER[(idx + 1) % TAB_ORDER.length];
          setActive(next);
          break;
        }
        default:
          break;
      }
    }
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [cycleTheme, togglePause, toggleSound, setActive]);
}
