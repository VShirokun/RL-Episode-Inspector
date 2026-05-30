const SHORTCUTS: [string, string][] = [
  ["Space", "Play / pause"],
  ["→ / ←", "Next / previous frame"],
  ["Shift + → / ←", "Jump ± 10 frames"],
  ["Home / End", "First / last frame"],
  ["Click chart", "Seek to that frame"],
  ["Drag chart", "Scrub through the episode"],
];

export function HelpModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  if (!open) return null;
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h3>Keyboard & mouse</h3>
        <table>
          <tbody>
            {SHORTCUTS.map(([k, v]) => (
              <tr key={k}>
                <td>
                  <kbd>{k}</kbd>
                </td>
                <td>{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <button className="button" onClick={onClose}>
          Close
        </button>
      </div>
    </div>
  );
}
