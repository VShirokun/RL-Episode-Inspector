export function LoadingState({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="state loading" role="status">
      <div className="spinner" />
      <span>{label}</span>
    </div>
  );
}
