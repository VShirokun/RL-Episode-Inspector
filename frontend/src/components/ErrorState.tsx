export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="state error" role="alert" data-testid="error-state">
      <strong>Something went wrong</strong>
      <p>{message}</p>
      {onRetry && (
        <button className="button" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}
