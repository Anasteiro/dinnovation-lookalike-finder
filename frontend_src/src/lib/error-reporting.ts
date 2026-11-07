// Reports an unexpected UI error to the browser console only (nothing leaves the machine).
export function reportUiError(error: unknown, context: Record<string, unknown> = {}) {
  console.error("[app]", error, context);
}
