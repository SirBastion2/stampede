/** Turn M34 reason codes into a sentence a viewer can act on. */

export function plainModerationError(reasons: string[] | undefined, fallback?: string): string {
  const list = reasons ?? [];
  const joined = list.join(" ").toLowerCase();
  if (joined.includes("blocked bank name") || joined.includes("blocked ticker")) {
    return "Real bank names are blocked. This demo only simulates fictional banks — try Kestrel Bank.";
  }
  if (joined.includes("slur")) {
    return "That wording is blocked. Rephrase the rumor without a slur.";
  }
  if (joined.includes("real-person") || joined.includes("real person")) {
    return "Real people's names are blocked. Use a fictional depositor.";
  }
  if (list.length > 0) {
    return "That rumor was blocked. Try a fictional bank such as Kestrel Bank.";
  }
  return fallback ?? "That rumor was blocked. Try a fictional bank such as Kestrel Bank.";
}
