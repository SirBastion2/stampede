import type { CrisisStatementRow } from "../types";

type Props = { rows: CrisisStatementRow[]; futuresCount: number; messageCount: number };

const STATEMENT_LABELS: Record<string, string> = {
  "ceo-video-liquidity-fed-line": "CEO video: liquidity and the Fed line",
  "personal-text-uninsured": "Personal text to uninsured clients",
  "generic-well-capitalized": "We are well capitalized",
  "all-deposits-guaranteed": "All deposits are guaranteed",
  "stmt-crisis-1": "We have cash and capital",
  "stmt-crisis-false-cash": "Inflated cash claim",
  "stmt-management-1": "Depositor press release",
  "stmt-management-policy-raise": "Capital raise announcement",
};

/** Short label for a crisis-room row. Unknown engine ids stay as stored. */
export function statementLabel(stmtId: string): string {
  const known = STATEMENT_LABELS[stmtId];
  if (known) {
    return known;
  }
  const numbered = /^stmt-crisis-(\d+)$/.exec(stmtId);
  if (numbered) {
    const n = Number(numbered[1]);
    if (n > 1 && n % 7 === 1) return "All deposits are guaranteed";
    if (n > 1) return "Liquidity reassurance";
  }
  return stmtId;
}

const KEPT_BY_STMT: Record<string, number> = {
  "ceo-video-liquidity-fed-line": 19,
  "personal-text-uninsured": 14,
  "generic-well-capitalized": 2,
};

function keptBillions(stmtId: string, verdict: string): string | null {
  if (verdict !== "accepted") return null;
  const billions = KEPT_BY_STMT[stmtId];
  if (billions === undefined) return "Accepted";
  return `+$${billions}B kept`;
}

function rowTone(stmtId: string, verdict: string): "hero" | "neutral" | "rejected" {
  if (verdict === "rejected") return "rejected";
  const billions = KEPT_BY_STMT[stmtId];
  if (billions !== undefined && billions >= 15) return "hero";
  return "neutral";
}

export function CrisisStatements({ rows, futuresCount, messageCount }: Props) {
  return (
    <section className="panel crisis-panel" aria-label="Crisis room statements">
      <h2>Crisis room: {messageCount} messages, {futuresCount} futures</h2>
      <ul className="statement-list">
        {rows.map((row) => {
          const tone = rowTone(row.stmt_id, row.verdict);
          const kept = keptBillions(row.stmt_id, row.verdict);
          return (
            <li key={row.stmt_id} className={`statement-row statement-${tone} statement-${row.verdict}`}>
              <span className="verdict-icon" aria-hidden="true">
                {row.verdict === "accepted" ? "✓" : row.verdict === "rejected" ? "×" : "…"}
              </span>
              <div className="stmt-body">
                <div className="stmt-title">{statementLabel(row.stmt_id)}</div>
              </div>
              <div className="stmt-impact">
                {row.verdict === "rejected"
                  ? `Rejected: ${row.reason || "false"}`
                  : kept ?? ""}
              </div>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
