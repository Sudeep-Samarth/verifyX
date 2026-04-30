"use client"

import { cn } from "@/lib/utils"

type Theme = {
  border: string
  surface: string
  text: string
  textSoft: string
  muted: string
  borderSubtle: string
}

export function SystemTrustRing({
  pct,
  label,
  t,
  isDark,
}: {
  pct: number
  label: string
  t: Theme
  isDark: boolean
}) {
  const p = Math.max(0, Math.min(100, pct))
  const deg = (p / 100) * 360
  return (
    <div className="flex flex-col items-center gap-2">
      <div
        className="relative flex size-24 items-center justify-center rounded-full sm:size-28"
        style={{
          background: `conic-gradient(${isDark ? "#22c55e" : "#16a34a"} ${deg}deg, ${t.border} ${deg}deg 360deg)`,
        }}
      >
        <div
          className="flex size-[82%] flex-col items-center justify-center rounded-full sm:size-[84%]"
          style={{ backgroundColor: t.surface }}
        >
          <span className="text-xl font-bold tabular-nums sm:text-2xl" style={{ color: t.text }}>
            {Math.round(p)}%
          </span>
          <span className="text-[10px] font-medium uppercase tracking-wide" style={{ color: t.muted }}>
            {label}
          </span>
        </div>
      </div>
    </div>
  )
}

export function QueryRagasPanel({
  artifact,
  trustGate,
  confidence,
  t,
  isDark,
}: {
  artifact: Record<string, unknown> | null
  trustGate?: string
  confidence?: number
  t: Theme
  isDark: boolean
}) {
  if (!artifact) return null
  const tb = artifact.trust_breakdown as Record<string, number> | undefined
  const ragas = artifact.ragas_scorecard as Record<string, number> | undefined
  const claims = (artifact.claims as Array<Record<string, unknown>>) || []
  const trustPct = typeof confidence === "number" ? confidence * 100 : 0

  const retrieval = (artifact.retrieval_explanation as Array<Record<string, unknown>>) || []

  const versionHistory = (() => {
    const vh = artifact.version_history as Array<Record<string, unknown>> | undefined
    if (vh && vh.length > 0) return vh
    const seen = new Set<string>()
    const out: Array<Record<string, unknown>> = []
    for (const r of retrieval) {
      const doc = String(r.doc_id ?? "")
      const ed = String(r.edition_date ?? "")
      const key = `${doc}|${ed}`
      if (!doc && !ed) continue
      if (seen.has(key)) continue
      seen.add(key)
      out.push({
        document_label: doc || ed || "Source",
        doc_id: r.doc_id,
        edition_date: r.edition_date,
        report_type: r.report_type,
        sample_section: r.section,
      })
    }
    return out
  })()

  const supportingEvidence = (() => {
    const se = artifact.supporting_evidence as Array<Record<string, unknown>> | undefined
    if (se && se.length > 0) return se
    return retrieval.map((r, i) => ({
      rank: i + 1,
      doc_id: r.doc_id,
      edition_date: r.edition_date,
      section: r.section,
      page: r.page,
      rerank_score: r.rerank_score,
      excerpt: typeof r.why_retrieved === "string" ? r.why_retrieved : "",
    }))
  })()

  return (
    <div
      className="mb-6 space-y-4 rounded-2xl border p-4 sm:p-5"
      style={{ borderColor: t.border, backgroundColor: t.surface }}
    >
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h2 className="text-lg font-semibold tracking-tight" style={{ color: t.text }}>
            Query analysis (RAG + XAI)
          </h2>
          <p className="mt-1 text-sm" style={{ color: t.muted }}>
            Trust gate, deterministic breakdown, and claim-level checks (RAGAS-style scorecard when enabled on
            backend).
          </p>
        </div>
        <SystemTrustRing pct={trustPct} label="Trust" t={t} isDark={isDark} />
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <span
          className="rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-wide"
          style={{ borderColor: t.border, color: t.text }}
        >
          {trustGate ?? (artifact.trust_gate as string) ?? "—"}
        </span>
      </div>

      {tb ? (
        <div className="space-y-2">
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {(
              [
                ["Raw score (NLI blend)", tb.raw_score],
                ["Unattrib. penalty", tb.unattributed_penalty],
                ["Conflict penalty", tb.conflict_penalty],
                ["Completeness penalty", tb.completeness_penalty],
                ["RAG context boost", tb.rag_context_boost],
                ["Linear (pre-norm)", tb.linear_score],
                ["Final (normalized)", tb.final_score],
              ] as const
            ).map(([k, v]) => (
              <div
                key={String(k)}
                className="rounded-xl border px-3 py-2.5"
                style={{ borderColor: t.borderSubtle, backgroundColor: isDark ? "rgba(0,0,0,0.2)" : "#fafafa" }}
              >
                <p className="text-[11px] font-medium uppercase tracking-wide" style={{ color: t.muted }}>
                  {k}
                </p>
                <p className="mt-1 font-mono text-sm font-semibold tabular-nums" style={{ color: t.text }}>
                  {typeof v === "number" && !Number.isNaN(v) ? v.toFixed(4) : "—"}
                </p>
                {k === "Final (normalized)" && typeof v === "number" && !Number.isNaN(v) ? (
                  <p className="mt-0.5 text-[11px] font-medium tabular-nums" style={{ color: t.textSoft }}>
                    ≈ {Math.round(v * 100)}% trust
                  </p>
                ) : null}
              </div>
            ))}
          </div>
          <p className="text-[11px] leading-snug" style={{ color: t.muted }}>
            Penalties are softened for hybrid RAG (retrieved context is treated as a positive prior). Tune via{" "}
            <span className="font-mono">XAI_AGG_*</span> env vars on the backend if needed.
          </p>
        </div>
      ) : null}

      {versionHistory.length > 0 ? (
        <div>
          <h3 className="mb-2 text-sm font-semibold" style={{ color: t.text }}>
            Version history (retrieved corpora)
          </h3>
          <ul className="space-y-2">
            {versionHistory.map((row, idx) => (
              <li
                key={`${String(row.doc_id)}-${String(row.edition_date)}-${idx}`}
                className="rounded-xl border px-3 py-2.5 text-[13px]"
                style={{ borderColor: t.borderSubtle, backgroundColor: isDark ? "rgba(0,0,0,0.15)" : "#f4f4f5" }}
              >
                <p className="font-medium" style={{ color: t.text }}>
                  {String(row.document_label ?? row.doc_id ?? "Edition")}
                </p>
                <p className="mt-1 text-[12px]" style={{ color: t.muted }}>
                  {[row.report_type, row.edition_date].filter(Boolean).join(" · ")}
                  {row.sample_section ? ` · ${String(row.sample_section)}` : ""}
                </p>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {supportingEvidence.length > 0 ? (
        <div className="overflow-x-auto">
          <h3 className="mb-2 text-sm font-semibold" style={{ color: t.text }}>
            Supporting evidence (retrieved passages)
          </h3>
          <table className="w-full min-w-[560px] border-collapse text-left text-[12px]">
            <thead>
              <tr style={{ borderBottom: `1px solid ${t.border}` }}>
                <th className="py-2 pr-2 font-semibold" style={{ color: t.muted }}>
                  #
                </th>
                <th className="py-2 pr-3 font-semibold" style={{ color: t.muted }}>
                  Source
                </th>
                <th className="py-2 pr-3 font-semibold" style={{ color: t.muted }}>
                  Section / page
                </th>
                <th className="py-2 font-semibold" style={{ color: t.muted }}>
                  Excerpt
                </th>
              </tr>
            </thead>
            <tbody>
              {supportingEvidence.slice(0, 12).map((row) => (
                <tr key={String(row.rank)} style={{ borderBottom: `1px solid ${t.borderSubtle}` }}>
                  <td className="py-2 pr-2 align-top font-mono tabular-nums" style={{ color: t.muted }}>
                    {row.rank as number}
                  </td>
                  <td className="max-w-[180px] py-2 pr-3 align-top" style={{ color: t.textSoft }}>
                    <span className="line-clamp-3">{String(row.doc_id ?? "—")}</span>
                    {row.edition_date ? (
                      <span className="mt-0.5 block text-[11px]" style={{ color: t.muted }}>
                        {String(row.edition_date)}
                      </span>
                    ) : null}
                  </td>
                  <td className="max-w-[140px] py-2 pr-3 align-top text-[11px]" style={{ color: t.muted }}>
                    {row.section != null && row.section !== "" ? String(row.section) : "—"}
                    {row.page != null && row.page !== "" ? ` · p.${String(row.page)}` : ""}
                    {typeof row.rerank_score === "number" ? (
                      <span className="mt-0.5 block font-mono">CE {row.rerank_score.toFixed(3)}</span>
                    ) : null}
                  </td>
                  <td className="max-w-[min(48vw,28rem)] py-2 align-top leading-relaxed" style={{ color: t.textSoft }}>
                    {String(row.excerpt || "").slice(0, 500)}
                    {String(row.excerpt || "").length > 500 ? "…" : ""}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}

      {ragas ? (
        <div>
          <h3 className="mb-2 text-sm font-semibold" style={{ color: t.text }}>
            RAGAS-style scorecard
          </h3>
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {Object.entries(ragas).map(([key, val]) => (
              <div
                key={key}
                className="rounded-xl border px-3 py-2"
                style={{ borderColor: t.borderSubtle, backgroundColor: isDark ? "rgba(0,0,0,0.15)" : "#f4f4f5" }}
              >
                <p className="text-[11px] capitalize" style={{ color: t.muted }}>
                  {key.replace(/_/g, " ")}
                </p>
                <p className="font-mono text-sm font-semibold tabular-nums" style={{ color: t.text }}>
                  {typeof val === "number" ? val.toFixed(4) : String(val)}
                </p>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <p className="text-xs" style={{ color: t.muted }}>
          Enable RAGAS on the backend (<span className="font-mono">XAI_RAGAS=1</span>) to populate the scorecard.
        </p>
      )}

      {claims.length > 0 ? (
        <div className="overflow-x-auto">
          <h3 className="mb-2 text-sm font-semibold" style={{ color: t.text }}>
            Claims &amp; verification
          </h3>
          <table className="w-full min-w-[520px] border-collapse text-left text-[13px]">
            <thead>
              <tr style={{ borderBottom: `1px solid ${t.border}` }}>
                <th className="py-2 pr-3 font-semibold" style={{ color: t.muted }}>
                  Claim
                </th>
                <th className="py-2 pr-3 font-semibold" style={{ color: t.muted }}>
                  NLI
                </th>
                <th className="py-2 font-semibold" style={{ color: t.muted }}>
                  Source
                </th>
              </tr>
            </thead>
            <tbody>
              {claims.slice(0, 12).map((c) => {
                const nli = c.nli as Record<string, unknown> | undefined
                const attr = c.attribution as Record<string, unknown> | undefined
                return (
                  <tr key={String(c.id)} style={{ borderBottom: `1px solid ${t.borderSubtle}` }}>
                    <td className="max-w-[240px] py-2 pr-3 align-top" style={{ color: t.textSoft }}>
                      {(c.text as string)?.slice(0, 200)}
                      {(c.text as string)?.length > 200 ? "…" : ""}
                    </td>
                    <td className="py-2 pr-3 align-top">
                      <span
                        className={cn(
                          "rounded-md px-2 py-0.5 text-[11px] font-semibold uppercase",
                          nli?.label === "entailment" && "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400",
                          nli?.label === "contradiction" && "bg-red-500/15 text-red-600 dark:text-red-400",
                          nli?.label === "neutral" && "bg-amber-500/15 text-amber-700 dark:text-amber-400"
                        )}
                      >
                        {String(nli?.label ?? "—")}
                      </span>
                      {typeof nli?.confidence === "number" ? (
                        <span className="ml-1 font-mono text-[11px]" style={{ color: t.muted }}>
                          {(nli.confidence as number).toFixed(2)}
                        </span>
                      ) : null}
                    </td>
                    <td className="py-2 align-top text-[12px]" style={{ color: t.muted }}>
                      {attr?.is_attributed
                        ? `${attr.source_doc ?? ""} · ${attr.source_section ?? ""}`
                        : "Unattributed"}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  )
}

export function BrdAnalysisPanel({
  data,
  t,
  isDark,
}: {
  data: Record<string, unknown> | null
  t: Theme
  isDark: boolean
}) {
  if (!data) return null
  const score = data.compliance_score as number | undefined
  const status = data.trust_status as string | undefined
  const reqs = (data.requirements as Array<Record<string, unknown>>) || []

  return (
    <div
      className="mb-6 space-y-4 rounded-2xl border p-4 sm:p-5"
      style={{ borderColor: t.border, backgroundColor: t.surface }}
    >
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h2 className="text-lg font-semibold tracking-tight" style={{ color: t.text }}>
            BRD analysis
          </h2>
          <p className="mt-1 text-sm" style={{ color: t.muted }}>
            Pipeline output: compliance score, trust status, and requirement rollup (see backend BRD workflow).
          </p>
        </div>
        <SystemTrustRing
          pct={typeof score === "number" ? Math.min(100, score) : 0}
          label="Compliance"
          t={t}
          isDark={isDark}
        />
      </div>
      <div className="flex flex-wrap gap-2">
        <span
          className="rounded-full border px-3 py-1 text-xs font-bold uppercase tracking-wide"
          style={{ borderColor: t.border, color: t.text }}
        >
          {status ?? "—"}
        </span>
        <span className="text-sm" style={{ color: t.muted }}>
          {reqs.length} requirement group(s)
        </span>
      </div>

      {reqs.length > 0 ? (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[480px] border-collapse text-left text-[13px]">
            <thead>
              <tr style={{ borderBottom: `1px solid ${t.border}` }}>
                <th className="py-2 pr-3 font-semibold" style={{ color: t.muted }}>
                  Requirement
                </th>
                <th className="py-2 pr-3 font-semibold" style={{ color: t.muted }}>
                  Status
                </th>
                <th className="py-2 font-semibold" style={{ color: t.muted }}>
                  Risk
                </th>
              </tr>
            </thead>
            <tbody>
              {reqs?.slice(0, 15).map((r, i) => (
                <tr key={String(r?.req_id ?? i)} style={{ borderBottom: `1px solid ${t.borderSubtle}` }}>
                  <td className="max-w-[280px] py-2 pr-3 align-top" style={{ color: t.textSoft }}>
                    {((r?.req_text || r?.text) as string)?.slice(0, 160)}
                    {String(r?.req_text || r?.text || "").length > 160 ? "…" : ""}
                  </td>
                  <td className="py-2 pr-3 align-top text-[12px] font-medium" style={{ color: t.text }}>
                    {String(r?.status ?? r?.rolled_up_status ?? "—")}
                  </td>
                  <td className="py-2 align-top text-[12px]" style={{ color: t.muted }}>
                    {String(r?.risk_level ?? r?.risk ?? "—")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  )
}

export function ComplianceAssistantLog({
  lines,
  onClear,
  t,
}: {
  lines: string[]
  onClear: () => void
  t: Theme
}) {
  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
      <div className="mb-3 flex shrink-0 items-center justify-between gap-2">
        <h2 className="text-[15px] font-semibold tracking-tight" style={{ color: t.text }}>
          Compliance assistant
        </h2>
        <button
          type="button"
          onClick={onClear}
          className="text-[11px] font-medium underline-offset-2 hover:underline"
          style={{ color: t.muted }}
        >
          Clear log
        </button>
      </div>
      <div
        className="min-h-0 flex-1 overflow-y-auto rounded-xl border p-3 font-mono text-[11px] leading-relaxed sm:text-xs"
        style={{
          borderColor: t.border,
          backgroundColor: t.surface,
          color: t.textSoft,
        }}
      >
        {lines.length === 0 ? (
          <p style={{ color: t.muted }}>Run a query or BRD analysis to see structured logs here.</p>
        ) : (
          lines.map((line, i) => (
            <pre key={i} className="mb-3 whitespace-pre-wrap break-words last:mb-0">
              {line}
            </pre>
          ))
        )}
      </div>
    </div>
  )
}
