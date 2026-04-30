"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { useTheme } from "next-themes"
import { motion } from "framer-motion"
import {
  Activity,
  ArrowRight,
  ArrowUp,
  Bot,
  ChevronRight,
  FileText,
  LogOut,
  MessageSquare,
  Plus,
  Settings,
  Share2,
  Shield,
  Sparkles,
  Upload,
  Zap,
} from "lucide-react"

import {
  BrdAnalysisPanel,
  ComplianceAssistantLog,
  QueryRagasPanel,
} from "@/components/compliance-panels"
import { cn } from "@/lib/utils"
import { createClient } from "@/lib/supabase/client"
import { logout } from "@/app/actions/auth"

const ACCENT_ORANGE = "#c9a574"

const SIDEBAR_W = "w-[300px] min-w-[300px] max-w-[300px]"
const RIGHT_W = "w-[min(100%,320px)] min-w-[280px] max-w-[320px]"
const CENTER_MAX = "max-w-7xl"

const BRD_ACCEPT =
  ".pdf,.doc,.docx,.txt,text/plain,application/pdf,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document"

interface Chat {
  id: string
  title: string
  created_at: string
  updated_at?: string
  mode?: string
}

type ChatSnapshot = {
  lastQueryArtifact?: Record<string, unknown> | null
  lastQueryMeta?: { trust_gate?: string; confidence?: number } | null
  lastBrdReport?: Record<string, unknown> | null
  assistantLog?: string[]
}

type ChatMessage = {
  id: string
  role: "user" | "assistant"
  content: string
  pending?: boolean
}

type ThemePalette = {
  bg: string
  surface: string
  muted: string
  border: string
  text: string
  textSoft: string
  navHover: string
  activeNav: string
  eyeSocket: string
  pupil: string
  chipBg: string
  menuBg: string
  borderSubtle: string
  shadow: string
  logoBadge: string
  logoBadgeText: string
}

const PALETTE_DARK: ThemePalette = {
  bg: "#000000",
  surface: "#212121",
  muted: "#9B9B9B",
  border: "#2a2a2a",
  text: "#ffffff",
  textSoft: "rgba(255,255,255,0.9)",
  navHover: "rgba(255,255,255,0.05)",
  activeNav: "#2a2a3a",
  eyeSocket: "rgba(0,0,0,0.35)",
  pupil: "#ffffff",
  chipBg: "#2a2a2a",
  menuBg: "#181818",
  borderSubtle: "rgba(255,255,255,0.08)",
  shadow: "0 8px 40px rgba(0,0,0,0.65)",
  logoBadge: "#e5e5e5",
  logoBadgeText: "#000000",
}

const PALETTE_LIGHT: ThemePalette = {
  bg: "#f4f4f5",
  surface: "#ffffff",
  muted: "#71717a",
  border: "#e4e4e7",
  text: "#18181b",
  textSoft: "#27272a",
  navHover: "rgba(0,0,0,0.05)",
  activeNav: "#e0e7ff",
  eyeSocket: "rgba(0,0,0,0.1)",
  pupil: "#18181b",
  chipBg: "#e4e4e7",
  menuBg: "#ffffff",
  borderSubtle: "rgba(0,0,0,0.08)",
  shadow: "0 8px 32px rgba(0,0,0,0.08)",
  logoBadge: "#18181b",
  logoBadgeText: "#fafafa",
}

function useDashboardPalette(): { t: ThemePalette; isDark: boolean; mounted: boolean } {
  const { resolvedTheme } = useTheme()
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  const isDark = !mounted || resolvedTheme !== "light"
  const t = isDark ? PALETTE_DARK : PALETTE_LIGHT
  return { t, isDark, mounted }
}

function ThemeSwitch({ checked, onChange }: { checked: boolean; onChange: () => void }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={onChange}
      className={cn(
        "relative h-7 w-11 shrink-0 rounded-full transition-colors duration-300 ease-out",
        checked ? "bg-[#3b82f6]" : "bg-zinc-400 dark:bg-zinc-600"
      )}
    >
      <span
        className={cn(
          "pointer-events-none absolute left-0.5 top-1/2 size-5 -translate-y-1/2 rounded-full bg-white shadow-sm transition-transform duration-300 ease-[cubic-bezier(0.34,1.56,0.64,1)]",
          checked ? "translate-x-5" : "translate-x-0"
        )}
      />
    </button>
  )
}

function clampOffset(dx: number, dy: number, max: number) {
  const d = Math.hypot(dx, dy) || 1
  const nx = (dx / d) * Math.min(max, d * 0.12)
  const ny = (dy / d) * Math.min(max, d * 0.12)
  return { x: nx, y: ny }
}

function TrackingEyes({ surface, eyeSocket, pupil }: Pick<ThemePalette, "surface" | "eyeSocket" | "pupil">) {
  const leftSocketRef = useRef<HTMLDivElement>(null)
  const rightSocketRef = useRef<HTMLDivElement>(null)
  const [left, setLeft] = useState({ x: 0, y: 0 })
  const [right, setRight] = useState({ x: 0, y: 0 })

  const onMove = useCallback((e: MouseEvent) => {
    const ls = leftSocketRef.current
    const rs = rightSocketRef.current
    if (!ls || !rs) return

    const lr = ls.getBoundingClientRect()
    const rr = rs.getBoundingClientRect()
    const lcx = lr.left + lr.width / 2
    const lcy = lr.top + lr.height / 2
    const rcx = rr.left + rr.width / 2
    const rcy = rr.top + rr.height / 2

    setLeft(clampOffset(e.clientX - lcx, e.clientY - lcy, 5))
    setRight(clampOffset(e.clientX - rcx, e.clientY - rcy, 5))
  }, [])

  useEffect(() => {
    window.addEventListener("mousemove", onMove, { passive: true })
    return () => window.removeEventListener("mousemove", onMove)
  }, [onMove])

  return (
    <div
      className="mb-8 flex size-14 items-center justify-center rounded-full sm:size-16"
      style={{ backgroundColor: surface }}
      aria-hidden
    >
      <div className="flex items-center gap-3 sm:gap-[14px]">
        <div
          ref={leftSocketRef}
          className="relative flex size-[17px] items-center justify-center rounded-full sm:size-[19px]"
          style={{ backgroundColor: eyeSocket }}
        >
          <span
            className="absolute size-2 rounded-full transition-transform duration-100 ease-out will-change-transform sm:size-[9px]"
            style={{
              backgroundColor: pupil,
              transform: `translate(${left.x}px, ${left.y}px)`,
            }}
          />
        </div>
        <div
          ref={rightSocketRef}
          className="relative flex size-[17px] items-center justify-center rounded-full sm:size-[19px]"
          style={{ backgroundColor: eyeSocket }}
        >
          <span
            className="absolute size-2 rounded-full transition-transform duration-100 ease-out will-change-transform sm:size-[9px]"
            style={{
              backgroundColor: pupil,
              transform: `translate(${right.x}px, ${right.y}px)`,
            }}
          />
        </div>
      </div>
    </div>
  )
}

export function OdysseyDashboard() {
  const { t, isDark, mounted } = useDashboardPalette()
  const { setTheme } = useTheme()
  type AppMode = "query" | "brd"
  const [appMode, setAppMode] = useState<AppMode>("query")
  const [brdMenuOpen, setBrdMenuOpen] = useState(false)
  const [brdFileName, setBrdFileName] = useState<string | null>(null)
  const brdFileRef = useRef<File | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const brdMenuRef = useRef<HTMLDivElement>(null)
  const [user, setUser] = useState<any>(null)
  const [chats, setChats] = useState<Chat[]>([])
  const [activeChatId, setActiveChatId] = useState<string | null>(null)
  const activeChatIdRef = useRef<string | null>(null)
  const [showLogout, setShowLogout] = useState(false)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [inputValue, setInputValue] = useState("")
  const [isSending, setIsSending] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const [lastQueryArtifact, setLastQueryArtifact] = useState<Record<string, unknown> | null>(null)
  const [lastQueryMeta, setLastQueryMeta] = useState<{ trust_gate?: string; confidence?: number } | null>(null)
  const [lastBrdReport, setLastBrdReport] = useState<Record<string, unknown> | null>(null)
  const [assistantLog, setAssistantLog] = useState<string[]>([])
  const [apiOk, setApiOk] = useState<boolean | null>(null)
  const supabase = createClient()
  const assistantLogRef = useRef<string[]>([])

  useEffect(() => {
    activeChatIdRef.current = activeChatId
  }, [activeChatId])

  const appendLog = useCallback((title: string, payload: unknown) => {
    const block = `${title}\n${typeof payload === "string" ? payload : JSON.stringify(payload, null, 2)}`
    const next = [...assistantLogRef.current.slice(-12), block]
    assistantLogRef.current = next
    setAssistantLog(next)
  }, [])

  useEffect(() => {
    async function loadData() {
      const { data: { user } } = await supabase.auth.getUser()
      if (user) {
        setUser(user)
        
        // Fetch chats
        const { data: chatData } = await supabase
          .from("chats")
          .select("*")
          .order("updated_at", { ascending: false })
          .order("created_at", { ascending: false })
        
        if (chatData) {
          setChats(chatData)
        }
      }
    }
    loadData()
  }, [])

  useEffect(() => {
    const base = process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "")
    if (!base) {
      setApiOk(null)
      return
    }
    fetch(`${base}/health`)
      .then((r) => setApiOk(r.ok))
      .catch(() => setApiOk(false))
  }, [])

  useEffect(() => {
    if (!brdMenuOpen) return
    const close = (e: MouseEvent) => {
      if (brdMenuRef.current && !brdMenuRef.current.contains(e.target as Node)) {
        setBrdMenuOpen(false)
      }
    }
    document.addEventListener("mousedown", close)
    return () => document.removeEventListener("mousedown", close)
  }, [brdMenuOpen])

  const openFilePicker = () => {
    fileInputRef.current?.click()
    setBrdMenuOpen(false)
  }

  const onBrdFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] ?? null
    brdFileRef.current = f
    setBrdFileName(f ? f.name : null)
    e.target.value = ""
  }

  const focusPlainText = () => {
    document.getElementById("ask-main")?.focus()
    setBrdMenuOpen(false)
  }

  const toggleTheme = () => setTheme(isDark ? "light" : "dark")

  const openChat = useCallback(
    async (chatId: string) => {
      if (!user?.id) return
      const { data: rows, error: msgErr } = await supabase
        .from("chat_messages")
        .select("id, role, content, created_at")
        .eq("chat_id", chatId)
        .order("created_at", { ascending: true })
      if (msgErr) {
        console.error(msgErr)
        return
      }
      const { data: chatRow, error: chatErr } = await supabase
        .from("chats")
        .select("mode, snapshot")
        .eq("id", chatId)
        .eq("user_id", user.id)
        .single()
      if (chatErr) {
        console.error(chatErr)
        return
      }
      setActiveChatId(chatId)
      activeChatIdRef.current = chatId
      if (chatRow?.mode === "query" || chatRow?.mode === "brd") {
        setAppMode(chatRow.mode)
      }
      setMessages(
        (rows ?? []).map((r) => ({
          id: r.id as string,
          role: r.role as "user" | "assistant",
          content: r.content as string,
        }))
      )
      const snap = (chatRow?.snapshot ?? {}) as ChatSnapshot
      setLastQueryArtifact(snap.lastQueryArtifact ?? null)
      setLastQueryMeta(snap.lastQueryMeta ?? null)
      setLastBrdReport(snap.lastBrdReport ?? null)
      const log = snap.assistantLog ?? []
      assistantLogRef.current = log
      setAssistantLog(log)
    },
    [supabase, user?.id]
  )

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, lastQueryArtifact, lastBrdReport])

  const persistRoundToSupabase = useCallback(
    async (params: {
      chatId: string
      userId: string
      userContent: string
      assistantContent: string
      snapshot: ChatSnapshot
      mode: AppMode
    }) => {
      const { chatId, userId, userContent, assistantContent, snapshot, mode } = params
      const { error: uerr } = await supabase.from("chat_messages").insert({
        chat_id: chatId,
        user_id: userId,
        role: "user",
        content: userContent,
      })
      if (uerr) console.error(uerr)
      const { error: aerr } = await supabase.from("chat_messages").insert({
        chat_id: chatId,
        user_id: userId,
        role: "assistant",
        content: assistantContent,
      })
      if (aerr) console.error(aerr)
      const { error: serr } = await supabase
        .from("chats")
        .update({
          snapshot: JSON.parse(JSON.stringify(snapshot)) as Record<string, unknown>,
          updated_at: new Date().toISOString(),
          mode,
        })
        .eq("id", chatId)
        .eq("user_id", userId)
      if (serr) console.error(serr)
    },
    [supabase]
  )

  const submitPrompt = useCallback(async () => {
    const text = inputValue.trim()
    const hasFile = !!brdFileRef.current
    if (isSending) return

    if (appMode === "query") {
      if (!text) return
    } else {
      if (!text && !hasFile) return
    }

    const userLabel =
      appMode === "brd"
        ? [text && `Note: ${text}`, hasFile && `Document: ${brdFileName || "file"}`].filter(Boolean).join("\n") ||
          "BRD upload"
        : text

    let chatId = activeChatIdRef.current
    if (user?.id && !chatId) {
      const { data: created, error: cerr } = await supabase
        .from("chats")
        .insert({
          user_id: user.id,
          title: userLabel.slice(0, 120) || "New chat",
          mode: appMode,
          updated_at: new Date().toISOString(),
        })
        .select("id,title,created_at,updated_at")
        .single()
      if (!cerr && created) {
        chatId = created.id as string
        activeChatIdRef.current = chatId
        setActiveChatId(chatId)
        setChats((prev) => [
          {
            id: created.id as string,
            title: created.title as string,
            created_at: created.created_at as string,
            updated_at: (created.updated_at as string) ?? (created.created_at as string),
          },
          ...prev.filter((c) => c.id !== created.id),
        ])
      }
    }

    const userMsg: ChatMessage = {
      id: `u-${Date.now()}`,
      role: "user",
      content: userLabel,
    }
    const pendingId = `a-${Date.now()}`
    setMessages((m) => [...m, userMsg, { id: pendingId, role: "assistant", content: "", pending: true }])
    setInputValue("")
    setIsSending(true)

    const apiBase = process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "")
    let reply =
      "Set **NEXT_PUBLIC_API_URL** in `.env.local` to your FastAPI backend (e.g. http://127.0.0.1:8000)."

    let snapArtifact: Record<string, unknown> | null = null
    let snapMeta: { trust_gate?: string; confidence?: number } | null = null
    let snapBrd: Record<string, unknown> | null = null

    try {
      if (apiBase) {
        if (appMode === "query") {
          const res = await fetch(`${apiBase}/query`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: text, mode: "query" }),
          })
          const data = await res.json().catch(() => ({}))
          appendLog("POST /query response", data)
          if (res.ok) {
            reply =
              typeof data.answer === "string" && data.answer
                ? data.answer
                : data.error === "no_chunks"
                  ? "No relevant chunks were retrieved. Check Qdrant/ES and your query."
                  : JSON.stringify(data, null, 2)
            setLastBrdReport(null)
            snapArtifact = (data.artifact as Record<string, unknown>) ?? null
            snapMeta = {
              trust_gate: data.trust_gate,
              confidence: typeof data.confidence === "number" ? data.confidence : undefined,
            }
            setLastQueryArtifact(snapArtifact)
            setLastQueryMeta(snapMeta)
          } else {
            reply = `Request failed (${res.status}): ${JSON.stringify(data)}`
            setLastQueryArtifact(null)
            setLastQueryMeta(null)
            snapArtifact = null
            snapMeta = null
          }
        } else {
          const fd = new FormData()
          if (hasFile && brdFileRef.current) {
            fd.append("file", brdFileRef.current)
          }
          if (text) {
            fd.append("user_query", text)
          }
          if (!hasFile && text) {
            fd.append("pasted_brd", text)
          }
          const res = await fetch(`${apiBase}/brd/analyze`, {
            method: "POST",
            body: fd,
          })
          const data = await res.json().catch(() => ({}))
          appendLog("POST /brd/analyze response (summary)", {
            brd_id: data.brd_id,
            trust_status: data.trust_status,
            compliance_score: data.compliance_score,
            requirements: (data.requirements as unknown[])?.length,
          })
          if (res.ok) {
            const score = data.compliance_score
            const ts = data.trust_status
            reply = `**BRD analysis complete.**\n\nCompliance score: **${score}** · Status: **${ts}**\n\nSee the BRD panel and assistant log for requirement-level detail.`
            snapBrd = data as Record<string, unknown>
            setLastBrdReport(snapBrd)
            setLastQueryArtifact(null)
            setLastQueryMeta(null)
            brdFileRef.current = null
            setBrdFileName(null)
          } else {
            reply = `BRD analysis failed (${res.status}): ${JSON.stringify(data)}`
            snapBrd = null
          }
        }
      }
    } catch (e) {
      reply = `Network error: ${e instanceof Error ? e.message : String(e)}`
      appendLog("Error", reply)
    }

    await new Promise((r) => setTimeout(r, apiBase ? 0 : 500))
    setMessages((m) => m.map((x) => (x.id === pendingId ? { ...x, content: reply, pending: false } : x)))
    setIsSending(false)

    if (user?.id && chatId) {
      const snapshot: ChatSnapshot = {
        lastQueryArtifact: snapArtifact,
        lastQueryMeta: snapMeta,
        lastBrdReport: snapBrd,
        assistantLog: [...assistantLogRef.current],
      }
      void persistRoundToSupabase({
        chatId,
        userId: user.id,
        userContent: userLabel,
        assistantContent: reply,
        snapshot,
        mode: appMode,
      })
    }
  }, [
    inputValue,
    isSending,
    appMode,
    brdFileName,
    appendLog,
    user?.id,
    supabase,
    persistRoundToSupabase,
  ])

  const onKeyDownAsk = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      void submitPrompt()
    }
  }

  const menuHover = isDark ? "hover:bg-white/10" : "hover:bg-black/[0.06]"
  const hasBrdFile = !!brdFileName
  const themePanel = {
    border: t.border,
    surface: t.surface,
    text: t.text,
    textSoft: t.textSoft,
    muted: t.muted,
    borderSubtle: t.borderSubtle,
  }

  return (
    <div
      className="relative flex min-h-svh w-full items-stretch gap-0 p-3 font-sans antialiased max-lg:flex-col lg:h-svh"
      style={{ backgroundColor: t.bg }}
    >
      <input
        ref={fileInputRef}
        type="file"
        className="sr-only"
        accept={BRD_ACCEPT}
        onChange={onBrdFile}
        aria-hidden
      />

      <aside
        className={cn(
          SIDEBAR_W,
          "relative z-10 flex min-h-0 max-h-none shrink-0 flex-col overflow-hidden rounded-3xl border p-5 max-lg:max-w-none max-lg:rounded-2xl lg:max-h-[calc(100vh-24px)] lg:self-center"
        )}
        style={{
          borderColor: t.border,
          backgroundColor: t.bg,
          boxShadow: t.shadow,
        }}
      >
        <div className="mb-4 shrink-0">
          {messages.length === 0 ? (
            <div className="mb-3 flex items-center gap-2">
              <Shield className="size-6 shrink-0 text-emerald-600 dark:text-emerald-400" strokeWidth={1.75} />
              <span className="text-[15px] font-bold tracking-tight" style={{ color: t.text }}>
                XAI COMPLIANCE
              </span>
            </div>
          ) : (
            <div className="mb-2" aria-hidden />
          )}
          <nav className="flex flex-col gap-1.5 text-[14px]">
            <button
              type="button"
              onClick={() => setAppMode("query")}
              className={cn(
                "flex w-full items-center gap-3 rounded-2xl px-3 py-3 text-left font-medium transition-colors",
                appMode === "query" ? "" : "opacity-90"
              )}
              style={{
                backgroundColor: appMode === "query" ? t.activeNav : "transparent",
                color: t.text,
              }}
            >
              <Bot className="size-5 shrink-0 text-blue-500 dark:text-blue-400" strokeWidth={1.75} />
              Query Bot
            </button>
            <button
              type="button"
              onClick={() => setAppMode("brd")}
              className={cn(
                "flex w-full items-center gap-3 rounded-2xl px-3 py-3 text-left font-medium transition-colors",
                appMode === "brd" ? "" : "opacity-90"
              )}
              style={{
                backgroundColor: appMode === "brd" ? (isDark ? "#14532d" : "#dcfce7") : "transparent",
                color: t.text,
              }}
            >
              <Upload className="size-5 shrink-0 text-emerald-600 dark:text-emerald-400" strokeWidth={1.75} />
              BRD Upload
            </button>
          </nav>
          <div
            className="mt-4 flex items-center gap-2 rounded-xl border px-2.5 py-2 text-[11px]"
            style={{ borderColor: t.border, color: t.muted }}
          >
            <Activity className="size-3.5 shrink-0" />
            <span className="font-medium">API</span>
            <span
              className={cn(
                "ml-auto rounded-md px-1.5 py-0.5 text-[10px] font-bold uppercase",
                apiOk === true && "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400",
                apiOk === false && "bg-red-500/15 text-red-600 dark:text-red-400",
                apiOk === null && "bg-zinc-500/15"
              )}
            >
              {apiOk === true ? "Up" : apiOk === false ? "Down" : "—"}
            </span>
          </div>
        </div>

        <div
          className="mt-4 shrink-0 rounded-2xl border p-5"
          style={{ borderColor: t.border, backgroundColor: t.surface, color: t.text }}
        >
          <div className="mb-4 flex items-start justify-between gap-2">
            <div>
              <p className="text-[17px] font-semibold tracking-tight">Premium Plan</p>
              <p className="mt-2 text-[13px] leading-relaxed" style={{ color: t.muted }}>
                Want to reach more features and grow much bigger?
              </p>
            </div>
            <div
              className="flex shrink-0 items-center gap-1 rounded-full border px-2.5 py-1 text-[12px] font-semibold"
              style={{ borderColor: t.border }}
            >
              <Zap className="size-4" fill="currentColor" />
              10/10
            </div>
          </div>
          <button
            type="button"
            className="flex w-full items-center justify-between rounded-xl border px-3 py-2.5 text-[14px] font-semibold transition-opacity hover:opacity-90"
            style={{ borderColor: t.border, color: t.text }}
          >
            Upgrade to Premium
            <ArrowRight className="size-4" />
          </button>
        </div>

        <div className="mt-auto space-y-0.5 pt-8 text-[15px]">
          {(["Feedback", "Invite People", "Settings"] as const).map((label, i) => (
            <a
              key={label}
              href="#"
              className="flex items-center gap-3 rounded-xl px-2 py-2.5 transition-colors"
              style={{ color: t.muted }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = t.navHover
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = "transparent"
              }}
            >
              {i === 0 ? (
                <MessageSquare className="size-[18px]" strokeWidth={1.75} />
              ) : i === 1 ? (
                <Share2 className="size-[18px]" strokeWidth={1.75} />
              ) : (
                <Settings className="size-[18px]" strokeWidth={1.75} />
              )}
              <span style={{ color: t.textSoft }}>{label}</span>
            </a>
          ))}
        </div>

        <div
          className="mt-5 flex items-center justify-between gap-3 rounded-2xl border px-3 py-2.5"
          style={{ borderColor: t.border, backgroundColor: t.surface }}
        >
          <div className="min-w-0">
            <p className="text-[14px] font-medium" style={{ color: t.text }}>
              Appearance
            </p>
            <p className="text-[12px]" style={{ color: t.muted }}>
              {mounted ? (isDark ? "Dark" : "Light") : "…"}
            </p>
          </div>
          <ThemeSwitch checked={mounted ? isDark : true} onChange={toggleTheme} />
        </div>

        <div className="relative mt-4">
          {showLogout && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="absolute bottom-full left-0 mb-2 w-full"
            >
              <button
                onClick={() => logout()}
                className="flex w-full items-center justify-center gap-2 rounded-xl bg-red-500/10 py-2.5 text-sm font-medium text-red-500 border border-red-500/20 hover:bg-red-500/20 transition-colors"
                title="Logout"
              >
                <LogOut className="size-4" />
                Sign Out
              </button>
            </motion.div>
          )}
          <div
            className="flex w-full items-center gap-3 rounded-2xl border px-3 py-3 text-left transition"
            style={{ borderColor: t.border, backgroundColor: t.surface }}
          >
            <div
              className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full text-[12px] font-bold tracking-tight text-primary-foreground bg-primary"
              aria-hidden
            >
              {user?.email?.substring(0, 2).toUpperCase() || "UC"}
            </div>
            <div className="min-w-0 flex-1">
              <p className="truncate text-[15px] font-semibold" style={{ color: t.text }}>
                {user?.user_metadata?.full_name || user?.email?.split("@")[0] || "User"}
              </p>
              <p className="truncate text-[13px]" style={{ color: t.muted }}>
                {user?.email || "loading..."}
              </p>
            </div>
            <button
              type="button"
              onClick={() => setShowLogout(!showLogout)}
              className={cn(
                "flex size-8 items-center justify-center rounded-lg transition-colors hover:bg-white/10",
                showLogout && "bg-white/10"
              )}
            >
              <ChevronRight className={cn("size-[18px] shrink-0 transition-transform", showLogout && "rotate-90")} style={{ color: t.muted }} strokeWidth={2} />
            </button>
          </div>
        </div>
      </aside>

      <main
        className={cn(
          "relative z-0 flex min-h-0 min-w-0 flex-1 flex-col max-lg:min-h-[50vh] max-lg:border-t",
          isDark ? "max-lg:border-zinc-800" : "max-lg:border-zinc-200"
        )}
        style={{ backgroundColor: t.bg }}
      >
        <div className="flex min-h-0 flex-1 flex-col">
          <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4 sm:px-6 lg:px-10 xl:px-12">
            <div className={cn("mx-auto w-full space-y-6", CENTER_MAX)}>
              {messages.length > 0 ? (
                <div className="flex flex-col gap-5 pb-2">
                  {messages.map((m) => (
                    <motion.div
                      key={m.id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.25, ease: "easeOut" }}
                      className={cn("flex w-full", m.role === "user" ? "justify-end" : "justify-start")}
                    >
                      {m.role === "assistant" ? (
                        <div
                          className="max-w-[min(100%,52rem)] rounded-2xl rounded-tl-md border px-4 py-3 shadow-sm sm:px-5 sm:py-4"
                          style={{
                            borderColor: t.borderSubtle,
                            background: isDark
                              ? "linear-gradient(145deg, rgba(39,39,42,0.95) 0%, rgba(24,24,27,0.98) 100%)"
                              : "linear-gradient(145deg, #ffffff 0%, #f4f4f5 100%)",
                            boxShadow: isDark ? "0 4px 24px rgba(0,0,0,0.35)" : "0 4px 20px rgba(0,0,0,0.06)",
                          }}
                        >
                          <div className="mb-2 flex items-center gap-2">
                            <span
                              className="flex size-7 items-center justify-center rounded-lg"
                              style={{
                                backgroundColor: isDark ? "rgba(59,130,246,0.2)" : "rgba(59,130,246,0.12)",
                              }}
                            >
                              <Sparkles className="size-3.5 text-blue-500 dark:text-blue-400" strokeWidth={2} />
                            </span>
                            <span
                              className="text-[12px] font-semibold uppercase tracking-wide"
                              style={{ color: t.muted }}
                            >
                              Assistant
                            </span>
                          </div>
                          {m.pending ? (
                            <div className="flex items-center gap-2 py-1" style={{ color: t.muted }}>
                              <span className="inline-flex gap-1">
                                <span className="size-1.5 animate-bounce rounded-full bg-current [animation-delay:-0.3s]" />
                                <span className="size-1.5 animate-bounce rounded-full bg-current [animation-delay:-0.15s]" />
                                <span className="size-1.5 animate-bounce rounded-full bg-current" />
                              </span>
                              <span className="text-[14px]">Thinking…</span>
                            </div>
                          ) : (
                            <p
                              className="whitespace-pre-wrap text-[15px] leading-relaxed sm:text-[16px]"
                              style={{ color: t.textSoft }}
                            >
                              {m.content}
                            </p>
                          )}
                        </div>
                      ) : (
                        <div
                          className="max-w-[min(100%,48rem)] rounded-2xl rounded-tr-md px-4 py-3 sm:px-5 sm:py-3.5"
                          style={{
                            backgroundColor: isDark ? "rgba(59,130,246,0.22)" : "rgba(59,130,246,0.12)",
                            color: t.text,
                          }}
                        >
                          <p className="whitespace-pre-wrap text-[15px] leading-relaxed sm:text-[16px]">{m.content}</p>
                        </div>
                      )}
                    </motion.div>
                  ))}
                  <div ref={messagesEndRef} />
                </div>
              ) : null}

              {appMode === "query" && lastQueryArtifact ? (
                <QueryRagasPanel
                  artifact={lastQueryArtifact}
                  trustGate={lastQueryMeta?.trust_gate}
                  confidence={lastQueryMeta?.confidence}
                  t={themePanel}
                  isDark={isDark}
                />
              ) : null}
              {appMode === "brd" && lastBrdReport ? (
                <BrdAnalysisPanel data={lastBrdReport} t={themePanel} isDark={isDark} />
              ) : null}

              {messages.length === 0 ? (
                <div className="flex min-h-[42vh] flex-col items-center justify-center gap-3 py-6">
                  <TrackingEyes surface={t.surface} eyeSocket={t.eyeSocket} pupil={t.pupil} />
                  <h1
                    className="text-center text-[30px] font-semibold tracking-tight sm:text-[38px] lg:text-[42px]"
                    style={{ color: t.text }}
                  >
                    XAI COMPLIANCE
                  </h1>
                  <p className="max-w-md text-center text-sm leading-relaxed sm:text-[15px]" style={{ color: t.muted }}>
                    {appMode === "query"
                      ? "Query Bot: RBI-grounded answers via hybrid RAG, NLI, and trust scoring."
                      : "BRD Upload: attach a document (or paste text) and optional instructions. Runs the full BRD pipeline on the backend."}
                  </p>
                </div>
              ) : null}
            </div>
          </div>

          <div className="shrink-0 px-4 pb-6 pt-2 sm:px-6 lg:px-10 xl:px-12">
            <div className={cn("mx-auto w-full", CENTER_MAX)}>
              <div
                className="rounded-2xl border px-3 py-3 sm:rounded-[22px] sm:px-4 sm:py-3.5"
                style={{
                  backgroundColor: t.surface,
                  borderColor: t.borderSubtle,
                  boxShadow: messages.length > 0 ? t.shadow : undefined,
                }}
              >
                <label htmlFor="ask-main" className="sr-only">
                  Message or BRD as plain text
                </label>
                <div className="flex items-end gap-2 sm:gap-3">
                  <div className="min-w-0 flex-1">
                    <textarea
                      id="ask-main"
                      value={inputValue}
                      onChange={(e) => setInputValue(e.target.value)}
                      onKeyDown={onKeyDownAsk}
                      placeholder={
                        appMode === "query"
                          ? "Ask a regulatory question… (Enter to send, Shift+Enter for newline)"
                          : "Optional instructions, or paste BRD plain text if not using a file…"
                      }
                      rows={messages.length > 0 ? 2 : 3}
                      disabled={isSending}
                      className={cn(
                        "min-h-[52px] w-full resize-none bg-transparent text-[16px] leading-normal focus:outline-none sm:min-h-[64px] sm:text-[17px]",
                        "placeholder:text-zinc-400 dark:placeholder:text-[#7a7a7a]",
                        isSending && "opacity-60"
                      )}
                      style={{ color: t.text }}
                    />
                    {brdFileName ? (
                      <p className="mt-1 truncate text-[13px] sm:text-sm" style={{ color: t.muted }}>
                        BRD file: {brdFileName}
                      </p>
                    ) : null}
                  </div>
                  <button
                    type="button"
                    disabled={
                      isSending ||
                      (appMode === "query" ? !inputValue.trim() : !inputValue.trim() && !hasBrdFile)
                    }
                    onClick={() => void submitPrompt()}
                    className="mb-0.5 flex size-10 shrink-0 items-center justify-center rounded-full transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
                    style={{ backgroundColor: ACCENT_ORANGE }}
                    aria-label="Submit"
                  >
                    <ArrowUp className="size-5 text-neutral-900" strokeWidth={2.5} />
                  </button>
                </div>
                {appMode === "brd" ? (
                  <div className="relative mt-2 flex items-center gap-2" ref={brdMenuRef}>
                    <button
                      type="button"
                      onClick={() => setBrdMenuOpen((o) => !o)}
                      className={cn(
                        "flex size-9 shrink-0 items-center justify-center rounded-full transition-colors",
                        isDark ? "hover:bg-white/10" : "hover:bg-black/10"
                      )}
                      style={{ backgroundColor: t.chipBg, color: t.text }}
                      aria-expanded={brdMenuOpen}
                      aria-haspopup="menu"
                      aria-label="Attach BRD file"
                    >
                      <Plus className="size-[17px]" strokeWidth={2} />
                    </button>
                    {brdMenuOpen ? (
                      <div
                        role="menu"
                        className={cn(
                          "absolute bottom-full left-0 z-30 mb-2 min-w-[252px] rounded-xl border py-1 shadow-xl sm:min-w-[268px]",
                          menuHover
                        )}
                        style={{
                          borderColor: t.border,
                          backgroundColor: t.menuBg,
                          boxShadow: t.shadow,
                        }}
                      >
                        <button
                          type="button"
                          role="menuitem"
                          className={cn(
                            "flex w-full items-center gap-2.5 px-3 py-3 text-left text-[14px] sm:px-3.5 sm:text-[15px]",
                            menuHover
                          )}
                          style={{ color: t.text }}
                          onClick={openFilePicker}
                        >
                          <FileText className="size-4 shrink-0 sm:size-[18px]" style={{ color: t.muted }} />
                          Upload BRD — PDF, Word (.doc/.docx), or .txt
                        </button>
                        <button
                          type="button"
                          role="menuitem"
                          className={cn(
                            "flex w-full items-center gap-2.5 px-3 py-3 text-left text-[14px] sm:px-3.5 sm:text-[15px]",
                            menuHover
                          )}
                          style={{ color: t.text }}
                          onClick={focusPlainText}
                        >
                          <FileText className="size-4 shrink-0 sm:size-[18px]" style={{ color: t.muted }} />
                          Focus text field (paste BRD)
                        </button>
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </div>
            </div>
          </div>
        </div>
      </main>

      <aside
        className={cn(
          RIGHT_W,
          "relative z-10 flex min-h-0 max-h-none shrink-0 flex-col overflow-hidden rounded-3xl border p-5 max-lg:max-w-none max-lg:rounded-2xl lg:max-h-[calc(100vh-24px)] lg:self-center"
        )}
        style={{
          borderColor: t.border,
          backgroundColor: t.bg,
          boxShadow: t.shadow,
        }}
      >
        <div className="flex min-h-0 flex-1 flex-col">
          <ComplianceAssistantLog
            lines={assistantLog}
            onClear={() => setAssistantLog([])}
            t={themePanel}
          />
        </div>
        <div
          className="mt-4 flex min-h-0 max-h-[min(40vh,280px)] flex-col border-t pt-4"
          style={{ borderColor: t.borderSubtle }}
        >
          <div className="mb-2 flex shrink-0 items-center justify-between gap-2">
            <p className="text-[12px] font-semibold uppercase tracking-wider" style={{ color: t.muted }}>
              Chat history
            </p>
            <button
              type="button"
              onClick={() => {
                setActiveChatId(null)
                activeChatIdRef.current = null
                setMessages([])
                setInputValue("")
                setLastQueryArtifact(null)
                setLastQueryMeta(null)
                setLastBrdReport(null)
                assistantLogRef.current = []
                setAssistantLog([])
                brdFileRef.current = null
                setBrdFileName(null)
              }}
              className="rounded-full px-2.5 py-1 text-[11px] font-medium transition-opacity hover:opacity-90"
              style={{ backgroundColor: t.surface, color: t.text }}
            >
              New chat
            </button>
          </div>
          <div className="min-h-0 flex-1 space-y-1.5 overflow-y-auto pr-0.5">
            {chats.length > 0 ? (
              chats.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => void openChat(item.id)}
                  className={cn(
                    "flex w-full flex-col gap-0.5 rounded-xl border px-2.5 py-2.5 text-left transition-colors",
                    item.id === activeChatId
                      ? isDark
                        ? "border-blue-500/40"
                        : "border-blue-400/50"
                      : cn(
                          "border-transparent",
                          isDark ? "hover:border-zinc-700 hover:bg-white/5" : "hover:border-zinc-300 hover:bg-black/4"
                        )
                  )}
                  style={{
                    backgroundColor:
                      item.id === activeChatId
                        ? isDark
                          ? "rgba(255,255,255,0.06)"
                          : "rgba(0,0,0,0.04)"
                        : t.surface,
                  }}
                >
                  <span className="line-clamp-2 text-[13px] font-medium leading-snug" style={{ color: t.text }}>
                    {item.title}
                  </span>
                  <span className="text-[11px]" style={{ color: t.muted }}>
                    {new Date(item.created_at).toLocaleDateString()}
                  </span>
                </button>
              ))
            ) : (
              <p className="py-3 text-center text-[11px] leading-relaxed" style={{ color: t.muted }}>
                No saved threads yet. Supabase <span className="font-mono">chats</span> appear here.
              </p>
            )}
          </div>
        </div>
      </aside>
    </div>
  )
}
