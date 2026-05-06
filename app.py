"""Streamlit UI — month-end close tool.

Flow (tabs mirror the workflow):
  1. Upload       — TB + GL files, pick Excel sheet, preview
  2. Map columns  — auto-suggest + manual override, validate
  3. Run          — pick period, kick off LangGraph pipeline
  4. Review       — snapshot, anomalies, forecasts, narrative, policy hits
  5. Approve      — human sign-off, seal snapshot, download reports
  6. History      — prior runs and audit log
"""
from __future__ import annotations
import io
import time
from typing import Dict, Optional

import pandas as pd
import streamlit as st

from config import (
    TB_CANONICAL_COLS, TB_REQUIRED_COLS,
    GL_CANONICAL_COLS, GL_REQUIRED_COLS,
    LLM_PROVIDER, LLM_MODEL, POLICY_DIR,
)
from ingestion import (
    read_file, list_excel_sheets, suggest_mapping,
    apply_mapping, validate_tb, validate_gl,
)
from storage.db import (
    init_db, load_snapshot, list_runs, get_audit,
)
from storage.vector_store import ingest_policies, collection_size
from orchestrator import stage_datasets, run_pipeline, approve_and_seal, new_run_id
from reports import generate_pdf, generate_xlsx


# ---------- Setup ----------
st.set_page_config(page_title="Month-End Close Agent", layout="wide",
                   initial_sidebar_state="expanded")
init_db()

# Session-state scaffolding
def _ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

_ss("run_id", None)
_ss("tb_raw", None)       # raw DataFrame after read
_ss("gl_raw", None)
_ss("tb_mapped", None)    # validated DataFrame after mapping
_ss("gl_mapped", None)
_ss("tb_mapping", {})     # canonical -> user_col
_ss("gl_mapping", {})
_ss("tb_suggestions", [])
_ss("gl_suggestions", [])
_ss("final_state", None)  # LangGraph output
_ss("approved", False)
_ss("pdf_path", None)
_ss("xlsx_path", None)


# ---------- Sidebar ----------
with st.sidebar:
    st.title("⚙️ Close Agent")
    st.caption("Agentic month-end close workflow")

    st.subheader("LLM provider")
    st.code(f"{LLM_PROVIDER} — {LLM_MODEL}", language="text")

    st.subheader("Policy library (RAG)")
    n_chunks = collection_size()
    st.metric("Indexed chunks", n_chunks)
    if st.button("📚 (Re)index policy docs", use_container_width=True):
        with st.spinner("Ingesting policies..."):
            added = ingest_policies()
        if added:
            st.success(f"Added: {added}")
        else:
            st.info("No new chunks (all up to date).")

    uploaded_policy = st.file_uploader(
        "Add extra policy .txt/.md", type=["txt", "md"],
        accept_multiple_files=True, key="policy_uploader",
    )
    if uploaded_policy:
        POLICY_DIR.mkdir(parents=True, exist_ok=True)
        for f in uploaded_policy:
            (POLICY_DIR / f.name).write_bytes(f.read())
        st.success(f"Saved {len(uploaded_policy)} file(s). Click '(Re)index' above.")

    st.divider()
    if st.session_state.run_id:
        st.caption(f"**Current run:** `{st.session_state.run_id}`")
    if st.button("🔄 New run", use_container_width=True):
        for k in ("run_id", "tb_raw", "gl_raw", "tb_mapped", "gl_mapped",
                  "tb_mapping", "gl_mapping", "tb_suggestions", "gl_suggestions",
                  "final_state", "approved", "pdf_path", "xlsx_path"):
            st.session_state[k] = None if "raw" in k or "mapped" in k or "path" in k or "state" in k or "run_id" == k else ({} if "mapping" in k else ([] if "suggestions" in k else False))
        st.rerun()


# ---------- Title ----------
st.title("📊 Month-End Close — Agentic Workflow")
st.caption("Upload TB + GL → map columns → run agents → review → approve → download")

tab_upload, tab_map, tab_run, tab_review, tab_approve, tab_history = st.tabs(
    ["1. Upload", "2. Map columns", "3. Run", "4. Review", "5. Approve & download", "📁 History"]
)


# ============================================================
# 1. UPLOAD
# ============================================================
with tab_upload:
    st.header("Step 1 — Upload Trial Balance & General Ledger")
    st.markdown(
        "Accepted formats: **CSV, TSV, XLSX, XLS, XLSM**. "
        "For Excel files you'll pick the sheet below."
    )

    col1, col2 = st.columns(2)

    def _file_block(label: str, state_key: str, kind: str, sheet_key: str):
        st.subheader(label)
        up = st.file_uploader(
            f"Choose {label} file", type=["csv", "tsv", "xlsx", "xls", "xlsm"],
            key=f"upl_{kind}",
        )
        if up is None:
            return
        fname = up.name
        fbytes = up.read()
        sheet_name = None
        header_row = 0

        if fname.lower().endswith((".xlsx", ".xls", ".xlsm")):
            sheets = list_excel_sheets(fbytes)
            sheet_name = st.selectbox(
                f"{label} — sheet", sheets, key=f"sheet_{kind}",
            )
            header_row = st.number_input(
                f"{label} — header row (0-indexed)", 0, 20, 0, key=f"hdr_{kind}",
            )

        try:
            df = read_file(fbytes, fname, sheet_name=sheet_name, header_row=header_row)
        except Exception as e:
            st.error(f"Read failed: {e}")
            return

        st.caption(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
        st.dataframe(df.head(10), use_container_width=True, height=240)
        st.session_state[state_key] = df

    with col1:
        _file_block("Trial Balance (TB)", "tb_raw", "tb", "sheet_tb")

    with col2:
        _file_block("General Ledger (GL)", "gl_raw", "gl", "sheet_gl")

    st.divider()
    tb_ok = st.session_state.tb_raw is not None
    gl_ok = st.session_state.gl_raw is not None
    if tb_ok and gl_ok:
        st.success("✅ Both files loaded. Go to **2. Map columns**.")
    else:
        missing = [n for n, ok in [("TB", tb_ok), ("GL", gl_ok)] if not ok]
        st.info(f"Waiting for: {', '.join(missing)}")


# ============================================================
# 2. MAP COLUMNS
# ============================================================
with tab_map:
    st.header("Step 2 — Map your columns to the canonical schema")
    st.markdown(
        "The tool tries to auto-detect each field. **Green** = high confidence, "
        "**yellow** = medium, **red** = unassigned. Required fields are marked ⭐."
    )

    def _confidence_emoji(c: float, assigned: bool) -> str:
        if not assigned: return "🔴"
        if c >= 0.85: return "🟢"
        if c >= 0.65: return "🟡"
        return "🟠"

    def _mapping_grid(
        label: str, df: pd.DataFrame, canonical: list[str], required: list[str],
        mapping_key: str, sugg_key: str,
    ) -> Dict[str, Optional[str]]:
        st.subheader(label)
        user_cols = list(df.columns.astype(str))

        # Run suggestions once per upload
        if not st.session_state[sugg_key]:
            st.session_state[sugg_key] = suggest_mapping(user_cols, canonical)

        mapping: Dict[str, Optional[str]] = {}
        opts = ["— none —"] + user_cols

        for s in st.session_state[sugg_key]:
            star = "⭐ " if s.canonical in required else ""
            default_idx = (opts.index(s.user_column)
                           if s.user_column and s.user_column in opts else 0)
            c1, c2, c3 = st.columns([2, 3, 1])
            with c1:
                st.markdown(f"**{star}{s.canonical}**")
            with c2:
                picked = st.selectbox(
                    f"→ user column for `{s.canonical}`",
                    opts, index=default_idx,
                    key=f"{mapping_key}_{s.canonical}",
                    label_visibility="collapsed",
                )
            with c3:
                st.markdown(
                    f"{_confidence_emoji(s.confidence, picked != '— none —')} "
                    f"`{s.confidence:.2f}`" if picked != "— none —" else "🔴 unset"
                )
            mapping[s.canonical] = None if picked == "— none —" else picked
        return mapping

    if st.session_state.tb_raw is None or st.session_state.gl_raw is None:
        st.warning("Upload both TB and GL first (tab 1).")
    else:
        tb_map = _mapping_grid(
            "Trial Balance mapping", st.session_state.tb_raw,
            TB_CANONICAL_COLS, TB_REQUIRED_COLS,
            "tb_map", "tb_suggestions",
        )
        st.divider()
        gl_map = _mapping_grid(
            "General Ledger mapping", st.session_state.gl_raw,
            GL_CANONICAL_COLS, GL_REQUIRED_COLS,
            "gl_map", "gl_suggestions",
        )

        st.divider()
        if st.button("✅ Apply mapping & validate", type="primary", use_container_width=True):
            try:
                tb_renamed = apply_mapping(
                    st.session_state.tb_raw, tb_map,
                    TB_CANONICAL_COLS, TB_REQUIRED_COLS,
                )
                tb_res = validate_tb(tb_renamed)
                gl_renamed = apply_mapping(
                    st.session_state.gl_raw, gl_map,
                    GL_CANONICAL_COLS, GL_REQUIRED_COLS,
                )
                gl_res = validate_gl(gl_renamed)
            except ValueError as e:
                st.error(f"Mapping incomplete: {e}")
                st.stop()

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**TB validation**")
                st.json(tb_res.stats)
                for w in tb_res.warnings: st.warning(w)
                for e in tb_res.errors: st.error(e)
            with c2:
                st.markdown("**GL validation**")
                st.json(gl_res.stats)
                for w in gl_res.warnings: st.warning(w)
                for e in gl_res.errors: st.error(e)

            if tb_res.ok and gl_res.ok:
                st.session_state.tb_mapped = tb_res.df
                st.session_state.gl_mapped = gl_res.df
                st.session_state.tb_mapping = tb_map
                st.session_state.gl_mapping = gl_map
                st.success("✅ Validation passed. Go to **3. Run**.")
            else:
                st.error("Validation failed. Fix the mapping and retry.")


# ============================================================
# 3. RUN
# ============================================================
with tab_run:
    st.header("Step 3 — Run the close pipeline")
    if st.session_state.tb_mapped is None or st.session_state.gl_mapped is None:
        st.warning("Complete mapping & validation first (tab 2).")
    else:
        tb = st.session_state.tb_mapped
        periods_tb = sorted(tb["period"].dropna().unique())
        periods_gl = sorted(st.session_state.gl_mapped["period"].dropna().unique())
        overlap = sorted(set(periods_tb) & set(periods_gl))

        st.markdown(f"**Periods in TB:** {', '.join(periods_tb) or '(none)'}")
        st.markdown(f"**Periods in GL:** {', '.join(periods_gl) or '(none)'}")
        if not overlap:
            st.error("No overlapping period between TB and GL. Cannot run.")
            st.stop()

        period = st.selectbox("Close period", overlap, index=len(overlap)-1)

        st.info(
            "The pipeline runs: Accounting → ML (anomaly + forecast) → RAG → "
            "Narrative (LLM) → Guardrails → Finalise. This typically takes "
            "15–90 seconds depending on LLM speed."
        )

        if st.button("🚀 Run pipeline", type="primary", use_container_width=True):
            run_id = new_run_id()
            st.session_state.run_id = run_id
            st.session_state.approved = False
            st.session_state.pdf_path = None
            st.session_state.xlsx_path = None

            progress = st.progress(0, "Starting...")
            log = st.empty()

            t0 = time.time()
            try:
                progress.progress(10, "Staging datasets into SQLite...")
                stage_datasets(run_id, period,
                               st.session_state.tb_mapped,
                               st.session_state.gl_mapped)

                progress.progress(25, "Running LangGraph pipeline...")
                with st.spinner("Agents working — accounting → ML → RAG → narrative → guardrails"):
                    final_state = run_pipeline(run_id, period)

                progress.progress(100, "Done.")
                st.session_state.final_state = final_state
                elapsed = time.time() - t0

                status = final_state.get("status", "unknown")
                if status == "awaiting_approval":
                    st.success(f"✅ Pipeline completed in {elapsed:.1f}s — awaiting approval. "
                               f"Go to **4. Review**.")
                elif status == "escalated":
                    st.error(f"⚠️  Pipeline escalated (guardrails). Review in tab 4 — "
                             f"a human must intervene before approval.")
                else:
                    st.warning(f"Pipeline ended with status: {status}")

            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.exception(e)


# ============================================================
# 4. REVIEW
# ============================================================
with tab_review:
    st.header("Step 4 — Review results")
    rid = st.session_state.run_id
    snap = load_snapshot(rid) if rid else None

    if not snap:
        st.info("No results yet. Run the pipeline first (tab 3).")
    else:
        # --- Top KPIs
        s = snap["snapshot"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Period", s["period"])
        c2.metric("TB debit", f"{s['tb_totals']['debit']:,.0f}")
        c3.metric("TB credit", f"{s['tb_totals']['credit']:,.0f}")
        c4.metric("TB diff", f"{s['tb_totals']['diff']:,.2f}",
                  delta_color=("normal" if abs(s['tb_totals']['diff']) < 0.01 else "inverse"))

        rec = s["reconciliation"]
        if rec["matched"]:
            st.success("✅ TB ↔ GL reconciled.")
        else:
            st.warning(f"⚠️  TB–GL debit diff: {rec['tb_minus_gl_debit']:,.2f}, "
                       f"credit diff: {rec['tb_minus_gl_credit']:,.2f}")

        gr = snap.get("guardrail", {})
        if gr:
            status = gr.get("status", "pass")
            badge = {"pass": "🟢 pass", "retry": "🟡 retry",
                     "escalate": "🔴 escalate"}.get(status, status)
            st.markdown(f"**Guardrails:** {badge}")
            if gr.get("violations"):
                with st.expander("Violations detail"):
                    st.json(gr["violations"])

        st.divider()

        # --- Tabs for each agent output
        r1, r2, r3, r4, r5 = st.tabs(
            ["📖 Narrative", "🚨 Anomalies", "📈 Forecasts", "📚 Policies", "🔍 Raw snapshot"]
        )

        with r1:
            st.markdown(snap.get("narrative", "(no narrative)"))

        with r2:
            anoms = snap.get("anomalies", [])
            if anoms:
                st.dataframe(pd.DataFrame(anoms), use_container_width=True)
            else:
                st.info("No anomalies flagged.")

        with r3:
            fc = snap.get("forecasts", {})
            if "_note" in fc:
                st.info(fc["_note"])
            elif fc:
                rows = [{"entity_or_total": k, **(v if isinstance(v, dict) else {})}
                        for k, v in fc.items()]
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.info("No forecast produced.")

        with r4:
            st.caption(f"RAG query: `{snap.get('rag_query','')}` | "
                       f"avg confidence: {snap.get('rag_confidence', 0):.2f}")
            for h in snap.get("policy_hits", []):
                with st.expander(f"{h['source']} (chunk {h['chunk']}, "
                                 f"confidence {h['confidence']})"):
                    st.write(h["text"])

        with r5:
            with st.expander("Top accounts"):
                st.dataframe(pd.DataFrame(s.get("top_accounts", [])),
                             use_container_width=True)
            with st.expander("Accrual candidates"):
                st.dataframe(pd.DataFrame(s.get("accrual_candidates", [])),
                             use_container_width=True)
            with st.expander("Entity rollup"):
                st.dataframe(pd.DataFrame(s.get("entity_rollup", [])),
                             use_container_width=True)


# ============================================================
# 5. APPROVE & DOWNLOAD
# ============================================================
with tab_approve:
    st.header("Step 5 — Approve & download close package")
    rid = st.session_state.run_id
    snap = load_snapshot(rid) if rid else None
    final = st.session_state.final_state

    if not snap:
        st.info("Nothing to approve yet.")
    elif snap.get("_sealed"):
        st.success(f"✅ Run `{rid}` is sealed.")
    elif final and final.get("status") == "escalated":
        st.error(
            "Pipeline was escalated by guardrails. Approval is blocked until a "
            "human reviews the violations and reruns the narrative manually."
        )
    else:
        approver = st.text_input("Approver ID (your name or email)")
        comment = st.text_area("Approval comment (optional)", height=80)

        confirm = st.checkbox(
            "I confirm I have reviewed the close package and authorise sealing."
        )
        if st.button("🖋  Approve & seal", type="primary",
                     disabled=not (approver and confirm),
                     use_container_width=True):
            approve_and_seal(rid, approver)
            st.session_state.approved = True
            st.success(f"✅ Approved by {approver} — snapshot sealed.")
            st.rerun()

    # Downloads (available once snapshot exists — approved or not)
    if snap:
        st.divider()
        st.subheader("📥 Download close package")
        c1, c2 = st.columns(2)

        with c1:
            if st.button("Generate PDF", use_container_width=True):
                with st.spinner("Generating PDF..."):
                    p = generate_pdf(rid, snap)
                    st.session_state.pdf_path = str(p)

            if st.session_state.pdf_path:
                with open(st.session_state.pdf_path, "rb") as fh:
                    st.download_button(
                        "📄 Download PDF", fh.read(),
                        file_name=f"close_{rid}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )

        with c2:
            if st.button("Generate Excel workbook", use_container_width=True):
                with st.spinner("Generating Excel..."):
                    # Re-fetch the TB/GL we staged (handles re-load after restart)
                    from storage.db import load_tb, load_gl
                    tb_df = load_tb(rid).drop(columns=["run_id"])
                    gl_df = load_gl(rid).drop(columns=["run_id", "row_id"])
                    p = generate_xlsx(rid, snap, tb_df, gl_df)
                    st.session_state.xlsx_path = str(p)

            if st.session_state.xlsx_path:
                with open(st.session_state.xlsx_path, "rb") as fh:
                    st.download_button(
                        "📊 Download Excel", fh.read(),
                        file_name=f"close_{rid}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )


# ============================================================
# 6. HISTORY
# ============================================================
with tab_history:
    st.header("📁 Prior runs")
    runs = list_runs()
    if runs.empty:
        st.info("No prior runs.")
    else:
        runs["started_at"] = pd.to_datetime(runs["started_at"], unit="s")
        runs["completed_at"] = pd.to_datetime(runs["completed_at"], unit="s")
        st.dataframe(runs, use_container_width=True)

        pick = st.selectbox("Inspect run", [""] + runs["run_id"].tolist())
        if pick:
            st.subheader(f"Audit log — {pick}")
            audit = get_audit(pick)
            if not audit.empty:
                audit["ts"] = pd.to_datetime(audit["ts"], unit="s")
                st.dataframe(audit, use_container_width=True, height=400)
            snap = load_snapshot(pick)
            if snap:
                with st.expander("Snapshot JSON"):
                    st.json(snap)
