import { FormEvent, useEffect, useRef, useState } from "react";
import { api, DatasetStatus, QueryAsset } from "./api";
import { clock, isSingleFrameCue, PlaybackObservation, rag, running, VideoCue, VideoSource, VideoTurn } from "./videoRagApi";
import DatasetView from "./views/DatasetView";
import ReportView from "./views/ReportView";
import { Icon } from "./views/RagIcons";
import VideoRagPlayer, { VideoJump } from "./views/VideoRagPlayer";
import { FrameTargetPicker, ImageTargetPicker, RagDialog } from "./views/VideoTargetPicker";
import "./workbench.css";

const HISTORY_KEY = "video-rag-last-turn";
const STATUS = { confirmed: "已确认", pending_review: "待复核", rejected: "已排除" };

function errorText(error: unknown) { return error instanceof Error ? error.message : String(error); }

function CueReview({ cue, taskId, onChange }: { cue: VideoCue; taskId: string; onChange: (cue: VideoCue) => void }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const review = async (action: "confirm" | "reject") => {
    if (!cue.hit_id || cue.review_revision === null) return;
    setBusy(true);
    setError("");
    try {
      const result = await api.review(taskId, cue.hit_id, action, cue.review_revision);
      onChange({ ...cue, review_status: result.status as VideoCue["review_status"], review_revision: result.revision });
    } catch (e) { setError(errorText(e)); }
    finally { setBusy(false); }
  };
  return <div className="cue-review" onClick={e => e.stopPropagation()}>
    <button type="button" className="rag-button quiet" disabled={busy || cue.review_status === "confirmed"} onClick={() => review("confirm")}><Icon name="check" size={13}/>确认</button>
    <button type="button" className="rag-button quiet" disabled={busy || cue.review_status === "rejected"} onClick={() => review("reject")}><Icon name="close" size={13}/>排除</button>
    {error && <span className="rag-error" role="alert">{error}</span>}
  </div>;
}

export default function App() {
  const [dataset, setDataset] = useState<DatasetStatus | null>(null);
  const [source, setSource] = useState<VideoSource>();
  const [turns, setTurns] = useState<VideoTurn[]>([]);
  const [activeTurnId, setActiveTurnId] = useState("");
  const [active, setActive] = useState<VideoJump | null>(null);
  const [observations, setObservations] = useState<PlaybackObservation[]>([]);
  const [draft, setDraft] = useState("");
  const [target, setTarget] = useState<QueryAsset | null>(null);
  const [file, setFile] = useState<File>();
  const [pickTime, setPickTime] = useState<number>();
  const [showSource, setShowSource] = useState(false);
  const [showReport, setShowReport] = useState(false);
  const [error, setError] = useState("");
  const [mediaError, setMediaError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [sort, setSort] = useState("default");
  const [reviewFilter, setReviewFilter] = useState("all");
  const [visibleCount, setVisibleCount] = useState(6);
  const [similarity, setSimilarity] = useState(0.68);
  const [refresh, setRefresh] = useState(0);
  const messages = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  const upload = useRef<HTMLInputElement>(null);
  const cueRequest = useRef(0);
  const playerPosition = useRef(0);
  const autoStarted = useRef(new Set<string>());
  const latest = turns[turns.length - 1];
  const activeTurn = turns.find(turn => turn.turn_id === activeTurnId && turn.snapshot) ??
    [...turns].reverse().find(turn => turn.snapshot);

  useEffect(() => {
    let live = true;
    api.datasetStatus().then(async status => {
      if (!live) return;
      setDataset(status);
      if (!status.source_asset_id) return;
      const value = await rag.source(status.source_asset_id);
      if (live) setSource(value);
    }).catch(e => { if (live) setError(errorText(e)); });
    return () => { live = false; };
  }, [refresh]);

  useEffect(() => {
    const id = localStorage.getItem(HISTORY_KEY);
    if (!source || !id) return;
    let live = true;
    rag.history(id).then(data => {
      if (live && data.turns[data.turns.length - 1]?.source_asset_id === source.source_asset_id) {
        setTurns(data.turns);
        data.turns.filter(t => t.snapshot).forEach(t => autoStarted.current.add(t.turn_id));
      }
    }).catch(() => { if (live) localStorage.removeItem(HISTORY_KEY); });
    return () => { live = false; };
  }, [source?.source_asset_id]);

  useEffect(() => {
    if (!latest || !running(latest)) return;
    let live = true;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const updated = await rag.turn(latest!.turn_id);
        if (!live) return;
        setTurns(current => current.map(t => t.turn_id === updated.turn_id ? updated : t));
        if (running(updated)) timer = setTimeout(poll, 850);
      } catch (e) {
        if (live) {
          setError(errorText(e));
          timer = setTimeout(poll, 1800);
        }
      }
    }
    timer = setTimeout(poll, 450);
    return () => { live = false; clearTimeout(timer); };
  }, [latest?.turn_id, latest?.status]);

  const playCue = async (turn: VideoTurn, cue: VideoCue) => {
    const request = ++cueRequest.current;
    setActiveTurnId(turn.turn_id);
    setMediaError("");
    try {
      const result = await rag.cue(turn.turn_id, cue.cue_id);
      if (request === cueRequest.current) {
        setObservations(result.items);
        setActive({ cue, key: request });
      }
    } catch (e) {
      if (request === cueRequest.current) setMediaError(errorText(e));
    }
  };

  useEffect(() => {
    if (!latest?.snapshot || autoStarted.current.has(latest.turn_id)) return;
    autoStarted.current.add(latest.turn_id);
    setActiveTurnId(latest.turn_id);
    const first = latest.snapshot.cues.find(cue => cue.cue_id === latest.snapshot?.autoplay_cue_id && cue.review_status !== "rejected") ??
      latest.snapshot.cues.find(cue => cue.review_status !== "rejected");
    if (first) void playCue(latest, first);
    else { ++cueRequest.current; setActive(null); setObservations([]); }
  }, [latest?.turn_id, latest?.snapshot !== null]);

  useEffect(() => {
    const element = messages.current;
    if (element) element.scrollTop = element.scrollHeight;
  }, [turns.length, latest?.reply?.answer, latest?.status]);

  const send = async (question: string) => {
    if (!source || submitting || running(latest)) return;
    const text = question.trim() || (target ? "请在全片查找这个人体的所有相似候选，并给出可回放的证据。" : "");
    if (!text) return;
    setSubmitting(true);
    setError("");
    try {
      const accepted = await rag.send({
        source_asset_id: source.source_asset_id,
        question: text,
        parent_id: latest?.turn_id,
        query_asset_id: target?.asset_id,
        min_similarity: source.frame_indexed && (target || latest?.query_asset_id) ? similarity : undefined,
        focused_cue_id: !target && activeTurnId === latest?.turn_id ? active?.cue.cue_id : undefined,
      });
      localStorage.setItem(HISTORY_KEY, accepted.turn_id);
      setTurns(current => [...current, accepted]);
      setDraft("");
      setTarget(null);
      setReviewFilter("all");
      setVisibleCount(6);
    } catch (e) { setError(errorText(e)); }
    finally { setSubmitting(false); }
  };
  const onSubmit = (event: FormEvent) => { event.preventDefault(); void send(draft); };
  const newChat = () => {
    localStorage.removeItem(HISTORY_KEY);
    ++cueRequest.current;
    setTurns([]); setActive(null); setActiveTurnId(""); setTarget(null); setDraft("");
    setObservations([]); setError(""); setMediaError(""); setVisibleCount(6); setSimilarity(0.68);
    input.current?.focus();
  };
  const useTarget = (qa: QueryAsset) => {
    setFile(undefined); setPickTime(undefined); setTarget(qa);
    input.current?.focus();
  };
  const filtered = (activeTurn?.snapshot?.cues ?? []).filter(cue => reviewFilter === "all" || cue.review_status === reviewFilter);
  const visible = sort === "time" ? [...filtered].sort((l, r) => l.start_us - r.start_us) : filtered;
  const taskId = activeTurn?.search_task_id;
  const preparing = running(latest);
  const status = source?.indexed_end_us ? "知识库就绪" : dataset ? "等待索引" : "连接中";
  const mode = activeTurn?.snapshot?.mode;

  return <div className="rag-shell">
    <header className="masthead">
      <div className="product"><span className="product-mark"><Icon name="video" size={21}/></span><strong>Video RAG</strong></div>
      <span className="masthead-rule" />
      <span className="masthead-name">视频检索工作台</span>
      <div className="source-links">
        <span className="library-state"><i className={dataset?.demo_ingest_run_id ? "online" : ""} />{status}</span>
        <button className="rag-button quiet source-button" type="button" onClick={() => setShowSource(true)}><Icon name="database" size={16}/>视频知识库</button>
      </div>
    </header>
    <main className="rag-layout">
      <section className="conversation pane" aria-label="视频对话">
        <header className="pane-head chat-head">
          <div className="pane-title"><Icon name="message" size={20}/><h1>视频对话</h1></div>
          <button className="rag-button quiet" type="button" disabled={preparing || submitting} onClick={newChat}><Icon name="plus" size={16}/>新对话</button>
        </header>
        <div className="message-feed" ref={messages} aria-live="polite">
          {!turns.length && <div className="chat-welcome">
            <div className="welcome-mark"><Icon name="video" size={23}/></div>
            <h2>想在视频里找什么？</h2>
            <p>上传人体图或从原片选人，可检索相似片段。也可以按片内时间查询。</p>
            <div className="starter-actions">
              <button type="button" onClick={() => setPickTime(0)} disabled={!source}><Icon name="target"/>从原片选人<Icon name="chevron" size={14}/></button>
              <button type="button" onClick={() => void send("请检索 00:10 到 00:15 的人体片段，给我一条便于复核的证据。")}
                disabled={!source || submitting}><Icon name="clock"/>看看 00:10–00:15<Icon name="chevron" size={14}/></button>
            </div>
          </div>}
          {turns.map(turn => {
            const refs = (turn.reply?.citation_ids ?? []).flatMap(id => {
              const cue = turn.snapshot?.cues.find(item => item.cue_id === id);
              return cue ? [cue] : [];
            });
            return <div className="turn" key={turn.turn_id} data-turn-id={turn.turn_id} data-status={turn.status}>
              <div className="user-message"><p>{turn.question}</p></div>
              <div className="assistant-message">
                <span className="assistant-mark"><Icon name="video" size={15}/></span>
                <div className="assistant-body">
                  <div className="assistant-label">{turn.provider === "qwen" ? "Qwen · 视频知识库" : "视频知识库"}</div>
                  {running(turn) && <div className="generation-state" role="status"><i />
                    {turn.status === "GENERATING" ? "证据已返回，Qwen 正在整理回答…" : "正在检索已索引的视频片段…"}
                  </div>}
                  {turn.error && <p className="rag-error" role="alert">{turn.error}</p>}
                  {turn.reply?.answer && <p className="answer">
                    {turn.reply.answer.split(/(E[1-9][0-9]*)/).map((part, index) => {
                      const id = turn.reply?.citation_labels?.[part];
                      const cue = id && refs.find(item => item.cue_id === id);
                      return cue ? <button type="button" className="inline-citation" key={index}
                        title="播放这条引用" onClick={() => void playCue(turn, cue)}>{clock(cue.matched_us ?? cue.start_us, true)}</button> : part;
                    })}
                  </p>}
                  {turn.review_updated && turn.reply?.answer && <p className="answer-footnote">复核状态已更新。可追问以整理最新结论。</p>}
                  {refs.length > 0 && <div className="citations" aria-label="回答引用">
                    {refs.map((cue, index) => <button type="button" key={cue.cue_id} className="citation"
                      onClick={() => void playCue(turn, cue)} aria-label={`播放引用 ${index + 1}，${clock(cue.matched_us ?? cue.start_us, true)}`}>
                      <Icon name="play" size={12}/><span>{clock(cue.matched_us ?? cue.start_us, true)}</span><Icon name="chevron" size={12}/>
                    </button>)}
                  </div>}
                </div>
              </div>
            </div>;
          })}
        </div>
        <div className="composer-area">
          {error && <p className="rag-error" role="alert">{error}</p>}
          <form className="composer" onSubmit={onSubmit}>
            {target && <div className="attachment">
              {target.image_url && <img src={target.image_url} alt="当前检索图" />}
              <span>{target.local_us != null ? `原片目标 · ${clock(target.local_us, true)}` : "已选上传目标"}</span>
              <button type="button" aria-label="移除目标" className="icon-button" onClick={() => setTarget(null)}><Icon name="close" size={14}/></button>
            </div>}
            <textarea ref={input} rows={3} maxLength={800} value={draft} aria-label="向视频知识库提问"
              placeholder={target ? "问这个目标出现在哪些片段…" : "问视频里的片段，或继续追问…"}
              onChange={e => setDraft(e.target.value)}
              onKeyDown={e => {
                if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                  e.preventDefault(); void send(draft);
                }
              }} />
            <div className="composer-tools">
              <input ref={upload} hidden type="file" accept="image/*" onChange={e => {
                const next = e.target.files?.[0]; e.target.value = ""; if (next) setFile(next);
              }} />
              <button type="button" className="rag-button quiet" disabled={!source || preparing || submitting} onClick={() => upload.current?.click()}><Icon name="clip" size={16}/>上传图片</button>
              <button type="button" className="rag-button quiet" disabled={!source || preparing || submitting}
                onClick={() => setPickTime(playerPosition.current)}><Icon name="target" size={16}/>原片选人</button>
              {source?.frame_indexed && (target || latest?.query_asset_id) && <select className="similarity-filter"
                aria-label="最低外观相似度" value={similarity} disabled={preparing || submitting}
                onChange={event => setSimilarity(Number(event.target.value))}>
                <option value={0.55}>最低 0.55 · 放宽</option>
                <option value={0.68}>最低 0.68 · 标准</option>
                <option value={0.8}>最低 0.80 · 严格</option>
              </select>}
              <button className="rag-button solid send-button" type="submit" disabled={!source || preparing || submitting || (!target && !draft.trim())}>
                <span>发送</span><Icon name="send" size={16}/>
              </button>
            </div>
          </form>
          <p className="composer-note">{latest?.query_asset_id && !target ? "后续问题会沿用该人体目标" : "时间均为原片内位置"}</p>
        </div>
      </section>

      <section className="evidence pane" aria-label="视频证据">
        <header className="pane-head evidence-head">
          <div className="pane-title"><Icon name="video" size={20}/><h2>原片回放</h2></div>
          <div className="evidence-tools">
            {source && <span className="indexed-label"><Icon name="clock" size={14}/>{source.frame_indexed && source.indexed_end_us === source.duration_us ? "全片索引" : "索引"} 00:00–{clock(source.indexed_end_us)}</span>}
            {taskId && <button type="button" className="rag-button quiet" onClick={() => setShowReport(true)}><Icon name="document" size={16}/>正式报告</button>}
          </div>
        </header>
        {source ? <VideoRagPlayer source={source} cues={activeTurn?.snapshot?.cues ?? []} active={active}
          observations={observations} onCue={cue => { if (activeTurn) void playCue(activeTurn, cue); }}
          onPick={setPickTime} onPosition={us => { playerPosition.current = us; }} /> : <div className="watch-placeholder"><Icon name="video" size={30}/><span>等待视频知识库连接</span></div>}
        {mediaError && <p className="rag-error media-error" role="alert">{mediaError}</p>}
        <div className="results">
          <div className="result-head">
            <div className="pane-title"><Icon name="list" size={19}/><h2>检索结果</h2>{mode && <span className="result-count">{filtered.length} 个{mode === "appearance" ? "候选" : "片段"}</span>}</div>
            {mode && <div className="result-tools">
              {mode === "appearance" && <select aria-label="筛选复核状态" value={reviewFilter} onChange={e => setReviewFilter(e.target.value)}>
                <option value="all">全部状态</option><option value="pending_review">待复核</option><option value="confirmed">已确认</option><option value="rejected">已排除</option>
              </select>}
              <select aria-label="候选排序" value={sort} onChange={e => setSort(e.target.value)}>
                <option value="default">{mode === "appearance" ? "按相似度" : "检索顺序"}</option><option value="time">按片内时间</option>
              </select>
            </div>}
          </div>
          {activeTurn?.snapshot?.min_similarity != null && <p className="result-notice">
            {clock(activeTurn.snapshot.window_us[0])}–{clock(activeTurn.snapshot.window_us[1])} · 外观相似度 &gt; {activeTurn.snapshot.min_similarity.toFixed(2)} · 候选待人工复核
          </p>}
          {activeTurn?.snapshot?.partial && <p className="result-notice">当前结果仅覆盖部分索引或召回受限。</p>}
          {!visible.length && <div className="results-empty"><Icon name="target" size={23}/>
            <div><strong>{activeTurn ? "这次没有可显示的片段" : "检索证据会显示在这里"}</strong>
            <p>{activeTurn ? "可降低相似度门槛后重查，或调整时间和复核筛选。" : "点候选或对话引用，直接跳到原片并播放。"}</p></div>
          </div>}
          {activeTurn && visible.length > 0 && <div className="cue-grid">
            {visible.slice(0, visibleCount).map(cue => {
              const current = active?.cue.cue_id === cue.cue_id && activeTurnId === activeTurn.turn_id;
              return <article className={`cue ${current ? "current" : ""}`} key={cue.cue_id}>
                <button className="cue-preview" type="button" aria-label={`播放候选 ${clock(cue.matched_us ?? cue.start_us, true)}`}
                  onClick={() => void playCue(activeTurn, cue)}>
                  {cue.image_url && <img src={cue.image_url} alt="" loading="lazy"/>}
                  <span className="cue-play"><Icon name="play" size={13}/>{clock(cue.matched_us ?? cue.start_us, true)}</span>
                </button>
                <div className="cue-info">
                  <div className="cue-line"><strong>{cue.same_continuation ? "同跟踪段后续" : mode === "appearance" ? "相似人体片段" : "人体观测片段"}</strong>
                    <span className={`cue-state ${cue.review_status ?? "observed"}`}>{cue.review_status ? STATUS[cue.review_status] : "轨迹 #" + cue.track_number}</span></div>
                  <div className="cue-meta"><time>{isSingleFrameCue(cue)
                    ? `单帧命中 · ${clock(cue.start_us, true)}`
                    : `${clock(cue.start_us, true)} – ${clock(cue.end_us, true)}`}</time>
                    {cue.score != null && <span>相似度 {cue.score.toFixed(3)}</span>}</div>
                  {taskId && cue.hit_id && <CueReview cue={cue} taskId={taskId}
                    onChange={updated => setTurns(currentTurns => currentTurns.map(turn => turn.search_task_id === taskId && turn.snapshot ?
                      { ...turn, review_updated: true, snapshot: { ...turn.snapshot, cues: turn.snapshot.cues.map(item => item.cue_id === updated.cue_id ? updated : item) } } : turn))} />}
                </div>
              </article>;
            })}
          </div>}
          {visible.length > visibleCount && <button className="rag-button quiet more-results" type="button" onClick={() => setVisibleCount(v => v + 9)}>显示更多片段</button>}
          {taskId && <div className="result-downloads">
            <a href={api.exportUrl(taskId, "json")} download><Icon name="download" size={14}/>结果 JSON</a>
            <a href={api.exportUrl(taskId, "csv")} download><Icon name="download" size={14}/>结果 CSV</a>
          </div>}
        </div>
      </section>
    </main>

    {showSource && <RagDialog title="视频知识库" onClose={() => { setShowSource(false); setRefresh(i => i + 1); }}><div className="legacy-dialog"><DatasetView source={source}/></div></RagDialog>}
    {showReport && taskId && <RagDialog title="正式报告与导出" onClose={() => setShowReport(false)}><div className="legacy-dialog"><ReportView taskId={taskId}/></div></RagDialog>}
    {source && pickTime !== undefined && <FrameTargetPicker source={source} initialTime={pickTime} onClose={() => setPickTime(undefined)} onTarget={useTarget}/>}
    {file && <ImageTargetPicker file={file} onClose={() => setFile(undefined)} onTarget={useTarget}/>}
  </div>;
}
