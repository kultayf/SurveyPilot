<script setup lang="ts">
import { computed, ref, watch } from "vue";
import type { SessionArtifact } from "../../types/sessions";

// 只读当前会话的已保存报告；切换会话时取消旧请求，避免把上一会话证据显示过来。
const props = defineProps<{ sessionKey: string; artifacts: SessionArtifact[]; running?: boolean; waitingForConfirmation?: boolean }>();
const emit = defineEmits<{ updated: []; resume: [] }>();
interface Evidence { chunkId: string; paperId: string; quote: string; page_start?: number; page_end?: number }
interface Cell { value: string; status: string; evidence: Evidence[]; edited_by?: string; review_note?: string }
interface MatrixRow { paperId: string; title: string; year?: number; status: string; reason?: string; cells: Record<string, Cell> }
interface Matrix { dimensions: Record<string, string>; rows: MatrixRow[]; located_cells: number; total_cells: number; review?: { revision: number; confirmed: boolean; confirmed_by?: string; audit?: { status: string; model?: string; rows?: { section_id: string; units: AuditUnit[] }[] } } }
interface AuditUnit { index: number; claim: string; status: string; reason: string; evidence: Evidence[] }
interface Audit { status: string; model: string; revision: number; sections: { section_id: string; status: string; reason?: string; units: AuditUnit[] }[] }
interface CitationGraph {
  limits?: { depth: number; pages_per_direction?: number };
  co_citations?: { paper_ids: string[]; citing_paper_count: number }[];
  papers: { paperId: string; title: string }[];
  seed_papers: { paperId: string; title: string }[];
  links: { source: string; target: string; direction: string }[];
  added_paper_ids: string[];
  errors: unknown[];
  skipped: unknown[];
  truncated: boolean;
}
const conflicts = ref<{ status: string; reason?: string; scope: string; findings: { kind: string; statement: string; comparison_conditions: string; evidence: Evidence[] }[] } | null>(null);
const graph = ref<CitationGraph | null>(null);
const matrix = ref<Matrix | null>(null);
const audit = ref<Audit | null>(null);
const error = ref("");
const filter = ref("");
const reviewer = ref("");
const saving = ref(false);
const editing = ref<{ paperId: string; dimension: string; value: string; note: string; evidence: { chunkId: string; quote: string }[] } | null>(null);
function editCell(row: MatrixRow, dimension: string) {
  const cell = row.cells[dimension];
  editing.value = { paperId: row.paperId, dimension, value: cell.value, note: cell.review_note || "",
    evidence: cell.evidence.map((item) => ({ chunkId: item.chunkId, quote: item.quote })) };
}
async function updateMatrix(action: "edit" | "confirm" | "recheck") {
  const sessionKey = props.sessionKey;
  const artifactId = sourceIds.value[0];
  if (!artifactId || saving.value) return;
  saving.value = true; error.value = "";
  try {
    const response = await fetch(`/api/sessions/${encodeURIComponent(sessionKey)}/matrix-reviews`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ base_artifact_id: artifactId, action, reviewer: reviewer.value, ...(action === "edit" ? editing.value : {}) }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error?.message || data.detail || "矩阵操作失败");
    // 切换会话后不能把旧请求返回的矩阵写进新页面。
    if (props.sessionKey !== sessionKey) return;
    matrix.value = data.matrix; editing.value = null;
    emit("updated");
  } catch (cause) {
    if (props.sessionKey === sessionKey) error.value = cause instanceof Error ? cause.message : "矩阵操作失败";
  } finally { saving.value = false; }
}
const onlyProblems = ref(true);
const labels: Record<string, string> = { supported: "原文支持", not_required: "无事实主张", insufficient: "证据不足", contradicted: "与原文矛盾", invalid_citation: "引用无效", unverified: "未验证" };
const latest = (name: string) => [...props.artifacts].reverse().find((item) => item.name === name);
const sourceIds = computed(() => {
  const files = [latest("evidence_matrix.json"), latest("citation_audit.json"), latest("citation_graph.json"), latest("conflict_report.json")];
  // 以最新报告所属轮次为准，不能把上一轮“核查通过”配在本轮矩阵或引文图下面。
  const newest = [...props.artifacts].reverse().find((item) => files.some((file) => file?.id === item.id));
  return files.map((file) => file?.metadata.turn_id === newest?.metadata.turn_id ? file?.id : undefined);
});
watch(() => [props.sessionKey, ...sourceIds.value], async (_, __, onCleanup) => {
  const controller = new AbortController();
  onCleanup(() => controller.abort());
  matrix.value = null;
  audit.value = null;
  graph.value = null;
  conflicts.value = null;
  error.value = "";
  editing.value = null;
  if (!props.sessionKey) return;
  const key = props.sessionKey;
  const ids = [...sourceIds.value];
  // 报告分别加载；其中一份不可用不隐藏其他已经保存的结果。
  const outcomes = await Promise.allSettled(ids.map(async (id) => {
    if (!id) return null;
    const response = await fetch(`/api/sessions/${encodeURIComponent(key)}/artifacts/${encodeURIComponent(id)}`, { signal: controller.signal });
    if (!response.ok) throw new Error(`报告读取失败 (${response.status})`);
    return response.json();
  }));
  if (controller.signal.aborted) return;
  outcomes.forEach((result, index) => {
    if (result.status === "rejected") { error.value = "部分证据报告读取失败，可通过右侧输出结果重新下载。"; return; }
    const value = result.value;
    if (!value) return;
    if (index === 0 && Array.isArray(value.rows) && value.dimensions) matrix.value = value;
    else if (index === 1 && Array.isArray(value.sections)) audit.value = value;
    else if (index === 2 && Array.isArray(value.links)) graph.value = value;
    else if (index === 3 && Array.isArray(value.findings)) conflicts.value = value;
    else error.value = "报告格式无法识别，请下载原始文件检查。";
  });
}, { immediate: true });
const rows = computed(() => matrix.value?.rows.filter((row) => {
  const query = filter.value.trim().toLocaleLowerCase();
  return !query || `${row.title} ${row.paperId} ${Object.values(row.cells).map((cell) => cell.value).join(" ")}`.toLocaleLowerCase().includes(query);
}) ?? []);
const auditSections = computed(() => audit.value?.sections.map((section) => ({ ...section,
  units: section.units.filter((unit) => !onlyProblems.value || !["supported", "not_required"].includes(unit.status)),
})).filter((section) => section.units.length || section.status !== "passed") ?? []);
/** 图中统一使用“引用者 → 被引用者”，优先显示论文题名。 */
function graphTitle(id: string) {
  return [...(graph.value?.seed_papers ?? []), ...(graph.value?.papers ?? [])].find((paper) => paper.paperId === id)?.title || id;
}
</script>

<template>
  <section v-if="matrix || audit || graph || conflicts || error" id="matrix-review-panel" class="research-panel" aria-label="实证矩阵与引用核查">
    <p v-if="error" role="alert">{{ error }}</p>
    <details v-if="graph" class="citation-graph">
      <summary>引文扩展 · {{ graph.papers.length }} 篇候选 · {{ graph.added_paper_ids.length }} 篇进入阅读</summary>
      <p>箭头表示“引用者 → 被引用者”。最多展开 3 篇种子的 {{ graph.limits?.depth ?? 1 }} 层、每方向 {{ graph.limits?.pages_per_direction ?? 1 }} 页，不能保证找全领域文献。</p>
      <p v-if="graph.truncated || graph.errors.length">部分关系未取全或接口暂不可用，完整记录见 citation_graph.json。</p>
      <p v-if="graph.skipped.length">{{ graph.skipped.length }} 个种子缺少可识别编号，已跳过。</p>
      <ul class="citation-links"><li v-for="(link, index) in graph.links" :key="index">{{ graphTitle(link.source) }} → {{ graphTitle(link.target) }}</li></ul>
      <details v-if="graph.co_citations?.length"><summary>本次图中的共同引用</summary>
        <p v-for="(pair, index) in graph.co_citations" :key="index">{{ pair.paper_ids.map(graphTitle).join(' 与 ') }}：被 {{ pair.citing_paper_count }} 篇已取得的论文共同引用</p>
        <p>这是局部计数，不代表全领域排名。</p>
      </details>
      <p v-if="!graph.links.length">本次没有找到可展示的引文关系。</p>
    </details>
    <details v-if="matrix" open>
      <summary>实证矩阵 · {{ matrix.rows.length }} 篇论文 · {{ matrix.located_cells }}/{{ matrix.total_cells }} 个字段已定位</summary>
      <p>点击单元格展开原句、页码与来源。未找到仅表示本次读取范围内没有找到，原文已定位仍需核查其含义。</p>
      <div class="matrix-review-controls">
        <label>审阅者 <input v-model="reviewer" maxlength="80" placeholder="姓名或本地标识" :disabled="saving || running" /></label>
        <button type="button" :disabled="saving || running || !reviewer.trim()" @click="updateMatrix('confirm')">确认整张矩阵</button>
        <button type="button" :disabled="saving || running" @click="updateMatrix('recheck')">{{ saving ? '正在保存或核查…' : '重新核查矩阵' }}</button>
        <button v-if="waitingForConfirmation && matrix.review?.confirmed && matrix.review.audit?.status === 'passed'" type="button" :disabled="saving || running" @click="emit('resume')">从确认阶段继续</button>
      </div>
      <p v-if="matrix.review">修订 {{ matrix.review.revision }} · {{ matrix.review.confirmed ? `人工已确认（${matrix.review.confirmed_by}）` : '尚未确认' }} · 矩阵核查：{{ matrix.review.audit?.status === 'passed' ? '模型核查通过' : '尚未通过' }}</p>
      <p>修改保存为新版本，不改写旧综述；再次修改会清除旧确认和核查结果。原文引用需与论文一致，缺失项请留空。</p>
      <details v-if="matrix.review?.audit?.rows?.length">
        <summary>查看矩阵核查详情</summary>
        <div v-for="row in matrix.review.audit.rows" :key="row.section_id">
          <p v-for="unit in row.units" :key="unit.index">{{ row.section_id }} · {{ labels[unit.status] || unit.status }}：{{ unit.claim }} {{ unit.reason }}</p>
        </div>
      </details>
      <form v-if="editing" class="matrix-edit-form" @submit.prevent="updateMatrix('edit')">
        <strong>编辑 {{ editing.paperId }} · {{ matrix.dimensions[editing.dimension] }}</strong>
        <label>实证内容（缺失时留空）<textarea v-model="editing.value" maxlength="8000" rows="3" /></label>
        <label>修改说明<textarea v-model="editing.note" maxlength="2000" rows="2" /></label>
        <div v-for="(source, index) in editing.evidence" :key="index">
          <label>原文片段编号<input v-model="source.chunkId" /></label>
          <label>连续原文<textarea v-model="source.quote" rows="2" /></label>
          <button type="button" @click="editing.evidence.splice(index, 1)">移除此依据</button>
        </div>
        <button type="button" :disabled="editing.evidence.length >= 8" @click="editing.evidence.push({ chunkId: '', quote: '' })">添加原文依据</button>
        <button type="submit" :disabled="saving || running || !reviewer.trim()">保存修订</button>
        <button type="button" @click="editing = null">取消编辑</button>
      </form>
      <label class="filter-label">筛选论文或实证内容 <input v-model="filter" type="search" placeholder="输入题名、数据集或方法名称" /></label>
      <div class="matrix-scroll" tabindex="0" aria-label="横向滚动实证矩阵">
        <table>
          <thead><tr><th scope="col">论文</th><th v-for="(label, key) in matrix.dimensions" :key="key" scope="col">{{ label.split(" / ")[0] }}</th></tr></thead>
          <tbody>
            <tr v-for="row in rows" :key="row.paperId">
              <th scope="row">{{ row.title }}<small>{{ row.year || "年份未知" }} · {{ row.paperId }}</small><small v-if="row.reason">{{ row.reason }}</small></th>
              <td v-for="(_, key) in matrix.dimensions" :key="key">
                <details v-if="row.cells[key]?.evidence.length">
                  <summary>{{ row.cells[key].value }}</summary>
                  <p class="source-status">原文已定位 · {{ row.cells[key].edited_by ? `人工修订：${row.cells[key].edited_by}` : "模型提取" }}</p>
                  <blockquote v-for="evidence in row.cells[key].evidence" :key="evidence.chunkId">
                    {{ evidence.quote }}
                    <small>{{ evidence.paperId }} · 页 {{ evidence.page_start ?? "?" }}–{{ evidence.page_end ?? "?" }}<br />{{ evidence.chunkId }}</small>
                  </blockquote>
                </details>
                <span v-else class="missing">{{ row.status === "unverified" ? "未验证" : "未找到" }}</span>
                <button class="cell-edit" type="button" :disabled="saving || running" @click="editCell(row, String(key))">编辑</button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <p v-if="!rows.length">没有匹配的论文。</p>
    </details>
    <details v-if="conflicts" class="audit-panel">
      <summary>跨文献比较线索 · {{ conflicts.findings.length }} 项待人工复核</summary>
      <p>{{ conflicts.scope }}。{{ conflicts.reason }}</p>
      <details v-for="(finding, index) in conflicts.findings" :key="index">
        <summary>{{ finding.kind === 'conflict' ? '可能存在争议' : finding.kind === 'gap' ? '当前证据缺口' : '比较条件不同' }}：{{ finding.statement }}</summary>
        <p>{{ finding.comparison_conditions }}</p>
        <blockquote v-for="source in finding.evidence" :key="source.chunkId + source.quote">{{ source.quote }}<small>{{ source.paperId }} · {{ source.chunkId }}</small></blockquote>
      </details>
      <p v-if="!conflicts.findings.length">未输出比较线索不代表文献间不存在争议。</p>
    </details>
    <details v-if="audit" open class="audit-panel">
      <summary>{{ audit.status === "passed" ? "独立模型核查通过" : "待核查草稿：仍有证据不足或未验证内容" }}</summary>
      <p>核查包含正文与摘要，结果仍需科研人员复核。修订次数：{{ audit.revision }}。</p>
      <label><input v-model="onlyProblems" type="checkbox" /> 只看需要处理的项目</label>
      <div v-for="section in auditSections" :key="section.section_id" class="audit-section">
        <h4>{{ section.section_id }}</h4><p v-if="section.reason">{{ section.reason }}</p>
        <details v-for="unit in section.units" :key="unit.index">
          <summary>{{ labels[unit.status] || "未验证" }} · {{ unit.claim.slice(0, 100) }}{{ unit.claim.length > 100 ? "…" : "" }}</summary>
          <p class="claim">{{ unit.claim }}</p><p>{{ unit.reason }}</p>
          <blockquote v-for="evidence in unit.evidence" :key="evidence.chunkId + evidence.quote">{{ evidence.quote }}<small>{{ evidence.paperId }} · 页 {{ evidence.page_start ?? "?" }}–{{ evidence.page_end ?? "?" }} · {{ evidence.chunkId }}</small></blockquote>
        </details>
      </div>
      <p v-if="!auditSections.length">当前筛选下没有需要处理的项目。</p>
    </details>
  </section>
</template>

<style scoped>
.research-panel { padding: 20px; margin: 16px 0; border: 1px solid #d4dbe5; border-radius: 14px; background: var(--surface, #fff); min-width: 0; }
summary { cursor: pointer; font-weight: 600; line-height: 1.6; }
p, label { font-size: 13px; line-height: 1.7; }
.filter-label { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin: 12px 0; }
input[type="search"] { min-width: 220px; padding: 8px 10px; border: 1px solid #b7c3d1; border-radius: 6px; }
.matrix-review-controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 12px 0; }
button { border: 1px solid #bac6d3; border-radius: 6px; padding: 6px 9px; background: #f5f8fc; cursor: pointer; }
button:disabled { opacity: .5; cursor: default; }
.cell-edit { display: block; margin-top: 10px; font-size: 11px; }
.matrix-edit-form { padding: 14px; border: 1px solid #b9c8dc; border-radius: 8px; margin: 10px 0; }
.matrix-edit-form label { display: block; margin: 10px 0; }
.matrix-edit-form input, .matrix-edit-form textarea { display: block; width: 100%; box-sizing: border-box; padding: 8px; border: 1px solid #bac6d3; border-radius: 6px; }
.citation-graph { margin-bottom: 18px; }
.citation-links { max-height: 280px; overflow: auto; font-size: 12px; line-height: 1.8; }
.matrix-scroll { overflow: auto; max-height: 520px; }
table { border-collapse: collapse; font-size: 12px; }
th, td { border: 1px solid #dce2eb; padding: 10px; vertical-align: top; min-width: 190px; max-width: 320px; overflow-wrap: anywhere; }
thead th { position: sticky; top: 0; background: #f2f5f9; z-index: 1; }
th[scope="row"] { text-align: left; }
td summary { font-weight: normal; max-height: 140px; overflow: auto; }
small { display: block; margin-top: 6px; font-size: 11px; font-weight: normal; color: #526177; overflow-wrap: anywhere; }
.missing, .source-status { color: #627087; }
blockquote { margin: 10px 0; padding-left: 10px; border-left: 3px solid #94a9bd; white-space: pre-wrap; overflow-wrap: anywhere; }
.audit-panel { margin-top: 20px; }
.audit-section { margin: 14px 0; padding: 12px; background: #f7f8fa; border-radius: 8px; }
.audit-section details { margin: 10px 0; }
.claim { white-space: pre-wrap; }
</style>
