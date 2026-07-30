#ifndef SEQUANT_CORE_EXPORT_CONTRACTION_IR_GENERATOR_HPP
#define SEQUANT_CORE_EXPORT_CONTRACTION_IR_GENERATOR_HPP

// ContractionIRGenerator -- a human-readable "Contraction IR" (CTIR) backend.
//
// Motivation (see sequant-ta-repro/docs/CONTRACTION_IR.md): the TiledArray
// einsum export is an *execution artifact* -- a flattened, linear list of
// TA::einsum calls with implicit sharing and no cost/semantic annotation. It
// cannot express (a) the per-outer-cell cost that dominates the CSV-CCSD cold
// runtime -- the giant (mu-tilde,K)-outer DF half-transform runs ~100x off
// peak purely from having millions of tiny per-pair ToT cells, and that fact
// is invisible in an einsum call -- nor (b) what MPQC actually does at runtime:
// a DAG over a CacheManager with amplitude-independent (t-indep) intermediates
// that are built once and reused across CC iterations (persistent), plus
// optional aux-K batched evaluation.
//
// CTIR makes all of that explicit on ONE shared DAG (the same post-CSE forest
// the einsum export consumes), so cross-term sharing shows as `uses=N`:
//   - the tensor-of-tensor OUTER;INNER split with its per-pair domain <i,j>;
//   - a cost line: `cells` (# outer ToT cells = # independent tasks), inner
//     extent, flops, per-cell, and a `CELL-BOUND` flag when the outer-cell
//     count is huge (the DF-half-transform pathology);
//   - `t-indep`/`t-dep` (volatile) classification and, at the t-indep/t-dep
//     boundary, `persistent` (MPQC builds it once, survives reset() across
//     iterations -- the cache_manager.hpp V->NP / NV-with-V-consumer->P rule);
//   - `batchable/K` where a contraction sums over the DF aux index (the
//     aux-K batching hook, make_batched_custom_evaluator).
//
// It is descriptive only (no lowering); it reuses the export framework's
// traversal, CSE and ref-count liveness, exactly like the einsum backend.

#include <SeQuant/core/export/context.hpp>
#include <SeQuant/core/export/generator.hpp>
#include <SeQuant/core/expr.hpp>
#include <SeQuant/core/index.hpp>
#include <SeQuant/core/utility/string.hpp>

#include <cmath>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace sequant {

/// Context for ContractionIRGenerator. `extent` supplies per-index sizes so the
/// IR can print cell counts / flops; default falls back to the index space's
/// own approximate size. Set it from the same idx_to_extent the optimizer used
/// so the numbers match the cost model that chose this factorization.
struct ContractionIRGeneratorContext : ExportContext {
  std::function<std::size_t(const Index &)> extent =
      [](const Index &idx) -> std::size_t {
    return idx.space().approximate_size();
  };
};

template <typename Context = ContractionIRGeneratorContext>
class ContractionIRGenerator : public Generator<Context> {
 public:
  ContractionIRGenerator() = default;
  ~ContractionIRGenerator() = default;

  std::string get_format_name() const override { return "Contraction IR"; }
  bool supports_named_sections() const override { return true; }
  bool requires_named_sections() const override { return false; }
  DeclarationScope index_declaration_scope() const override {
    return DeclarationScope::Global;
  }
  DeclarationScope variable_declaration_scope() const override {
    return DeclarationScope::Section;
  }
  DeclarationScope tensor_declaration_scope() const override {
    return DeclarationScope::Section;
  }
  PrunableScalars prunable_scalars() const override {
    return PrunableScalars::All;
  }

  std::string represent(const Index &idx, const Context &) const override {
    return toUtf8(idx.label());
  }
  std::string represent(const Tensor &tensor, const Context &) const override {
    return var_name(tensor);
  }
  std::string represent(const Variable &v, const Context &) const override {
    return toUtf8(v.label());
  }
  std::string represent(const Constant &c, const Context &) const override {
    std::ostringstream oss;
    oss << to_double(c.value().real());
    return oss.str();
  }
  std::string represent(const Power &, const Context &) const override {
    return "pow(...)";
  }
  std::string wrap_conj(std::string s) const override { return s; }

  // --- leaf discovery ------------------------------------------------------
  void declare(const Tensor &tensor, UsageSet usage,
               const Context &ctx) override {
    if (usage == Usage::Terminal) register_leaf(tensor, ctx);
  }
  void declare(const Variable &, UsageSet, const Context &) override {}
  void declare(const Index &, const Context &) override {}

  // --- the computation: one contraction step == one CTIR `def` -------------
  void compute(const Expr &expression, const Tensor &result,
               const Context &ctx) override {
    Def d;
    d.name = var_name(result);
    split_indices(result, d.outer, d.inner, d.pair_key);
    d.is_tot = !d.inner.empty();

    // Collect operands + scalar from the (binarized) expression.
    std::vector<Tensor> operands;
    collect(expression, operands, d.scalar_str, ctx);

    // Result index labels, to identify contracted (summed) indices.
    std::set<std::wstring> result_idx;
    for (const Index &i : result.const_indices())
      result_idx.insert(std::wstring(i.label()));

    bool any_operand_volatile = false;
    std::set<std::wstring> seen_contracted;
    for (const Tensor &op : operands) {
      const std::string oname = var_name(op);
      d.operand_names.push_back(oname);
      m_uses[oname]++;
      if (is_volatile_name(op, oname)) any_operand_volatile = true;
      for (const Index &i : op.const_indices()) {
        const std::wstring lbl(i.label());
        if (result_idx.count(lbl) || seen_contracted.count(lbl)) continue;
        seen_contracted.insert(lbl);
        d.contracted.push_back(i);
        if (space_key(i) == L"K" || space_key(i) == L"Κ") {
          d.batch_aux = true;
          d.batch_axis = toUtf8(i.label());
        }
      }
    }
    d.volatile_ = any_operand_volatile;

    // Cost. `cells` = number of OUTER (block-sparse) ToT cells = number of
    // independent per-cell tasks. The outer axes are the non-proto direct
    // indices (e.g. μ̃, Κ) AND the pair-key proto-target indices (the occupied
    // pair i_1,i_2 -- each pair is a distinct block; these appear only as proto
    // tags, never as direct indices, so they must be counted explicitly).
    d.cells = 1;
    for (const Index &i : result.const_indices())
      if (!i.has_proto_indices()) d.cells *= ext(ctx, i);
    for (const Index &i : d.pair_key) d.cells *= ext(ctx, i);
    // inner extent = per-pair (proto) axes; use the ORIGINAL proto-carrying
    // indices so idx_to_extent returns the PNO/proto extent, not the
    // dropped-proto fallback.
    d.inner_ext = 1;
    for (const Index &i : result.const_indices())
      if (i.has_proto_indices()) d.inner_ext *= ext(ctx, i);
    // flops ~= product of every distinct index extent in the contraction
    // (outer direct + pair-key + inner + contracted).
    double flops = 1.0;
    std::set<std::wstring> counted;
    auto mul_idx = [&](const Index &i) {
      const std::wstring lbl(i.label());
      if (counted.insert(lbl).second) flops *= static_cast<double>(ext(ctx, i));
    };
    for (const Index &i : result.const_indices()) mul_idx(i);
    for (const Index &i : d.pair_key) mul_idx(i);
    for (const Tensor &op : operands)
      for (const Index &i : op.const_indices()) mul_idx(i);
    d.flops = flops;
    d.per_cell = d.cells ? flops / static_cast<double>(d.cells) : flops;
    d.cell_bound = d.is_tot && d.cells >= kCellBoundThreshold;

    d.is_accumulation = !m_written.insert(d.name).second;
    d.writes_result = (d.name == m_result_name);

    m_def_of[d.name] = m_defs.size();
    m_order.push_back(Event{Event::Def, m_defs.size()});
    m_defs.push_back(std::move(d));
  }
  void compute(const Expr &, const Variable &, const Context &) override {}

  // --- static (repro) liveness: record release points ----------------------
  void unload(const Tensor &tensor, const Context &ctx) override {
    const std::string name = var_name(tensor);
    if (m_leaf.count(name)) return;  // parameters are never freed
    m_order.push_back(Event{Event::Free, m_defs.size()});
    m_freed.push_back(name);
  }
  void destroy(const Tensor &tensor, const Context &ctx) override {
    unload(tensor, ctx);
  }
  void persist(const Tensor &tensor, const Context &) override {
    m_result_name = var_name(tensor);
  }

  // --- unused Generator hooks (no-ops) -------------------------------------
  void create(const Tensor &, bool, const Context &) override {}
  void load(const Tensor &, bool, const Context &) override {}
  void set_to_zero(const Tensor &, const Context &) override {}
  void create(const Variable &, bool, const Context &) override {}
  void load(const Variable &, bool, const Context &) override {}
  void set_to_zero(const Variable &, const Context &) override {}
  void unload(const Variable &, const Context &) override {}
  void destroy(const Variable &, const Context &) override {}
  void persist(const Variable &, const Context &) override {}
  void all_indices_declared(std::size_t, const Context &) override {}
  void all_variables_declared(std::size_t, const Context &) override {}
  void all_tensors_declared(std::size_t, const Context &) override {}
  void begin_declarations(DeclarationScope, const Context &) override {}
  void end_declarations(DeclarationScope, const Context &) override {}
  void insert_comment(const std::string &, const Context &) override {}
  void begin_named_section(std::string_view name, const Context &) override {
    m_fn = std::string(name);
  }
  void end_named_section(std::string_view, const Context &) override {}
  void begin_expression(const Context &) override {}
  void end_expression(const Context &) override {}
  void begin_export(const Context &) override { reset(); }
  void end_export(const Context &) override {}

  std::string get_generated_code() const override { return render(); }

 private:
  // ------------------------------------------------------------------ types
  struct Leaf {
    std::string name, label, outer, inner, pair;
    bool is_tot = false, volatile_ = false;
  };
  struct Def {
    std::string name;
    std::vector<Index> outer, inner, pair_key, contracted;
    bool is_tot = false, is_accumulation = false, volatile_ = false,
         cell_bound = false, batch_aux = false, writes_result = false;
    std::vector<std::string> operand_names;
    std::string scalar_str, batch_axis;
    std::size_t cells = 0, inner_ext = 0;
    double flops = 0, per_cell = 0;
  };
  struct Event {
    enum Kind { Def, Free } kind;
    std::size_t idx;  // Def: index into m_defs; Free: index into m_freed
  };

  static constexpr std::size_t kCellBoundThreshold = 1000000;  // 1e6 ToT cells

  // ---------------------------------------------------------------- helpers
  static double to_double(const rational &r) {
    return static_cast<double>(numerator(r)) /
           static_cast<double>(denominator(r));
  }
  static std::wstring space_key(const Index &i) {
    return std::wstring(i.space().base_key());
  }
  static std::size_t ext(const Context &ctx, const Index &i) {
    return ctx.extent ? ctx.extent(i) : i.space().approximate_size();
  }

  /// Stable per-array name (mirrors TiledArrayGenerator::tensor_var_name so
  /// CTIR names match the einsum's, enabling a line-by-line side-by-side).
  std::string var_name(const Tensor &tensor) const {
    std::string key = toUtf8(tensor.label());
    for (const Index &idx : tensor.const_indices()) {
      key += "_" + toUtf8(idx.space().base_key());
      if (idx.has_proto_indices())
        key += "p" + std::to_string(idx.proto_indices().size());
    }
    auto it = m_names.find(key);
    if (it != m_names.end()) return it->second;
    std::string name = sanitize(key);
    m_names.emplace(key, name);
    return name;
  }
  static std::string sanitize(std::string s) {
    if (s.empty()) return "_";
    for (char &c : s) {
      unsigned char uc = static_cast<unsigned char>(c);
      if (uc < 128 && !std::isalnum(uc) && c != '_') c = '_';
    }
    if (std::isdigit(static_cast<unsigned char>(s[0]))) s = "_" + s;
    return s;
  }

  /// Split a tensor's indices into outer (block-sparse), inner (per-pair PNO),
  /// and pair-key (the proto/occupied tags). Mirrors classify_indices: direct
  /// non-proto indices are outer (own order); proto-carrying indices are inner
  /// (proto stripped); proto tags with no direct counterpart are pair-keys,
  /// prepended.
  void split_indices(const Tensor &t, std::vector<Index> &outer,
                     std::vector<Index> &inner,
                     std::vector<Index> &pair_key) const {
    for (const Index &idx : t.const_indices()) {
      if (idx.has_proto_indices())
        inner.push_back(idx.drop_proto_indices());
      else if (!contains(outer, idx))
        outer.push_back(idx);
    }
    for (const Index &idx : t.const_indices()) {
      if (!idx.has_proto_indices()) continue;
      for (const Index &p : idx.proto_indices())
        if (!contains(outer, p) && !contains(pair_key, p)) pair_key.push_back(p);
    }
  }
  static bool contains(const std::vector<Index> &v, const Index &x) {
    for (const Index &e : v)
      if (e == x) return true;
    return false;
  }

  /// A tensor is volatile iff it is (transitively) amplitude-dependent: the
  /// amplitude leaf carries label "t" (matching MPQC's is_volatile predicate),
  /// or it is a previously-computed def that we already marked volatile.
  bool is_volatile_name(const Tensor &op, const std::string &oname) const {
    if (toUtf8(op.label()) == "t") return true;
    auto it = m_def_of.find(oname);
    if (it != m_def_of.end()) return m_defs[it->second].volatile_;
    return false;  // t-independent leaf (g, C, f, s)
  }

  void register_leaf(const Tensor &t, const Context &) {
    const std::string name = var_name(t);
    if (!m_leaf.insert(name).second) return;
    Leaf lf;
    lf.name = name;
    lf.label = toUtf8(t.label());
    lf.volatile_ = (lf.label == "t");
    std::vector<Index> o, i, p;
    split_indices(t, o, i, p);
    lf.is_tot = !i.empty();
    lf.outer = join(o);
    lf.inner = join(i);
    lf.pair = join(p);
    m_leaves.push_back(std::move(lf));
  }

  /// Flatten a (binarized) product/tensor into operand tensors + a scalar.
  void collect(const Expr &e, std::vector<Tensor> &ops, std::string &scalar,
               const Context &ctx) const {
    if (e.is<Tensor>()) {
      ops.push_back(e.as<Tensor>());
    } else if (e.is<Constant>()) {
      append_scalar(scalar, to_double(e.as<Constant>().value().real()));
    } else if (e.is<Product>()) {
      const Product &p = e.as<Product>();
      if (!p.scalar().is_identity())
        append_scalar(scalar, to_double(p.scalar().real()));
      for (std::size_t k = 0; k < p.size(); ++k)
        collect(*p.factor(k), ops, scalar, ctx);
    } else if (e.is<Sum>()) {
      const Sum &s = e.as<Sum>();
      for (std::size_t k = 0; k < s.size(); ++k)
        collect(*s.summand(k), ops, scalar, ctx);
    }
  }
  static void append_scalar(std::string &scalar, double v) {
    std::ostringstream oss;
    oss << v;
    scalar = scalar.empty() ? oss.str() : scalar + "*" + oss.str();
  }

  std::string join(const std::vector<Index> &v) const {
    std::string s;
    for (std::size_t i = 0; i < v.size(); ++i) {
      if (i) s += " ";
      s += toUtf8(v[i].label());
    }
    return s;
  }
  static std::string sci(double x) {
    std::ostringstream oss;
    if (x >= 1e5)
      oss.setf(std::ios::scientific), oss.precision(1), oss << x;
    else
      oss << static_cast<long long>(x);
    return oss.str();
  }

  void reset() {
    m_defs.clear();
    m_leaves.clear();
    m_order.clear();
    m_freed.clear();
    m_written.clear();
    m_leaf.clear();
    m_names.clear();
    m_uses.clear();
    m_def_of.clear();
    m_result_name.clear();
    m_fn.clear();
  }

  // ------------------------------------------------------------------ render
  std::string render() const {
    // A t-indep def whose value is consumed by a t-dep (volatile) def sits on
    // the persistent boundary: MPQC builds it once and reuses it across CC
    // iterations (cache_manager.hpp: NV-with-V-consumer -> P).
    std::map<std::string, bool> feeds_volatile;
    for (const Def &d : m_defs)
      if (d.volatile_)
        for (const std::string &op : d.operand_names) feeds_volatile[op] = true;

    std::ostringstream o;
    o << "ctir v1  " << (m_fn.empty() ? "residual" : m_fn)
      << "   (CSV-CCSD closed-shell, DF+CSV)\n";
    o << "; one shared DAG (post-CSE). `uses=N` = # consumers (cross-term "
         "sharing MPQC caches with min_repeats=2).\n";
    o << "; ToT tensors print as [outer ; inner]<pair-key>; inner = per-pair "
         "PNO domain.\n";
    o << "; cost: cells = # outer ToT cells (= # independent tasks); "
         "CELL-BOUND flags the DF-half-transform pathology.\n";
    o << "; t-indep = amplitude-independent (build-once candidate); "
         "persistent = t-indep feeding a t-dep op (MPQC reuses across iters).\n\n";

    o << "leaves:\n";
    for (const Leaf &lf : m_leaves) {
      o << "  " << lf.name << "  " << bracket(lf.outer, lf.inner, lf.pair, lf.is_tot)
        << (lf.is_tot ? "  tot" : "  flat")
        << (lf.volatile_ ? "  t-dep(amplitude)" : "  t-indep") << "\n";
    }
    o << "\ncomputation:\n";
    std::size_t free_cursor = 0;
    for (const Event &ev : m_order) {
      if (ev.kind == Event::Free) {
        if (free_cursor < m_freed.size())
          o << "    free " << m_freed[free_cursor++]
            << "        ; repro: release slot (static ref-count liveness)\n";
        continue;
      }
      const Def &d = m_defs[ev.idx];
      const std::string arrow = d.is_accumulation ? "+=" : "=";
      o << (d.writes_result ? "  result " : "  def ") << d.name << " "
        << bracket(join(d.outer), join(d.inner), join(d.pair_key), d.is_tot)
        << (d.is_tot ? " tot" : " flat") << "  uses=" << uses_of(d.name)
        << (d.volatile_ ? "  t-dep" : "  t-indep")
        << (!d.volatile_ && feeds_volatile.count(d.name)
                ? "  [persistent: build-once, reused across iters]"
                : "")
        << "\n";
      o << "      " << arrow << " " << (d.scalar_str.empty() ? "" : d.scalar_str + " ")
        << (d.contracted.empty() ? "copy " : "contract{" + join(d.contracted) + "} ");
      for (std::size_t k = 0; k < d.operand_names.size(); ++k) {
        if (k) o << " * ";
        o << d.operand_names[k];
      }
      o << "\n";
      o << "      cost: cells=" << sci(static_cast<double>(d.cells))
        << "  inner=" << d.inner_ext << "  flops=" << sci(d.flops)
        << "  per-cell=" << sci(d.per_cell);
      if (d.cell_bound) o << "   ⚠ CELL-BOUND (" << sci(static_cast<double>(d.cells))
                          << " tiny ToT tasks)";
      if (d.batch_aux) o << "   · batchable/" << d.batch_axis << " (aux)";
      o << "\n";
    }
    o << "\n; totals: " << m_defs.size() << " contraction steps, "
      << m_leaves.size() << " leaves.\n";
    return o.str();
  }

  std::size_t uses_of(const std::string &n) const {
    auto it = m_uses.find(n);
    return it == m_uses.end() ? 0 : it->second;
  }
  static std::string bracket(const std::string &outer, const std::string &inner,
                             const std::string &pair, bool is_tot) {
    std::string s = "[" + outer;
    if (is_tot) s += " ; " + inner;
    s += "]";
    if (!pair.empty()) s += "⟨" + pair + "⟩";
    return s;
  }

  // ------------------------------------------------------------------ state
  std::vector<Def> m_defs;
  std::vector<Leaf> m_leaves;
  std::vector<Event> m_order;
  std::vector<std::string> m_freed;
  std::set<std::string> m_written;
  std::set<std::string> m_leaf;
  mutable std::unordered_map<std::string, std::string> m_names;
  std::unordered_map<std::string, std::size_t> m_uses;
  std::unordered_map<std::string, std::size_t> m_def_of;
  std::string m_result_name;
  std::string m_fn;
};

}  // namespace sequant

#endif  // SEQUANT_CORE_EXPORT_CONTRACTION_IR_GENERATOR_HPP
