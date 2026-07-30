#ifndef SEQUANT_CORE_EXPORT_CONTRACTION_IR_GENERATOR_HPP
#define SEQUANT_CORE_EXPORT_CONTRACTION_IR_GENERATOR_HPP

// ContractionIRGenerator -- a human-readable "Contraction IR" (CTIR) backend.
//
// Motivation (see sequant-ta-repro/docs/CONTRACTION_IR.md): the TiledArray
// einsum export is an *execution artifact* -- a flattened, linear list of
// TA::einsum calls with implicit sharing and no cost/semantic annotation. It
// cannot express (a) the per-outer-cell cost that dominates the CSV-CCSD cold
// runtime (the giant (mu-tilde,K)-outer DF half-transform runs ~100x off peak
// purely from having millions of tiny per-pair ToT cells -- invisible in an
// einsum call), nor (b) what MPQC actually does at runtime: a DAG over a
// CacheManager with amplitude-independent (t-indep) intermediates built once
// and reused across CC iterations (persistent), plus optional aux-K batching.
//
// CTIR renders the whole-residual computation as a *value DAG*: each computed
// value is one `def` (cross-term sharing shows as `uses=N`), an accumulated
// value is `= Σ` of its contributions, and each is annotated with its ToT
// outer;inner split + per-pair domain, a cost line (cells = # outer ToT cells,
// with a CELL-BOUND flag on the dominant cell-count node), a t-indep/t-dep
// classification + persistent boundary, and aux-K batchability.
//
// DESIGN: the export framework hands us an imperative stream (declare / compute
// / unload) that REUSES C++ slot names across live ranges. We recover the value
// DAG in three phases:
//   collect  -- turn the stream into Values (a new value each time a slot is
//               (re)written after a free; += within a live range = one value
//               with multiple contributions);
//   analyze  -- whole-DAG passes for uses, a volatility fixpoint (t-dep iff it
//               or any operand transitively contracts a `t` leaf), last-use
//               liveness (the repro's static free points), and the persistent
//               boundary (t-indep feeding t-dep);
//   render   -- with a spaces/extents legend, per-value canonical index names,
//               a relative CELL-BOUND flag, and a summary.
// Descriptive only (no lowering); reuses the framework's traversal/CSE.

#include <SeQuant/core/export/context.hpp>
#include <SeQuant/core/export/generator.hpp>
#include <SeQuant/core/expr.hpp>
#include <SeQuant/core/index.hpp>
#include <SeQuant/core/utility/string.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cwctype>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace sequant {

/// Context for ContractionIRGenerator. `extent` supplies per-index sizes so the
/// IR can print cell counts / flops; set it from the same idx_to_extent the
/// optimizer used so the numbers match the cost model that chose this
/// factorization. Default falls back to the index space's approximate size.
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
    record_space_extents(tensor, ctx);
    if (usage == Usage::Terminal) register_leaf(tensor);
  }
  void declare(const Variable &, UsageSet, const Context &) override {}
  void declare(const Index &, const Context &) override {}

  // --- collect: one contraction step ---------------------------------------
  void compute(const Expr &expression, const Tensor &result,
               const Context &ctx) override {
    record_space_extents(result, ctx);
    const std::string slot = var_name(result);

    // A new value starts whenever the slot is not currently live (first write,
    // or reuse after a free); += into a live slot appends a contribution to the
    // existing value. This is what turns the imperative slot stream into a DAG.
    std::size_t vi;
    auto live_it = m_live.find(slot);
    if (live_it == m_live.end()) {
      vi = m_values.size();
      Value v;
      v.slot = slot;
      std::size_t &ver = m_version[slot];
      ver += 1;
      v.id = (ver == 1) ? slot : slot + "#" + std::to_string(ver);
      split_indices(result, v.outer, v.inner, v.pair_key);
      v.is_tot = !v.inner.empty();
      // cells = # outer ToT cells = product of the block-sparse outer axes:
      // the non-proto direct indices AND the proto-target pair keys (each
      // occupied pair is its own block; these are proto tags, not direct
      // indices, so count them explicitly).
      v.cells = 1;
      for (const Index &i : result.const_indices())
        if (!i.has_proto_indices()) v.cells *= ext(ctx, i);
      for (const Index &i : v.pair_key) v.cells *= ext(ctx, i);
      v.inner_ext = 1;  // ORIGINAL proto-carrying indices -> proto extent
      for (const Index &i : result.const_indices())
        if (i.has_proto_indices()) v.inner_ext *= ext(ctx, i);
      m_values.push_back(std::move(v));
      m_live[slot] = vi;
      m_last_value[slot] = vi;  // persists across unload (for operand resolution)
    } else {
      vi = live_it->second;
    }

    Contribution c;
    std::vector<Tensor> operands;
    collect(expression, operands, c.scalar, ctx);

    // Proto-aware index sets: const_indices() omits proto-tag (pair-key)
    // indices, so expand proto_indices() too -- otherwise a summed pair key is
    // dropped and a surviving pair key is falsely reported contracted.
    std::set<std::wstring> result_idx = index_labels_with_protos(result);
    std::set<std::wstring> seen;
    for (const Tensor &op : operands) {
      const std::string oname = var_name(op);
      const bool op_is_leaf = is_leaf_tensor(op, oname);
      if (op_is_leaf) register_leaf(op);  // lazy (declare() can skip variants)
      // Resolve to the last value written to this slot (NOT the currently-live
      // one): the framework frees a shared CSE node per-tree, so a later tree's
      // reference must still bind to that value, or its cross-term edges (and
      // uses count) are lost.
      c.operands.push_back({oname, op_is_leaf ? SIZE_MAX : last_value(oname)});
      if (op_is_leaf && toUtf8(op.label()) == "t") c.intrinsic_volatile = true;
      // An index is contracted (summed) if it is on an operand but not on the
      // result. Consider proto-tag (pair-key) indices too, not just direct
      // ones: a pair key summed BETWEEN two ToT operands appears only as a
      // proto tag and would otherwise be dropped (result_idx already includes
      // protos, so surviving pair keys are still correctly excluded).
      auto consider_contracted = [&](const Index &i) {
        record_space_extents_one(i, ctx);
        const std::wstring lbl(i.label());
        if (result_idx.count(lbl) || !seen.insert(lbl).second) return;
        c.contracted.push_back(i);
        if (space_key(i) == L"Κ") {
          c.batch_aux = true;
          c.batch_axis = toUtf8(i.label());
        }
      };
      for (const Index &i : op.const_indices()) {
        consider_contracted(i);
        for (const Index &p : i.proto_indices()) consider_contracted(p);
      }
    }
    // per-contribution flops ~= product over distinct index labels in the
    // contraction (result outer/inner + pair keys + everything in operands,
    // including operands' proto (pair-key) tags).
    double flops = 1.0;
    std::set<std::wstring> counted;
    auto mul = [&](const Index &i) {
      if (counted.insert(std::wstring(i.label())).second)
        flops *= static_cast<double>(ext(ctx, i));
    };
    for (const Index &i : result.const_indices()) mul(i);
    for (const Index &i : m_values[vi].pair_key) mul(i);
    for (const Tensor &op : operands)
      for (const Index &i : op.const_indices()) {
        mul(i);
        for (const Index &p : i.proto_indices()) mul(p);
      }
    c.flops = flops;
    m_values[vi].contribs.push_back(std::move(c));
  }
  void compute(const Expr &, const Variable &, const Context &) override {}

  // --- end a value's live range (slot reuse boundary) ----------------------
  void unload(const Tensor &tensor, const Context &) override {
    m_live.erase(var_name(tensor));
  }
  void destroy(const Tensor &tensor, const Context &ctx) override {
    unload(tensor, ctx);
  }
  void persist(const Tensor &tensor, const Context &) override {
    // persist() fires once per tree root; the residual is exported as a forest
    // of many trees (hoisted-CSE roots + residual-summand roots), so record ALL
    // persisted slots rather than last-write-wins -- the true residual is
    // identified in analyze() as the persisted, still-live, unconsumed value.
    m_result_slots.insert(var_name(tensor));
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
  // analyze() mutates the collected DAG, so it runs here (end_export is
  // non-const and fires before the driver calls get_generated_code()).
  void end_export(const Context &) override { analyze(); }

  std::string get_generated_code() const override { return render(); }

 private:
  // ------------------------------------------------------------------ types
  struct OperandRef {
    std::string name;
    std::size_t value;  // index into m_values, or SIZE_MAX for a leaf
  };
  struct Contribution {
    std::string scalar, batch_axis;
    std::vector<OperandRef> operands;
    std::vector<Index> contracted;
    bool intrinsic_volatile = false, batch_aux = false;
    double flops = 0;
  };
  struct Value {
    std::string id, slot;
    std::vector<Index> outer, inner, pair_key;
    std::vector<Contribution> contribs;
    bool is_tot = false;
    std::size_t cells = 0, inner_ext = 0;
    // analyzed:
    bool volatile_ = false, persistent = false, is_result = false,
         cell_bound = false, batch_aux = false;
    std::size_t uses = 0, last_use = 0;  // last_use = index of latest consumer
    bool freed = false;
  };
  struct Leaf {
    std::string name, label;
    std::vector<Index> outer, inner, pair_key;
    bool is_tot = false, volatile_ = false;
  };
  struct Canon;  // per-value index namer (defined below; used in signatures)

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
  void record_space_extents_one(const Index &i, const Context &ctx) {
    m_space_ext.emplace(space_key(i), ext(ctx, i));
    // A proto-carrying index's OWN space is the per-pair PNO space -- record
    // its key so the legend lists it once as a per-pair domain, not twice.
    if (i.has_proto_indices()) m_pno_keys.insert(space_key(i));
  }
  void record_space_extents(const Tensor &t, const Context &ctx) {
    for (const Index &i : t.const_indices()) record_space_extents_one(i, ctx);
  }

  /// Stable per-array name (mirrors TiledArrayGenerator::tensor_var_name so
  /// CTIR names line up with the einsum's -- EXCEPT tensors the einsum touches
  /// with its text-level fix_rank_collision (RANKFIX) / fix_cse_axis_order
  /// passes, which have no CTIR counterpart).
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
  /// Synthesized intermediates are labeled "I"/"I2".../"CSEn" (see
  /// eval_expr.cpp label_tensor + the CSE replacer); everything else (g, C, t,
  /// f, s) is a genuine input leaf.
  static bool is_leaf_tensor(const Tensor &t, const std::string &) {
    std::wstring_view l = t.label();
    if (l.rfind(L"CSE", 0) == 0) return false;
    std::size_t n = l.size();
    while (n > 0 && std::iswdigit(l[n - 1])) --n;
    return l.substr(0, n) != L"I";
  }
  std::size_t last_value(const std::string &slot) const {
    auto it = m_last_value.find(slot);
    return it == m_last_value.end() ? SIZE_MAX : it->second;
  }

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
  /// All index labels of a tensor INCLUDING proto (pair-key) tags -- so summed
  /// vs surviving pair keys are classified correctly (const_indices() alone
  /// omits proto tags).
  static std::set<std::wstring> index_labels_with_protos(const Tensor &t) {
    std::set<std::wstring> s;
    for (const Index &i : t.const_indices()) {
      s.insert(std::wstring(i.label()));
      for (const Index &p : i.proto_indices()) s.insert(std::wstring(p.label()));
    }
    return s;
  }

  void register_leaf(const Tensor &t) {
    const std::string name = var_name(t);
    if (!m_leaf_names.insert(name).second) return;
    Leaf lf;
    lf.name = name;
    lf.label = toUtf8(t.label());
    lf.volatile_ = (lf.label == "t");
    split_indices(t, lf.outer, lf.inner, lf.pair_key);
    lf.is_tot = !lf.inner.empty();
    m_leaves.push_back(std::move(lf));
  }

  /// Flatten a (binarized) product/tensor into operand tensors + a scalar.
  void collect(const Expr &e, std::vector<Tensor> &ops, std::string &scalar,
               const Context &ctx) const {
    if (e.is<Tensor>()) {
      ops.push_back(e.as<Tensor>());
    } else if (e.is<Constant>()) {
      append_scalar(scalar, to_double(e.as<Constant>().value().real()));
    } else if (e.is<Variable>()) {
      // A symbolic scalar coefficient (prunable_scalars()==All permits the
      // driver to fold Variables into the product). CC residual prefactors are
      // numeric, so this is rarely hit, but fold the variable's name into the
      // scalar rather than falling through to the throw below.
      const std::string nm = toUtf8(e.as<Variable>().label());
      scalar = scalar.empty() ? nm : scalar + "*" + nm;
    } else if (e.is<Product>()) {
      const Product &p = e.as<Product>();
      if (!p.scalar().is_identity())
        append_scalar(scalar, to_double(p.scalar().real()));
      for (std::size_t k = 0; k < p.size(); ++k)
        collect(*p.factor(k), ops, scalar, ctx);
    } else {
      // Binarization guarantees compute() sees only a leaf or a binary Product;
      // a Sum here would mean an un-binarized tree -- fail loudly rather than
      // silently concatenate summands as if they were factors.
      throw Exception(
          "ContractionIRGenerator: expected a Tensor/Product in compute(), got "
          "a non-binarized expression");
    }
  }
  static void append_scalar(std::string &scalar, double v) {
    std::ostringstream oss;
    oss << v;
    scalar = scalar.empty() ? oss.str() : scalar + "*" + oss.str();
  }

  // ------------------------------------------------------------------ analyze
  void analyze() {
    // (a) uses + last-use over the whole DAG (order-independent).
    for (std::size_t vi = 0; vi < m_values.size(); ++vi)
      for (const Contribution &c : m_values[vi].contribs)
        for (const OperandRef &op : c.operands)
          if (op.value != SIZE_MAX && op.value < m_values.size()) {
            m_values[op.value].uses += 1;
            m_values[op.value].last_use =
                std::max(m_values[op.value].last_use, vi);
          }
    // (b) volatility fixpoint: t-dep iff any contribution is intrinsically
    // volatile (contracts a `t` leaf) or any operand value is t-dep. Iterating
    // to a fixpoint makes this independent of emission/tree order.
    bool changed = true;
    while (changed) {
      changed = false;
      for (Value &v : m_values) {
        if (v.volatile_) continue;
        bool vol = false;
        for (const Contribution &c : v.contribs) {
          if (c.intrinsic_volatile) vol = true;
          for (const OperandRef &op : c.operands)
            if (op.value != SIZE_MAX && op.value < m_values.size() &&
                m_values[op.value].volatile_)
              vol = true;
        }
        if (vol) v.volatile_ = changed = true;
      }
    }
    // (c) persistent boundary: a t-indep value consumed by a t-dep value is
    // what MPQC builds once and keeps across iterations (cache_manager.hpp:
    // NV-with-V-consumer -> P).
    for (const Value &v : m_values)
      if (v.volatile_)
        for (const Contribution &c : v.contribs)
          for (const OperandRef &op : c.operands)
            if (op.value != SIZE_MAX && op.value < m_values.size() &&
                !m_values[op.value].volatile_)
              m_values[op.value].persistent = true;
    // (d) per-value rollups: batchable, flops-sum, result, freed (repro
    // liveness = has a last consumer and is not the persisted result).
    double max_cells = 0;
    for (std::size_t vi = 0; vi < m_values.size(); ++vi) {
      Value &v = m_values[vi];
      for (const Contribution &c : v.contribs)
        if (c.batch_aux) v.batch_aux = true;
      // the residual is the value whose slot was persisted, is still the live
      // value for that slot at end (never freed), and is not consumed by any
      // other value (uses==0). This is order-independent -- unlike keying on a
      // last-write-wins persisted slot -- and excludes hoisted CSE roots (which
      // are persisted but have uses>=1). `uses` was filled in pass (a) above.
      auto lit = m_live.find(v.slot);
      v.is_result = (m_result_slots.count(v.slot) && lit != m_live.end() &&
                     lit->second == vi && v.uses == 0);
      // max over ToT values only: CELL-BOUND is about the per-pair-cell
      // tile-task overhead specific to tensor-of-tensor contractions, so a
      // large FLAT intermediate must not inflate the threshold.
      if (v.is_tot) max_cells = std::max(max_cells, static_cast<double>(v.cells));
    }
    // (e) relative CELL-BOUND: flag the dominant cell-count ToT node(s) --
    // travels across problem sizes, unlike an absolute threshold.
    for (Value &v : m_values)
      v.cell_bound = v.is_tot && max_cells > 0 &&
                     static_cast<double>(v.cells) >= 0.5 * max_cells &&
                     v.cells >= 1000;
    // freed: any value with a consumer, that isn't the final result, is
    // released by the repro after its last use.
    for (Value &v : m_values)
      v.freed = (v.uses > 0) && !v.is_result;
  }

  // ------------------------------------------------------------------ render
  std::string render() const {
    std::ostringstream o;
    o << "ctir v1  " << (m_fn.empty() ? "residual" : m_fn)
      << "   (CSV-CCSD closed-shell, DF+CSV)\n";
    o << "; value DAG: each `def` is one computed value; `uses=N` = # consumers "
         "(cross-term sharing MPQC caches, min_repeats=2).\n";
    o << "; ToT prints [outer ; inner]⟨pair-key⟩; inner = per-pair PNO domain. "
         "cells = # outer ToT cells (= # tasks).\n";
    o << "; t-indep = amplitude-independent (built once = 'cold precompute'); "
         "t-dep = rebuilt every CC iteration ('warm').\n";
    o << "; persistent = a t-indep value feeding a t-dep one -- MPQC builds it "
         "once and reuses it across iterations; the repro rebuilds it every "
         "pass.\n";
    // spaces legend: global (block-sparse) spaces first, then the per-pair PNO
    // space(s) noted separately as per-pair domains.
    o << "\nspaces:";
    for (const auto &[k, e] : m_space_ext)
      if (!m_pno_keys.count(k)) o << "  " << toUtf8(k) << "=" << e;
    for (const std::wstring &k : m_pno_keys)
      o << "  " << toUtf8(k) << "=PNO⟨per-pair⟩~" << m_space_ext.at(k);
    o << "\n";

    o << "\nleaves:\n";
    for (const Leaf &lf : m_leaves) {
      Canon cn;
      o << "  " << lf.name << "  "
        << bracket(lf.outer, lf.inner, lf.pair_key, lf.is_tot, cn)
        << (lf.is_tot ? "  tot" : "  flat")
        << (lf.volatile_ ? "  t-dep(amplitude)" : "  t-indep") << "\n";
    }

    o << "\ncomputation:\n";
    for (std::size_t vi = 0; vi < m_values.size(); ++vi) {
      const Value &v = m_values[vi];
      Canon cn;  // per-value canonical index names
      o << (v.is_result ? "  result " : "  def ") << v.id << " "
        << bracket(v.outer, v.inner, v.pair_key, v.is_tot, cn)
        << (v.is_tot ? " tot" : " flat") << "  uses=" << v.uses
        << (v.volatile_ ? "  t-dep" : "  t-indep")
        << (v.persistent ? "  [persistent: built once, reused across iters]"
                         : "")
        << "\n";
      if (v.contribs.size() == 1) {
        o << "      = " << render_contrib(v.contribs[0], cn) << "\n";
      } else {
        o << "      = Σ " << v.contribs.size() << " contributions:\n";
        for (const Contribution &c : v.contribs)
          o << "          " << render_contrib(c, cn) << "\n";
      }
      // For a Σ (multi-contribution) value, flops is summed over contributions
      // while cells is the single shared result shape, so per-cell here is an
      // aggregate ratio (total work / result cells), not one contraction's cost.
      double tot_flops = 0;
      for (const Contribution &c : v.contribs) tot_flops += c.flops;
      o << "      cost: cells=" << sci(static_cast<double>(v.cells))
        << "  inner=" << v.inner_ext << "  flops=" << sci(tot_flops)
        << "  per-cell=" << sci(v.cells ? tot_flops / v.cells : tot_flops);
      if (v.cell_bound)
        o << "   ⚠ CELL-BOUND (" << sci(static_cast<double>(v.cells))
          << " tiny ToT tasks)";
      if (v.batch_aux) o << "   · batchable/Κ (aux)";
      o << "\n";
      // repro static free points (whole-DAG last-use), suppressing the final
      // result: shown to contrast with MPQC's cross-iteration cache lifetime.
      for (std::size_t oj = 0; oj < m_values.size(); ++oj)
        if (m_values[oj].freed && m_values[oj].last_use == vi &&
            m_values[oj].uses > 0)
          o << "    free " << m_values[oj].id
            << "        ; repro: release slot after last use (static liveness)\n";
    }

    render_summary(o);
    return o.str();
  }

  std::string render_contrib(const Contribution &c, Canon &cn) const {
    std::ostringstream s;
    if (!c.scalar.empty()) s << c.scalar << " ";
    if (c.contracted.empty())
      s << "copy ";
    else {
      s << "contract{";
      for (std::size_t k = 0; k < c.contracted.size(); ++k) {
        if (k) s << " ";
        s << cn.name(c.contracted[k]);
      }
      s << "} ";
    }
    for (std::size_t k = 0; k < c.operands.size(); ++k) {
      if (k) s << " * ";
      s << c.operands[k].name;
    }
    return s.str();
  }

  void render_summary(std::ostringstream &o) const {
    std::size_t t_indep = 0, t_dep = 0, persistent = 0, batchable = 0,
                cell_bound = 0;
    double flops_indep = 0, flops_total = 0;
    for (const Value &v : m_values) {
      double f = 0;
      for (const Contribution &c : v.contribs) f += c.flops;
      flops_total += f;
      if (v.volatile_)
        ++t_dep;
      else {
        ++t_indep;
        flops_indep += f;
      }
      if (v.persistent) ++persistent;
      if (v.batch_aux) ++batchable;
      if (v.cell_bound) ++cell_bound;
    }
    o << "\nsummary:\n";
    o << "  " << m_values.size() << " values (" << t_indep
      << " t-indep / " << t_dep << " t-dep), " << m_leaves.size()
      << " input leaves.\n";
    o << "  t-indep (cold precompute) ≈ " << sci(flops_indep) << " flops of "
      << sci(flops_total) << " total ("
      << (flops_total > 0 ? static_cast<int>(100 * flops_indep / flops_total)
                          : 0)
      << "%): MPQC builds these once (";
    o << persistent << " persistent); the repro's cold driver rebuilds them "
         "every residual pass.\n";
    o << "  " << cell_bound << " CELL-BOUND node(s) (the DF half-transform "
         "tiny-cell hotspot); " << batchable << " aux-Κ-batchable.\n";
  }

  // Per-value canonical index namer: strips SeQuant's internal numeric tags and
  // disambiguates same-space indices with primes (occ -> i,j,k,...), so a node
  // reads as e.g. `[μ̃ Κ ; a]⟨i j⟩ = contract{μ̃'} …` instead of `μ̃_19906` noise.
  struct Canon {
    std::map<std::wstring, std::string> assigned;  // full index label -> name
    std::map<std::wstring, int> per_space;         // base key -> count
    static const char *occ_names(int n) {
      static const char *o[] = {"i", "j", "k", "l", "m", "n"};
      return (n >= 0 && n < 6) ? o[n] : nullptr;
    }
    std::string name(const Index &idx) {
      const std::wstring full(idx.label());
      auto it = assigned.find(full);
      if (it != assigned.end()) return it->second;
      const std::wstring sk(idx.space().base_key());
      const std::string base = toUtf8(sk);
      int n = per_space[sk]++;
      std::string nm;
      if (base == "i") {
        const char *on = occ_names(n);
        nm = on ? on : base + std::to_string(n);
      } else {
        nm = base;
        for (int p = 0; p < n; ++p) nm += "'";
      }
      assigned.emplace(full, nm);
      return nm;
    }
  };

  std::string bracket(const std::vector<Index> &outer,
                      const std::vector<Index> &inner,
                      const std::vector<Index> &pair, bool is_tot,
                      Canon &cn) const {
    auto join = [&](const std::vector<Index> &v) {
      std::string s;
      for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) s += " ";
        s += cn.name(v[i]);
      }
      return s;
    };
    std::string s = "[" + join(outer);
    if (is_tot) s += " ; " + join(inner);
    s += "]";
    if (!pair.empty()) s += "⟨" + join(pair) + "⟩";
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
    m_values.clear();
    m_leaves.clear();
    m_leaf_names.clear();
    m_live.clear();
    m_last_value.clear();
    m_version.clear();
    m_names.clear();
    m_space_ext.clear();
    m_pno_keys.clear();
    m_result_slots.clear();
    m_fn.clear();
  }

  // ------------------------------------------------------------------ state
  std::vector<Value> m_values;
  std::vector<Leaf> m_leaves;
  std::set<std::string> m_leaf_names;
  std::unordered_map<std::string, std::size_t> m_live;    // slot -> value index
  std::unordered_map<std::string, std::size_t> m_last_value;  // slot -> last value
  std::unordered_map<std::string, std::size_t> m_version;  // slot -> # values
  mutable std::unordered_map<std::string, std::string> m_names;
  std::map<std::wstring, std::size_t> m_space_ext;
  std::set<std::wstring> m_pno_keys;
  std::set<std::string> m_result_slots;
  std::string m_fn;
};

}  // namespace sequant

#endif  // SEQUANT_CORE_EXPORT_CONTRACTION_IR_GENERATOR_HPP
