#ifndef SEQUANT_CORE_EXPORT_TILEDARRAY_GENERATOR_HPP
#define SEQUANT_CORE_EXPORT_TILEDARRAY_GENERATOR_HPP

// Native SeQuant -> TiledArray/C++ code generator. Modeled on
// PythonEinsumGeneratorBase (python_einsum.hpp) -- the closest existing
// analog, since TA::einsum is itself an einsum-shaped binary-contraction
// call -- but adapted for two things Python doesn't need:
//   1. TA::einsum is BINARY ONLY (two tensor operands, one annotation
//      string for the result). GenerationVisitor::process_computation
//      (export.hpp) already hands compute() an expression tree that is
//      binarized by construction (a Product node's factors, after
//      re-multiplying any pruned scalar prefactor back in, are at most
//      one scalar term plus AT MOST TWO tensor factors) -- so no
//      additional N-ary-to-binary decomposition is needed here, unlike
//      NumPy/PyTorch's einsum which natively supports N-ary contractions
//      in one call.
//   2. C++ is statically typed: unlike Python's dynamically-typed
//      variables, every intermediate/result tensor needs an explicit
//      `TA::TSpArrayD name;` declaration before first use. declare() (not
//      create()/load()) is where this happens, since declare() fires once
//      per unique tensor, before any create/load/compute calls target it
//      (see export.hpp's export_groups: declarations happen in a
//      dedicated pre-pass).
//
// Leaf-tensor binding convention (v1, flat-only): every tensor whose ONLY
// usage is Usage::Terminal (a pure leaf, never separately computed)
// becomes a `const TA::TSpArrayD&` FUNCTION PARAMETER, named after its
// label + index-space signature (so e.g. two different blocks of the
// same conceptual tensor, like an occ-occ vs. occ-virt slice, don't
// collide). This mirrors the pattern gen_ta_trace_equations.py already
// established informally via its TATensors struct fields, but is
// implemented generically here (SeQuant has no notion of MPQC's specific
// leaf tables) via each tensor's own index-space signature. There is no
// ToT (tensor-of-tensor / PNO-restriction) support in this v1 -- that is
// an MPQC-specific optimization layered on top by gen_ta_trace_equations.py,
// not a concept SeQuant's generic Product/Sum/Tensor expression tree
// carries; every tensor here is emitted as a plain, dense TA::TSpArrayD.
//
// Real-valued only: represent(Constant) throws on nonzero imaginary part,
// and the Adjoint-derived transpose (EvalOp::Adjoint, see export.hpp's own
// documented limitation) omits complex conjugation -- both match the
// existing, explicitly-documented limitation of the export IR itself
// (see export.hpp's EvalOp::Adjoint comment).

#include <SeQuant/core/export/context.hpp>
#include <SeQuant/core/export/generator.hpp>
#include <SeQuant/core/expr.hpp>
#include <SeQuant/core/index.hpp>
#include <SeQuant/core/rational.hpp>
#include <SeQuant/core/space.hpp>
#include <SeQuant/core/utility/string.hpp>

#include <cctype>
#include <functional>
#include <iomanip>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace sequant {

/// Context for the TiledArrayGenerator. No extra state needed for v1 --
/// index/tensor naming is entirely derived from each Index/Tensor's own
/// label and index-space signature, not from any externally-configured
/// shape/tag map (contrast PythonEinsumGeneratorContext, which needs a
/// shape map because NumPy arrays don't carry a TiledRange with them the
/// way a TA::TSpArrayD does at runtime).
class TiledArrayGeneratorContext : public ExportContext {
 public:
  using ExportContext::ExportContext;
};

/// Generates a single self-contained C++ function per exported expression,
/// computing it via TA::einsum. Flat (TA::TSpArrayD) tensors only.
class TiledArrayGenerator : public Generator<TiledArrayGeneratorContext> {
 public:
  TiledArrayGenerator() = default;
  ~TiledArrayGenerator() override = default;

  std::string get_format_name() const override { return "TiledArray (C++)"; }

  bool supports_named_sections() const override { return true; }
  bool requires_named_sections() const override { return true; }

  DeclarationScope index_declaration_scope() const override {
    return DeclarationScope::Expression;
  }
  DeclarationScope variable_declaration_scope() const override {
    return DeclarationScope::Expression;
  }
  DeclarationScope tensor_declaration_scope() const override {
    return DeclarationScope::Expression;
  }

  PrunableScalars prunable_scalars() const override {
    return PrunableScalars::All;
  }

  std::string represent(const Index &idx, const Context &) const override {
    // Plain label() (NOT full_label()): full_label() embeds the
    // proto-index list as "<...>" text for proto-indexed (PNO/CSV-
    // restricted) indices, which would mangle into a meaningless token
    // after sanitize_identifier. The restriction relationship is instead
    // conveyed structurally by index_annotation()'s outer;inner split (see
    // classify_indices()) -- represent() only ever needs the bare index
    // identity here.
    return sanitize_identifier(toUtf8(idx.label()));
  }

  std::string represent(const Tensor &tensor,
                        const Context &) const override {
    const std::string name = tensor_var_name(tensor);
    // The export framework's own declaration pass deduplicates tensors via
    // TensorBlockLessThanComparator (core/utility/tensor.hpp), which
    // compares (label, num_slots, num_indices, per-slot IndexSpace) --
    // deliberately NOT proto-indices, since most backends don't need that
    // distinction. But two ToT leaves sharing a label AND per-slot space
    // (e.g. a virtual index restricted to a single-occupied-index PNO
    // domain vs. an occupied-PAIR PNO domain -- osv vs. pno-proper, both
    // just "the virtual space" as far as IndexSpace equality is concerned)
    // are physically DIFFERENT arrays; tensor_var_name()'s proto-rank
    // tagging (see its own comment) distinguishes them, but that means the
    // framework's declare() pass calls back for only ONE representative
    // per block-equivalence-class, silently skipping the other variant(s)
    // -- confirmed empirically: a real CSV-CCSD T1 residual referenced a
    // "C" variant in compute() whose declare() call never fired. Lazily
    // declaring here on first reference is the fix: declare() already
    // guards on m_declared_names, so this is a no-op whenever declare()
    // legitimately got there first. Defaults newly-discovered tensors to
    // Terminal (a function parameter) -- every case seen so far (C, t) is
    // always a genuine external leaf, never a separately-computed
    // intermediate; revisit if a lazily-declared tensor ever turns out to
    // need Usage::Intermediate semantics instead.
    if (m_declared_names.insert(name).second) {
      const std::string type = tensor_cpp_type(tensor);
      m_params.emplace_back("const " + type + "&", name);
      m_leaf_names.insert(name);
    }
    return name;
  }

  std::string represent(const Variable &variable,
                        const Context &) const override {
    return sanitize_identifier(toUtf8(variable.label()));
  }

  std::string represent(const Constant &constant,
                        const Context &) const override {
    if (constant.value().imag() != 0) {
      throw Exception(
          "TiledArrayGenerator: complex constants are not supported "
          "(real-valued backend only)");
    }
    std::ostringstream oss;
    oss << std::setprecision(17) << to_double(constant.value().real());
    return "(" + oss.str() + ")";
  }

  std::string represent(const Power &power, const Context &ctx) const override {
    const ExprPtr &base = power.base();
    if (!base->is<Constant>()) {
      throw Exception(
          "TiledArrayGenerator: only Power(Constant, n) is supported");
    }
    double base_val = to_double(base->as<Constant>().value().real());
    double exponent = to_double(power.exponent());
    std::ostringstream oss;
    oss << std::setprecision(17) << "std::pow(" << base_val << ", " << exponent
        << ")";
    return oss.str();
  }

  std::string wrap_conj(std::string s) const override {
    // Real-valued backend: conjugation is the identity (see file header).
    return s;
  }

  void create(const Tensor &, bool, const Context &) override {
    // Declaration already emitted in declare(); nothing further to do --
    // a default-constructed TA::TSpArrayD is a valid "not yet assigned"
    // placeholder until the first compute() gives it a real value via `=`.
  }
  void load(const Tensor &, bool, const Context &) override {
    // Leaves ARE the function parameters (see file header) -- nothing to
    // emit, the parameter name is already directly usable. set_to_zero is
    // ignored: a genuine external leaf's caller is responsible for
    // providing the correct data (whether that's zero or not); there is
    // no const-correct way to zero a `const TA::TSpArrayD&` parameter
    // in-place anyway.
  }
  void set_to_zero(const Tensor &tensor, const Context &ctx) override {
    const std::string type = tensor_cpp_type(tensor);
    m_body += m_indent + represent(tensor, ctx) + " = " + type + "();\n";
  }
  void unload(const Tensor &tensor, const Context &ctx) override {
    const std::string name = represent(tensor, ctx);
    if (m_leaf_names.count(name)) return;  // never release a parameter
    const std::string type = tensor_cpp_type(tensor);
    m_body += m_indent + name + " = " + type + "();  // release\n";
  }
  void destroy(const Tensor &tensor, const Context &ctx) override {
    unload(tensor, ctx);
  }
  void persist(const Tensor &tensor, const Context &ctx) override {
    m_result_name = represent(tensor, ctx);
    m_result_is_tensor = true;
    m_result_is_tot = is_tot_tensor(tensor);
  }

  void create(const Variable &, bool, const Context &) override {}
  void load(const Variable &, bool, const Context &) override {
    // See the Tensor overload's comment -- set_to_zero is ignored for the
    // same reason.
  }
  void set_to_zero(const Variable &variable, const Context &ctx) override {
    m_body += m_indent + represent(variable, ctx) + " = 0.0;\n";
  }
  void unload(const Variable &, const Context &) override {}
  void destroy(const Variable &, const Context &) override {}
  void persist(const Variable &variable, const Context &ctx) override {
    m_result_name = represent(variable, ctx);
    m_result_is_tensor = false;
  }

  void compute(const Expr &expression, const Tensor &result,
              const Context &ctx) override {
    const std::string name = represent(result, ctx);
    const std::string ann = index_annotation(result, ctx);
    const std::string rhs = compute_rhs(expression, ann, ctx);
    const bool first_write = m_written.insert(name).second;
    m_body += m_indent + name + "(\"" + ann + "\") " +
              (first_write ? "=" : "+=") + " " + rhs + ";\n";
  }
  void compute(const Expr &expression, const Variable &result,
              const Context &ctx) override {
    const std::string name = represent(result, ctx);
    const bool first_write = m_written.insert(name).second;
    m_body += m_indent + name + (first_write ? " = " : " += ") +
              compute_scalar_rhs(expression, ctx) + ";\n";
  }

  void declare(const Index &, const Context &) override {
    // TA index labels are just annotation-string tokens, not separately
    // declared C++ entities.
  }
  void declare(const Variable &variable, UsageSet usage,
              const Context &) override {
    const std::string name = sanitize_identifier(toUtf8(variable.label()));
    if (!m_declared_names.insert(name).second) return;
    if (usage == Usage::Terminal) {
      m_params.emplace_back("double", name);
      m_leaf_names.insert(name);
    } else {
      m_local_decls += m_indent + "double " + name + " = 0.0;\n";
    }
  }
  void declare(const Tensor &tensor, UsageSet usage, const Context &) override {
    const std::string name = tensor_var_name(tensor);
    if (!m_declared_names.insert(name).second) return;
    const std::string type = tensor_cpp_type(tensor);
    if (usage == Usage::Terminal) {
      m_params.emplace_back("const " + type + "&", name);
      m_leaf_names.insert(name);
    } else {
      m_local_decls += m_indent + type + " " + name + ";\n";
    }
  }

  void all_indices_declared(std::size_t, const Context &) override {}
  void all_variables_declared(std::size_t, const Context &) override {}
  void all_tensors_declared(std::size_t, const Context &) override {}
  void begin_declarations(DeclarationScope, const Context &) override {}
  void end_declarations(DeclarationScope, const Context &) override {}

  void insert_comment(const std::string &comment, const Context &) override {
    m_body += m_indent + "// " + comment + "\n";
  }

  void begin_named_section(std::string_view name, const Context &) override {
    m_current_name = sanitize_identifier(std::string(name));
    m_body.clear();
    m_local_decls.clear();
    m_params.clear();
    m_declared_names.clear();
    m_tensor_names.clear();
    m_written.clear();
    m_leaf_names.clear();
    m_result_name.clear();
    m_result_is_tensor = true;
    m_result_is_tot = false;
  }
  void end_named_section(std::string_view, const Context &) override {
    m_generated += render_function() + "\n";
  }

  void begin_expression(const Context &) override {}
  void end_expression(const Context &) override {}

  void begin_export(const Context &) override { m_generated.clear(); }
  void end_export(const Context &) override {}

  std::string get_generated_code() const override { return m_generated; }

 private:
  std::string m_generated;
  std::string m_body;
  std::string m_local_decls;
  std::string m_indent = "  ";
  std::string m_current_name;
  std::string m_result_name;
  bool m_result_is_tensor = true;
  bool m_result_is_tot = false;
  mutable std::vector<std::pair<std::string, std::string>> m_params;
  mutable std::set<std::string> m_declared_names;
  mutable std::set<std::string> m_leaf_names;
  std::set<std::string> m_written;
  mutable std::unordered_map<std::string, std::string> m_tensor_names;

  static double to_double(const sequant::rational &r) {
    return static_cast<double>(numerator(r)) /
           static_cast<double>(denominator(r));
  }

  static std::string sanitize_identifier(std::string s) {
    if (s.empty()) return "_";
    for (char &c : s) {
      unsigned char uc = static_cast<unsigned char>(c);
      if (uc < 128 && !std::isalnum(uc) && c != '_') c = '_';
    }
    if (std::isdigit(static_cast<unsigned char>(s[0]))) s = "_" + s;
    return s;
  }

  /// Distinguishes different "blocks" of a tensor sharing the same label
  /// (e.g. an occ-occ vs. an occ-virtual slice of a conceptually single
  /// tensor) by appending each index's space to the label -- same idea as
  /// PythonEinsumGeneratorBase::tensor_name's index-space tagging. Also
  /// tags each proto-indexed axis with its proto-index count ("p1"/"p2")
  /// so e.g. a rank-1-PNO-restricted C and a rank-2-PNO-restricted C don't
  /// collide on the same generated variable/parameter name.
  std::string tensor_var_name(const Tensor &tensor) const {
    std::string key = toUtf8(tensor.label());
    for (const Index &idx : tensor.const_indices()) {
      key += "_" + toUtf8(idx.space().base_key());
      if (idx.has_proto_indices())
        key += "p" + std::to_string(idx.proto_indices().size());
    }
    auto it = m_tensor_names.find(key);
    if (it != m_tensor_names.end()) return it->second;
    std::string name = sanitize_identifier(key);
    m_tensor_names.emplace(key, name);
    return name;
  }

  /// Outer (free/contractable, no proto-indices of their own -- plus any
  /// proto-index a ToT axis is restricted by) vs. inner (the per-proto-
  /// group PNO/CSV axis itself, proto-info stripped via drop_proto_indices)
  /// index classification for a single Tensor. This is SeQuant's native
  /// equivalent of Phase 1's DSL <i1,i2> pairarg bracket classification --
  /// no external label table needed, since has_proto_indices()/
  /// proto_indices() already carry the full restriction structure on every
  /// Index natively.
  struct IndexClass {
    std::vector<Index> outer;
    std::vector<Index> inner;
    bool is_tot = false;
  };

  static bool contains_index(const std::vector<Index> &v, const Index &idx) {
    for (const Index &x : v)
      if (x == idx) return true;
    return false;
  }

  IndexClass classify_indices(const Tensor &tensor) const {
    IndexClass result;
    for (const Index &idx : tensor.const_indices()) {
      if (!idx.has_proto_indices()) {
        if (!contains_index(result.outer, idx)) result.outer.push_back(idx);
      } else {
        for (const Index &proto : idx.proto_indices()) {
          if (!contains_index(result.outer, proto))
            result.outer.push_back(proto);
        }
        result.inner.push_back(idx.drop_proto_indices());
      }
    }
    result.is_tot = !result.inner.empty();
    return result;
  }

  static bool is_tot_tensor(const Tensor &tensor) {
    for (const Index &idx : tensor.const_indices())
      if (idx.has_proto_indices()) return true;
    return false;
  }

  /// The tensor-of-tensor (PNO/CSV-restricted) array type, spelled out in
  /// full rather than relying on an externally-defined "ArrayToT" alias
  /// (e.g. ta_tensors.h's), so a generated function is compilable given
  /// only <tiledarray.h> -- matching this generator's "self-contained
  /// function" design (see file header).
  static constexpr const char *kArrayToTType =
      "TA::DistArray<TA::Tensor<TA::Tensor<double>>, TA::SparsePolicy>";

  static std::string tensor_cpp_type(const Tensor &tensor) {
    return is_tot_tensor(tensor) ? kArrayToTType : "TA::TSpArrayD";
  }

  std::string join_index_labels(const std::vector<Index> &indices,
                                const Context &ctx) const {
    std::string s;
    bool first = true;
    for (const Index &idx : indices) {
      if (!first) s += ",";
      s += represent(idx, ctx);
      first = false;
    }
    return s;
  }

  /// Flat tensors get the usual comma-joined annotation. ToT tensors get
  /// TA's own "outer;inner" ToT annotation convention (matching Phase 1's
  /// gen_ta_trace_equations.py, which established this convention against
  /// real TiledArray ToT usage already).
  std::string index_annotation(const Tensor &tensor, const Context &ctx) const {
    IndexClass cls = classify_indices(tensor);
    std::string outer_s = join_index_labels(cls.outer, ctx);
    if (!cls.is_tot) return outer_s;
    return outer_s + ";" + join_index_labels(cls.inner, ctx);
  }

  std::string annotated(const Tensor &tensor, const Context &ctx) const {
    return represent(tensor, ctx) + "(\"" + index_annotation(tensor, ctx) +
           "\")";
  }

  std::string stringify_scalar(const Expr &expr, const Context &ctx) const {
    if (expr.is<Variable>()) return represent(expr.as<Variable>(), ctx);
    if (expr.is<Constant>()) return represent(expr.as<Constant>(), ctx);
    if (expr.is<Power>()) return represent(expr.as<Power>(), ctx);
    throw Exception(
        "TiledArrayGenerator: expected a scalar leaf (Variable/Constant/"
        "Power), got " +
        expr.type_name());
  }

  /// Walks a (possibly scalar-nested, per prune_scalar_factor's
  /// re-multiplication) Product, separating out its scalar factors from
  /// its tensor factors. Asserts at most 2 tensor factors survive --
  /// process_computation (export.hpp) only ever constructs a bare
  /// {left, right} Product for EvalOp::Product, so more than 2 tensor
  /// factors here would mean the tree wasn't binarized as expected.
  void collect_product_factors(const Product &product, std::string &scalar_text,
                               std::vector<const Tensor *> &tensors,
                               const Context &ctx) const {
    if (!product.scalar().is_identity()) {
      std::string s = represent(Constant(product.scalar()), ctx);
      scalar_text = scalar_text.empty() ? s : (scalar_text + " * " + s);
    }
    for (std::size_t i = 0; i < product.size(); ++i) {
      const Expr &factor = *product.factor(i);
      if (factor.is<Tensor>()) {
        tensors.push_back(&factor.as<Tensor>());
      } else if (factor.is<Product>()) {
        collect_product_factors(factor.as<Product>(), scalar_text, tensors,
                                ctx);
      } else if (factor.is<Variable>() || factor.is<Constant>() ||
                factor.is<Power>()) {
        std::string s = stringify_scalar(factor, ctx);
        scalar_text = scalar_text.empty() ? s : (scalar_text + " * " + s);
      } else {
        throw Exception(
            "TiledArrayGenerator: unsupported product factor type " +
            factor.type_name());
      }
    }
  }

  /// Whether a 2-tensor ToT product needs a plain TA::einsum call, or the
  /// explicit de-nesting form.
  enum class ContractionMode { Plain, DeNest };

  /// Guards against ToT contraction patterns that don't correspond to a
  /// single valid TA::einsum call, and identifies the one that does but
  /// needs the explicit de-nesting template argument. TA::einsum's ToT
  /// support treats a shared inner (PNO/CSV) index between the two operands
  /// as EITHER a Hadamard (elementwise-preserved, survives into the result)
  /// OR a contracted (summed away) axis -- never both within the same call,
  /// and never partially (mixed Hadamard+contraction --
  /// gen_ta_trace_equations.py classify_equation()'s `hadamard and
  /// contracted` check, line ~165) -- that combination is genuinely
  /// unsupported and throws below.
  ///
  /// A ToT x ToT product whose shared inner indices are ALL contracted away
  /// AND whose result is itself flat (non-ToT) is a DIFFERENT, and valid,
  /// case: full de-nesting. Confirmed by direct inspection (not guessed):
  /// SeQuant's own existing TiledArray eval backend
  /// (core/eval/backends/tiledarray/result.hpp:~619-624) dispatches exactly
  /// this case ("ToT * ToT -> T", `node.left()->tot() && node.right()->tot()
  /// && !node->tot()`) to `TA::einsum<TA::DeNest::True>(A, B, result_ann)`
  /// with a result annotation carrying NO ';' -- and TiledArray's own test
  /// suite (tests/dot_inner.cpp, tests/einsum.cpp) exercises exactly this
  /// call shape. This is the real, supported PNO/CSV-to-flat back-transform
  /// pattern that MPQC's own equations hit on essentially every PNO-
  /// touching term (confirmed against the Phase 0/1 T1/T2 fixture) -- NOT
  /// the separate, still-unresolved EMPIRICALLY_UNSAFE_CATALOG
  /// (eq62/63/74/75) data-dependent crash gen_ta_trace_equations.py
  /// isolated, which this check does not attempt to reproduce (no
  /// structural signature distinguishes those four terms from the many
  /// others that work).
  ///
  /// A shared inner identity being cleanly contracted (not partially) is
  /// NOT by itself special -- if one operand also carries a SEPARATE,
  /// unshared inner axis that survives untouched into the result (e.g. t's
  /// OTHER PNO index passing through while its shared one contracts
  /// against C), the result stays genuinely ToT and an ordinary (non-
  /// de-nest) TA::einsum call handles it correctly; only when the shared
  /// contraction drains the result's inner dimension to nothing (a
  /// genuinely flat/non-ToT result) is the explicit de-nest dispatch
  /// needed -- confirmed against the real Phase 0/1 T1/T2 fixture, which
  /// exercises both this ordinary partial-contraction case and the full
  /// de-nest case.
  ContractionMode check_tot_contraction_safety(
      const Tensor &a, const Tensor &b,
      const std::string &result_annotation) const {
    IndexClass ca = classify_indices(a);
    IndexClass cb = classify_indices(b);
    if (!ca.is_tot && !cb.is_tot) return ContractionMode::Plain;

    // CONFIRMED REAL CRASH (2026-07-18, Phase 3 numeric ground-truth
    // testing, twinkly-dazzling-shamir.md): a plain (non-DeNest)
    // TA::einsum(flat_operand, ToT_operand, ann) call -- exactly what
    // product_rhs emits by default whenever only ONE of the two operands
    // is ToT -- segfaults inside TiledArray::Einsum::einsum<DeNest::False>
    // when the flat operand shares an OUTER index directly with the ToT
    // operand (reproduced standalone: g(i,x) * C(i,x;a') -> R(i;a'), i
    // shared+surviving, x shared+contracted; gdb backtrace confirms the
    // crash is inside TA's own DeNest::False einsum internals, not this
    // generator's annotation). This is not a rare corner case: 40/120 (R1)
    // and 115/399 (R2) of the plain-einsum calls in the real T1/T2
    // residuals have exactly this shape. Until TiledArray's own bug is
    // root-caused/fixed or a safe alternative call pattern is found, refuse
    // to emit it -- matches this generator's standing principle (see
    // EMPIRICALLY_UNSAFE_CATALOG note above) of throwing rather than
    // silently emitting code known to crash on real data.
    if (ca.is_tot != cb.is_tot) {
      const IndexClass &tot_side = ca.is_tot ? ca : cb;
      const IndexClass &flat_side = ca.is_tot ? cb : ca;
      for (const Index &idx : flat_side.outer) {
        if (contains_index(tot_side.outer, idx)) {
          throw Exception(
              "TiledArrayGenerator: a flat operand sharing an outer index "
              "directly with a ToT operand, contracted via plain (non-"
              "DeNest) TA::einsum, is a confirmed TiledArray-internal crash "
              "(segfault in Einsum::einsum<DeNest::False>) -- refusing to "
              "emit; see Phase 3 ground-truth testing notes in "
              "twinkly-dazzling-shamir.md");
        }
      }
    }

    std::string result_inner_part;
    auto semi = result_annotation.find(';');
    if (semi != std::string::npos)
      result_inner_part = result_annotation.substr(semi + 1);

    auto inner_survives = [&](const Index &idx) {
      // classify_indices() gives us drop_proto_indices() copies whose
      // label() equals what represent()/index_annotation() emits, so a
      // plain string containment check against the comma-separated inner
      // part is a correct (if crude) membership test.
      std::string tok = sanitize_identifier(toUtf8(idx.label()));
      std::string field;
      std::istringstream iss(result_inner_part);
      while (std::getline(iss, field, ',')) {
        if (field == tok) return true;
      }
      return false;
    };

    std::vector<Index> shared;
    for (const Index &ia : ca.inner)
      for (const Index &ib : cb.inner)
        if (ia == ib && !contains_index(shared, ia)) {
          shared.push_back(ia);
          break;
        }
    if (shared.empty()) return ContractionMode::Plain;

    bool any_survive = false;
    bool any_contracted = false;
    for (const Index &idx : shared) {
      if (inner_survives(idx))
        any_survive = true;
      else
        any_contracted = true;
    }
    if (any_survive && any_contracted) {
      throw Exception(
          "TiledArrayGenerator: mixed Hadamard+contraction over ToT inner "
          "(PNO/CSV) indices in a single product is not expressible as one "
          "TA::einsum call");
    }
    // any_contracted here means every shared identity is cleanly contracted
    // (the mixed case above already threw) -- this is an ordinary
    // TA::einsum ToT contraction UNLESS it also drains the result's
    // combined inner dimension to nothing, which is the special de-nest
    // case (the result TYPE itself changes from ArrayToT to TA::TSpArrayD,
    // requiring the explicit TA::einsum<DeNest::True> entry point). An
    // unshared inner index surviving from just one operand (e.g. t's OTHER
    // PNO axis passing through untouched while the SHARED one contracts
    // against C) is completely ordinary and needs no special dispatch --
    // by this point shared is nonempty, so ca.is_tot/cb.is_tot are already
    // both guaranteed true.
    if (any_contracted && result_inner_part.empty()) {
      return ContractionMode::DeNest;
    }
    return ContractionMode::Plain;
  }

  std::string product_rhs(const Product &product,
                          const std::string &result_annotation,
                          const Context &ctx) const {
    std::string scalar_text;
    std::vector<const Tensor *> tensors;
    collect_product_factors(product, scalar_text, tensors, ctx);

    std::string tensor_expr;
    if (tensors.size() == 2) {
      ContractionMode mode = check_tot_contraction_safety(
          *tensors[0], *tensors[1], result_annotation);
      const std::string einsum_call =
          mode == ContractionMode::DeNest ? "TA::einsum<TA::DeNest::True>"
                                          : "TA::einsum";
      // TA::einsum(...) returns a concrete DistArray, not a TsrExpr -- it
      // must be re-annotated before it can participate in a `+=`
      // accumulation or a scalar-multiply expression (both require an
      // actual tensor *expression*, not a bare array). Self-annotating
      // with the SAME result_annotation immediately turns it into one.
      tensor_expr = einsum_call + "(" + annotated(*tensors[0], ctx) + ", " +
                    annotated(*tensors[1], ctx) + ", \"" + result_annotation +
                    "\")(\"" + result_annotation + "\")";
    } else if (tensors.size() == 1) {
      tensor_expr = annotated(*tensors[0], ctx);
    } else if (tensors.empty()) {
      throw Exception(
          "TiledArrayGenerator: a Tensor-valued result's Product has no "
          "tensor factors at all");
    } else {
      throw Exception(
          "TiledArrayGenerator: Product has " + std::to_string(tensors.size()) +
          " tensor factors -- expected at most 2 (tree not binarized as "
          "process_computation guarantees; see export.hpp)");
    }

    if (scalar_text.empty()) return tensor_expr;
    // represent(Constant)/represent(Power) already parenthesize their own
    // output, so scalar_text needs no extra wrapping here.
    return "(" + tensor_expr + ") * " + scalar_text;
  }

  std::string compute_rhs(const Expr &expr, const std::string &result_annotation,
                          const Context &ctx) const {
    if (expr.is<Tensor>()) return annotated(expr.as<Tensor>(), ctx);
    if (expr.is<Product>())
      return product_rhs(expr.as<Product>(), result_annotation, ctx);
    throw Exception(
        "TiledArrayGenerator: unsupported compute() expression type " +
        expr.type_name());
  }

  /// Like product_rhs, but for a Product whose result is a bare scalar
  /// (Variable), e.g. a fully-contracted energy-like term with no free
  /// indices remaining. TA has no "einsum with empty output" call --
  /// TA::dot(A, B) is the real-valued full contraction that plays that
  /// role (matching cck.ipp's own use of TA::dot for exactly this case).
  std::string product_scalar_rhs(const Product &product,
                                 const Context &ctx) const {
    std::string scalar_text;
    std::vector<const Tensor *> tensors;
    collect_product_factors(product, scalar_text, tensors, ctx);

    std::string tensor_expr;
    if (tensors.size() == 2) {
      // TA::dot(...) is only ever used in MPQC's real cck.ipp on flat
      // (non-ToT) arrays -- confirmed by direct inspection, no call site
      // there passes an ArrayToT to it. Its semantics for a nested
      // (PNO/CSV) tile type are therefore unverified in practice, so
      // rather than guess (and risk silently emitting a call that either
      // fails to compile or -- worse -- compiles but contracts the inner
      // dimension incorrectly), refuse to emit it here. Phase 3's
      // from-scratch numpy ground truth + small compiled probes are where
      // the correct ToT full-contraction call pattern gets pinned down
      // and this is upgraded from a throw to real code.
      if (is_tot_tensor(*tensors[0]) || is_tot_tensor(*tensors[1])) {
        throw Exception(
            "TiledArrayGenerator: fully-contracted (scalar-result) product "
            "of ToT (PNO/CSV-restricted) tensors is not yet supported -- "
            "TA::dot's semantics for nested (ArrayToT) tiles are unverified "
            "(MPQC's own cck.ipp never calls it on ArrayToT); see Phase 3 "
            "of twinkly-dazzling-shamir.md");
      }
      tensor_expr =
          "TA::dot(" + annotated(*tensors[0], ctx) + ", " +
          annotated(*tensors[1], ctx) + ")";
    } else if (tensors.size() == 1) {
      // A single tensor fully reduced to a scalar (e.g. sum of all
      // elements) -- not needed by any case exercised so far, but handled
      // for completeness via TA's own reduction. Same ToT caveat as the
      // two-tensor TA::dot case above applies to .sum() on a nested tile.
      if (is_tot_tensor(*tensors[0])) {
        throw Exception(
            "TiledArrayGenerator: fully-contracted (scalar-result) "
            "reduction of a single ToT (PNO/CSV-restricted) tensor is not "
            "yet supported -- see Phase 3 of twinkly-dazzling-shamir.md");
      }
      tensor_expr = annotated(*tensors[0], ctx) + ".sum()";
    } else if (tensors.empty()) {
      return scalar_text.empty() ? "0.0" : scalar_text;
    } else {
      throw Exception(
          "TiledArrayGenerator: Product (scalar result) has " +
          std::to_string(tensors.size()) +
          " tensor factors -- expected at most 2");
    }

    if (scalar_text.empty()) return tensor_expr;
    return "(" + tensor_expr + ") * " + scalar_text;
  }

  std::string compute_scalar_rhs(const Expr &expr, const Context &ctx) const {
    if (expr.is<Variable>() || expr.is<Constant>() || expr.is<Power>())
      return stringify_scalar(expr, ctx);
    if (expr.is<Product>()) return product_scalar_rhs(expr.as<Product>(), ctx);
    throw Exception(
        "TiledArrayGenerator: unsupported compute() (scalar result) "
        "expression type " +
        expr.type_name());
  }

  std::string render_function() const {
    std::ostringstream oss;
    std::string result_type =
        !m_result_is_tensor ? "double"
                            : (m_result_is_tot ? kArrayToTType : "TA::TSpArrayD");
    oss << result_type << " " << m_current_name << "(";
    for (std::size_t i = 0; i < m_params.size(); ++i) {
      if (i > 0) oss << ", ";
      oss << m_params[i].first << " " << m_params[i].second;
    }
    oss << ") {\n";
    oss << m_local_decls;
    oss << m_body;
    oss << m_indent << "return " << m_result_name << ";\n";
    oss << "}\n";
    return oss.str();
  }
};

}  // namespace sequant

#endif  // SEQUANT_CORE_EXPORT_TILEDARRAY_GENERATOR_HPP
