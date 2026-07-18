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
    return sanitize_identifier(toUtf8(idx.full_label()));
  }

  std::string represent(const Tensor &tensor,
                        const Context &) const override {
    return tensor_var_name(tensor);
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
    m_body += m_indent + represent(tensor, ctx) + " = TA::TSpArrayD();\n";
  }
  void unload(const Tensor &tensor, const Context &ctx) override {
    const std::string name = represent(tensor, ctx);
    if (m_leaf_names.count(name)) return;  // never release a parameter
    m_body += m_indent + name + " = TA::TSpArrayD();  // release\n";
  }
  void destroy(const Tensor &tensor, const Context &ctx) override {
    unload(tensor, ctx);
  }
  void persist(const Tensor &tensor, const Context &ctx) override {
    m_result_name = represent(tensor, ctx);
    m_result_is_tensor = true;
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
    if (usage == Usage::Terminal) {
      m_params.emplace_back("const TA::TSpArrayD&", name);
      m_leaf_names.insert(name);
    } else {
      m_local_decls += m_indent + "TA::TSpArrayD " + name + ";\n";
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
  std::vector<std::pair<std::string, std::string>> m_params;
  std::set<std::string> m_declared_names;
  std::set<std::string> m_leaf_names;
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
  /// PythonEinsumGeneratorBase::tensor_name's index-space tagging.
  std::string tensor_var_name(const Tensor &tensor) const {
    std::string key = toUtf8(tensor.label());
    for (const Index &idx : tensor.const_indices()) {
      key += "_" + toUtf8(idx.space().base_key());
    }
    auto it = m_tensor_names.find(key);
    if (it != m_tensor_names.end()) return it->second;
    std::string name = sanitize_identifier(key);
    m_tensor_names.emplace(key, name);
    return name;
  }

  std::string index_annotation(const Tensor &tensor, const Context &ctx) const {
    std::string s;
    bool first = true;
    for (const Index &idx : tensor.const_indices()) {
      if (!first) s += ",";
      s += represent(idx, ctx);
      first = false;
    }
    return s;
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

  std::string product_rhs(const Product &product,
                          const std::string &result_annotation,
                          const Context &ctx) const {
    std::string scalar_text;
    std::vector<const Tensor *> tensors;
    collect_product_factors(product, scalar_text, tensors, ctx);

    std::string tensor_expr;
    if (tensors.size() == 2) {
      // TA::einsum(...) returns a concrete DistArray, not a TsrExpr -- it
      // must be re-annotated before it can participate in a `+=`
      // accumulation or a scalar-multiply expression (both require an
      // actual tensor *expression*, not a bare array). Self-annotating
      // with the SAME result_annotation immediately turns it into one.
      tensor_expr = "TA::einsum(" + annotated(*tensors[0], ctx) + ", " +
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
      tensor_expr =
          "TA::dot(" + annotated(*tensors[0], ctx) + ", " +
          annotated(*tensors[1], ctx) + ")";
    } else if (tensors.size() == 1) {
      // A single tensor fully reduced to a scalar (e.g. sum of all
      // elements) -- not needed by any case exercised so far, but handled
      // for completeness via TA's own reduction.
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
    oss << (m_result_is_tensor ? "TA::TSpArrayD " : "double ")
        << m_current_name << "(";
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
