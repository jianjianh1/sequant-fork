TA::DistArray<TA::Tensor<TA::Tensor<double>>, TA::SparsePolicy> compute_R2(const TA::TSpArrayD& g_i_a, const TA::DistArray<TA::Tensor<TA::Tensor<double>>, TA::SparsePolicy>& C_ap1_a) {
  TA::DistArray<TA::Tensor<TA::Tensor<double>>, TA::SparsePolicy> R2_i_ap1;
  R2_i_ap1("i_1;a_2") = TA::einsum(g_i_a("i_1,a_1"), C_ap1_a("i_1,a_1;a_2"), "i_1;a_2")("i_1;a_2");
  return R2_i_ap1;
}

