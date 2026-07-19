TA::TSpArrayD compute_R(const TA::DistArray<TA::Tensor<TA::Tensor<double>>, TA::SparsePolicy>& T_i_ap1, const TA::DistArray<TA::Tensor<TA::Tensor<double>>, TA::SparsePolicy>& C_ap1_a) {
  TA::TSpArrayD R_i_a;
  R_i_a("i_1,a_1") = TA::einsum<TA::DeNest::True>(T_i_ap1("i_1;a_2"), C_ap1_a("i_1,a_1;a_2"), "i_1,a_1")("i_1,a_1");
  return R_i_a;
}

