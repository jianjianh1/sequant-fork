# Standalone manual driver (has its own main(), not a Catch2 unit test):
# derives the real closed-shell CSV-CCSD T1/T2 residual equations via the
# full pipeline MPQC's own cck.ipp/sequant.cpp uses, and exports them as
# static C++ via TiledArrayGenerator (core/export/tiledarray_generator.hpp).
# See the file's own header comment for the exact pipeline steps replicated.
add_executable(ta_generator_cc_test test_csv_ccsd_derivation.cpp)
target_link_libraries(ta_generator_cc_test PRIVATE SeQuant::export SeQuant::mbpt)
