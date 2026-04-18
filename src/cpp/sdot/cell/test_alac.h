void test_alac( auto &out, const auto &inp ) {
    out() = inp * inp;
}

void test_alac_backward( auto &/* out */, const auto &inp, auto &grad_inp, const auto &grad_out ) {
    grad_inp() = 2 * grad_out() * inp;
}
