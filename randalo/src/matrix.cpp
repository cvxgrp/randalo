#include "decl.hpp"
#include <adelie_core/io/io_snp_unphased.hpp>
#include <adelie_core/matrix/utils.hpp>
#include <adelie_core/util/types.hpp>
#include <omp.h>

namespace ad = adelie_core;
namespace py = pybind11;

template <class IOType, 
          class ValueType=double,
          class IndexType=Eigen::Index>
void hessian(
    const std::vector<const IOType*>& ios,
    const Eigen::Ref<const ad::util::rowvec_type<IndexType>>& subset,
    const Eigen::Ref<const ad::util::rowvec_type<ValueType>>& weights,
    Eigen::Ref<ad::util::rowmat_type<ValueType>> out,
    size_t n_threads
)
{
    using io_t = std::decay_t<IOType>;
    using index_t = IndexType;
    using value_t = ValueType;
    using vec_index_t = ad::util::rowvec_type<index_t>;
    using vec_value_t = ad::util::rowvec_type<value_t>;

    constexpr auto _max = std::numeric_limits<value_t>::max();

    ad::util::rowvec_type<value_t> buff; // unused argument

    if (ios.size() == 0 || subset.size() == 0) return;

    for (Eigen::Index i = 1; i < subset.size(); ++i) {
        if (subset[i-1] <= subset[i]) continue;
        throw std::runtime_error(
            "subset must be in increasing order."
        );
    }

    // n should be the same across all IOs
    const size_t n = ios[0]->rows();
    ad::util::rowarr_type<value_t> vbuff(n_threads, n);
    ad::util::rowarr_type<index_t> ibuff(n_threads, n);
    vbuff.fill(_max);

    // compute IO index and beginning index into X corresponding to each subset index value.
    // NOTE: subset is assumed to be sorted in increasing order.
    ad::util::rowvec_type<index_t> io_index(subset.size());
    ad::util::rowvec_type<index_t> io_begin(subset.size());
    {
        size_t io_idx = 0;
        size_t begin = 0;
        for (Eigen::Index i = 0; i < subset.size(); ++i) {
            const size_t j = subset[i];
            while (j >= begin + ios[io_idx]->cols()) {
                begin += ios[io_idx]->cols();
                ++io_idx;
            }
            io_index[i] = io_idx;
            io_begin[i] = begin;
        }
    }

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (size_t i1 = 0; i1 < subset.size(); ++i1) {
        const auto index_1 = subset[i1] - io_begin[i1];
        const auto& io_1 = *ios[io_index[i1]];
        const value_t imp_1 = io_1.impute()[index_1];
        const auto thr_id = omp_get_thread_num();
        auto vbuff_thr = vbuff.row(thr_id);
        auto ibuff_thr = ibuff.row(thr_id);

        // cache index_1 information
        size_t nnz = 0;
        for (size_t c = 0; c < io_t::n_categories; ++c) {
            auto it = io_1.begin(index_1, c);
            const auto end = io_1.end(index_1, c);
            const value_t val = (c == 0) ? imp_1 : c;
            for (; it != end; ++it) {
                const auto idx = *it;
                vbuff_thr[idx] = val;
                ibuff_thr[nnz] = idx;
                ++nnz;
            }
        }

        for (size_t i2 = 0; i2 <= i1; ++i2) {
            const auto index_2 = subset[i2] - io_begin[i2];
            const auto& io_2 = *ios[io_index[i2]];
            out(i1, i2) = ad::matrix::snp_unphased_dot(
                [](auto x) { return x; },
                io_2, 
                index_2,
                weights * (
                    (vbuff_thr != _max).template cast<value_t>() * vbuff_thr
                ),
                1,
                buff
            );
            out(i2, i1) = out(i1, i2);
        }

        // keep invariance by populating with inf
        for (size_t i = 0; i < nnz; ++i) {
            vbuff_thr[ibuff_thr[i]] = _max;
        }
    }
}

void register_matrix(py::module_& m)
{
    using io_snp_unphased_t = ad::io::IOSNPUnphased<>;

    m.def("hessian", hessian<io_snp_unphased_t>);
}
