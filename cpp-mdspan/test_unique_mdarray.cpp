#include "unique_mdarray.hpp"
#include <iostream>

namespace test {

using namespace md;

template<class ElementType,
  class Extents,
  class Layout,
  class Deleter,
  class ... Indices>
std::remove_const_t<ElementType> get_value_at(
  const unique_mdarray<ElementType, Extents, Layout, Deleter>& x,
  Indices... indices)
{
#if defined(MDSPAN_USE_BRACKET_OPERATOR) && (MDSPAN_USE_BRACKET_OPERATOR != 0)
  return x[indices...];
#else
  return x(indices...);
#endif
}

template<class ElementType,
  class Extents,
  class Layout,
  class Deleter,
  class ... Indices>
void set_value_at(
  const unique_mdarray<ElementType, Extents, Layout, Deleter>& x,
  std::remove_const_t<ElementType> value,
  Indices... indices)
{
#if defined(MDSPAN_USE_BRACKET_OPERATOR) && (MDSPAN_USE_BRACKET_OPERATOR != 0)
  x[indices...] = value;
#else
  x(indices...) = value;
#endif
}

template<class ElementType,
  class Extents,
  class Layout,
  class Deleter,
  class ... Indices>
std::remove_const_t<ElementType> get_value_at_with_array(
  const unique_mdarray<ElementType, Extents, Layout, Deleter>& x,
  Indices... indices)
{
  using index_type = typename Extents::index_type;
  std::array<index_type, Extents::rank()> inds{indices...};
  return x[inds];
}

template<class ElementType,
  class Extents,
  class Layout,
  class Deleter,
  class ... Indices>
void set_value_at_with_array(
  const unique_mdarray<ElementType, Extents, Layout, Deleter>& x,
  std::remove_const_t<ElementType> value,
  Indices... indices)
{
  using index_type = typename Extents::index_type;
  std::array<index_type, Extents::rank()> inds{indices...};
  x[inds] = value;
}

template<class ElementType,
  class Extents,
  class Layout,
  class Deleter,
  class ... Indices>
std::remove_const_t<ElementType> get_value_at_with_span(
  const unique_mdarray<ElementType, Extents, Layout, Deleter>& x,
  Indices... indices)
{
  using index_type = typename Extents::index_type;
  std::array<index_type, Extents::rank()> inds{
    static_cast<index_type>(indices)...
  };
  std::span<index_type, Extents::rank()> sp{inds.data(), inds.size()};
  return x[sp];
}

template<class ElementType,
  class Extents,
  class Layout,
  class Deleter,
  class ... Indices>
void set_value_at_with_span(
  const unique_mdarray<ElementType, Extents, Layout, Deleter>& x,
  std::remove_const_t<ElementType> value,
  Indices... indices)
{
  using index_type = typename Extents::index_type;
  std::array<index_type, Extents::rank()> inds{
    static_cast<index_type>(indices)...
  };
  std::span<index_type, Extents::rank()> sp{inds.data(), inds.size()};
  x[sp] = value;
}

template<class ElementType, class Extents, class Layout>
void test_implicit_conversion(
  mdspan<ElementType, Extents, Layout, default_accessor<ElementType>>)
{}

template<class Mapping, class Deleter, size_t SpanExtent = std::dynamic_extent>
void test_float_construction(const Mapping& m, const Deleter& d, std::integral_constant<size_t, SpanExtent> = {}) {
  std::unique_ptr<float[]> ptr = std::make_unique<float[]>(m.required_span_size());
  using extents_type = typename Mapping::extents_type;
  using layout_type = typename Mapping::layout_type;
  using index_type = typename Mapping::index_type;

  float* raw_ptr_x = ptr.get();
  std::span<float, SpanExtent> sp_x{ptr.release(), static_cast<size_t>(m.required_span_size())};
  unique_mdarray<float, extents_type, layout_type, Deleter> x{sp_x, m, d};

  static_assert(x.rank() == extents_type::rank());
  static_assert(x.rank_dynamic() == extents_type::rank_dynamic());
  assert(x.get() == raw_ptr_x);
  assert(x.extents() == m.extents());
  assert(x.mapping() == m);

  set_value_at(x, 42.0f, 1, 2);
  assert(get_value_at(x, 1, 2) == 42.0f);

  set_value_at_with_array(x, 43.0f, 1, 2);
  assert(get_value_at_with_array(x, 1, 2) == 43.0f);

  set_value_at_with_span(x, 44.0f, 1, 2);
  assert(get_value_at_with_span(x, 1, 2) == 44.0f);

  set_value_at(x, 45.0f, 1, 2);
  assert(get_value_at(x, 1, 2) == 45.0f);

  if constexpr (extents_type::rank_dynamic() != 0) {
    try {
      float* raw_ptr = x.release();
      delete [] raw_ptr;
    }
    catch (...) {
      std::cerr << "x.release() threw an exception\n";
      assert(false);
    }
    assert(x.get() == nullptr);

    // This only works if Deleter uses the same allocation strategy.
    auto ptr_x2 = std::make_unique<float[]>(m.required_span_size());
    std::span<float, SpanExtent> sp_x2{
      ptr_x2.release(),
      static_cast<size_t>(m.required_span_size())
    };
    x = unique_mdarray<float, extents_type, layout_type, Deleter>{sp_x2, m, d};
  }

  for (index_type r = 0; r < x.extent(0); ++r) {
    for (index_type c = 0; c < x.extent(1); ++c) {
      set_value_at(x, 1.0f + static_cast<float>(r + c * x.extent(1)), r, c);
    }
  }

  mdspan<float, extents_type, layout_type, default_accessor<float>> x_view = x;
  for (index_type r = 0; r < x.extent(0); ++r) {
    for (index_type c = 0; c < x.extent(1); ++c) {
#if defined(MDSPAN_USE_BRACKET_OPERATOR) && (MDSPAN_USE_BRACKET_OPERATOR != 0)
      const float x_rc = x_view[r, c];
#else
      const float x_rc = x_view(r, c);
#endif
      const float val = 1.0f + static_cast<float>(r + c * x.extent(1));
      assert(x_rc == val);
    } 
  }

  if constexpr (std::is_swappable_v<Deleter>) {
    // This only works if Deleter uses the same allocation strategy.
    std::unique_ptr<float[]> ptr_y = std::make_unique<float[]>(m.required_span_size());
    std::span<float> sp_y{ptr_y.release(), static_cast<size_t>(m.required_span_size())};
    unique_mdarray<float, extents_type, layout_type, Deleter> y{sp_y, m, d};

    for (index_type r = 0; r < y.extent(0); ++r) {
      for (index_type c = 0; c < y.extent(1); ++c) {
        set_value_at(y, -(1.0f + static_cast<float>(r + c * x.extent(1))), r, c);
      }
    }

    using std::swap;
    swap(x, y);
    for (index_type r = 0; r < x.extent(0); ++r) {
      for (index_type c = 0; c < x.extent(1); ++c) {
        const float x_rc = get_value_at(x, r, c);
        const float y_rc = get_value_at(y, r, c);
        const float val = 1.0f + static_cast<float>(r + c * x.extent(1));
        assert(x_rc == -val);
        assert(y_rc == val);
      } 
    }
  }
}

template<class T>
struct my_array_deleter {
  void operator() (T* ptr) const {
    delete [] ptr;
  }

  // Make it not swappable, just to test constraints on swap.
  friend constexpr void
  swap(my_array_deleter<T>&, my_array_deleter<T>&) = delete;
};

template<class Deleter>
void construction(const Deleter& d) {
  using layout_type = layout_right;

  {
    using extents_type = extents<int, 2, 3>;
    extents_type e{2, 3};
    using mapping_type = layout_type::mapping<extents_type>;
    test_float_construction(mapping_type{e}, d);
    test_float_construction(mapping_type{e}, d, std::integral_constant<size_t, 6>{});
  }

  {
    using extents_type = extents<int, std::dynamic_extent, 3>;
    extents_type e{2, 3};
    using mapping_type = layout_type::mapping<extents_type>;
    test_float_construction(mapping_type{e}, d);
    test_float_construction(mapping_type{e}, d, std::integral_constant<size_t, 6>{});
  }

  {
    using extents_type = extents<int, 2, std::dynamic_extent>;
    extents_type e{2, 3};
    using mapping_type = layout_type::mapping<extents_type>;
    test_float_construction(mapping_type{e}, d);
    test_float_construction(mapping_type{e}, d, std::integral_constant<size_t, 6>{});
  }

  {
    using extents_type = extents<int, std::dynamic_extent, std::dynamic_extent>;
    extents_type e{2, 3};
    using mapping_type = layout_type::mapping<extents_type>;
    test_float_construction(mapping_type{e}, d);
    test_float_construction(mapping_type{e}, d, std::integral_constant<size_t, 6>{});
  }
}

void make_unique_mdarray_with_mapping() {
  {
    using extents_type = md::dims<0>;
    const extents_type exts{};
    const auto mapping = md::layout_right::template mapping<extents_type>{exts};
    auto x = md::make_unique_mdarray<float>(mapping);
    using x_type = decltype(x);
    static_assert(std::is_same_v<x_type::extents_type, extents_type>);
    static_assert(std::is_same_v<x_type::layout_type, md::layout_right>);
    static_assert(x.rank() == 0);
    assert(x.mapping().extents() == exts);
  }
  {
    using extents_type = md::dims<1>;
    const extents_type exts{3};
    const auto mapping = md::layout_right::template mapping<extents_type>{exts};
    auto x = md::make_unique_mdarray<float>(mapping);
    using x_type = decltype(x);
    static_assert(std::is_same_v<x_type::extents_type, extents_type>);
    static_assert(std::is_same_v<x_type::layout_type, md::layout_right>);
    static_assert(x.rank() == 1);
    assert(x.mapping().extents() == exts);
    assert(x.extent(0) == 3);
  }
  {
    using extents_type = md::dims<2>;
    const extents_type exts{3, 5};
    const auto mapping = md::layout_right::template mapping<extents_type>{exts};
    auto x = md::make_unique_mdarray<float>(mapping);
    using x_type = decltype(x);
    static_assert(std::is_same_v<x_type::extents_type, extents_type>);
    static_assert(std::is_same_v<x_type::layout_type, md::layout_right>);
    static_assert(x.rank() == 2);
    assert(x.mapping().extents() == exts);
    assert(x.extent(0) == 3);
    assert(x.extent(1) == 5);
  }
}

void make_unique_mdarray_with_extents() {
  const auto exts = md::dims<3>{3, 5, 7};
  auto x = md::make_unique_mdarray<float>(exts);
  using x_type = decltype(x);
  static_assert(std::is_same_v<x_type::extents_type, md::dims<3>>);
  static_assert(std::is_same_v<x_type::layout_type, md::layout_right>);
  static_assert(x.rank() == 3);
  assert(x.mapping().extents() == exts);
  assert(x.extent(0) == 3);
  assert(x.extent(1) == 5);
  assert(x.extent(2) == 7);
}

} // namespace test

int main() {
  test::construction(std::default_delete<float[]>{});
  test::construction(test::my_array_deleter<float>{});
  test::make_unique_mdarray_with_mapping();
  test::make_unique_mdarray_with_extents();
  return 0;
}
