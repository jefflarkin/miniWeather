#pragma once

#include "mdspan/mdspan.hpp"
#include <cassert>
#include <memory>
#include <span>
#include <type_traits>

// operator[] would need C++23 for multiple parameters.
// The reference mdspan has its own macros (please see above).
#if defined(MDSPAN_USE_BRACKET_OPERATOR) && (MDSPAN_USE_BRACKET_OPERATOR != 0)
#  define MDSPAN_ARRAY_ACCESS_OPERATOR operator[]
#else
#  define MDSPAN_ARRAY_ACCESS_OPERATOR operator()
#endif

namespace md {

namespace stdex = MDSPAN_IMPL_STANDARD_NAMESPACE :: MDSPAN_IMPL_PROPOSED_NAMESPACE ;

using std::dynamic_extent;
using std::size_t;
using MDSPAN_IMPL_STANDARD_NAMESPACE :: extents;
using MDSPAN_IMPL_STANDARD_NAMESPACE :: layout_right;
using MDSPAN_IMPL_STANDARD_NAMESPACE :: mdspan;
using MDSPAN_IMPL_STANDARD_NAMESPACE :: default_accessor;

namespace impl {

template<class IndexType, size_t ... Extents>
constexpr size_t product_of_static_extents(extents<IndexType, Extents...> e) {
  return ((e.static_extent(Extents) == dynamic_extent ? size_t(0) : e.static_extent(Extents)) * ... * size_t(1));
}

template<class IndexType>
constexpr typename extents<IndexType>::size_type
  forward_product_of_extents(extents<IndexType>)
{
  return 0;
}

template<class IndexType, size_t ... Exts>
constexpr typename extents<IndexType, Exts...>::size_type
  forward_product_of_extents(extents<IndexType, Exts...> e)
{
  return (e.extent(Exts) * ... * size_t(1));
}

template<class IndexType, size_t ... Exts>
constexpr bool empty(extents<IndexType, Exts...> e)
{
  return ((e.extent(Exts) == 0) || ... || true);
}

} // namespace impl

// Differences from P1684:
//
// 1. Separate dynamic allocation case from static allocation case.
//
// This sidesteps all the issues with mdarray having
// moved-from behavior that depends on the container type.
//
// 2. It's not a container adapter.
//
// Users don't really want to think about container adapter behavior.
// std::vector has to store capacity(), which mdarray doesn't use.
//
// 3. It's not even a container.
//
// Containers are "synchronous by nature."
// Their constructors have a postconditition
// that element access is valid, and thus,
// that the fill or copy that created them is done.
//
// Containers need to know how to fill and copy elements.
// That means needing to know about CUDA streams and synchronization.
// Users already need to handle that in their parallel algorithms.
// Some container constructors (e.g., mdarray construction from mdspan)
// have no convenient way to convey a CUDA stream.
//
// Containers need allocators for two reasons:
// because their constructors need to allocate,
// and because they permit resizing.
// We don't want to permit resizing
// (as extents can be any combination of static or dynamic),
// and it's a lot easier to let users allocate
// than to make the constructor do it.

// Key features:
//
// 1. Behaves like a pointer (unique_ptr, specifically),
//    not like a container.
//    It doesn't need an allocator or a CUDA stream.
//
// 2. Callers are responsible for element fills and (deep) copies.
//    Callers don't have to pay for initialization of elements
//    (if value_type is implicit-lifetime).
//
// 3. Default-constructing it or moving from it results
//    in a valid (empty) multidimensional array.
//    You can't do either if the extents are all static.
//
// 4. You can reuse its storage safely by calling release().
//    This has the same constraints as default construction
//    or the move constructor, and results in a valid (empty)
//    multidimensional array.
//
// Execution policies should be separate from allocations,
// just like parallel algorithms are separate from allocations.
// If the deleter does cudaFreeAsync, it should store its own stream.
// Otherwise, users would manage the stream as part of
// their asynchronous call graph, however they choose to do that.
// For example, if using std::execution,
// a sender adapter for asynchronous allocations would return
// a sender of unique_mdarray
// with the CUDA stream stored in the scheduler.

// Design questions:
//
// 1. Always act as-if default_accessor<element_type>,
//    or permit custom accessors (e.g., aligned_accessor)?
//
// 2. Permit reset(p) (with nonnull pointer p)?
//
// 3. Take span<ElementType, Extent> instead of ElementType* ?
//
// Regarding (1), the simpler design would make the viewing mdspan
// always have default_accessor<ElementType>.
// This won't give host vs. device access protection.  It would also
// lose information that the creator has, like overalignment.
// A more general design would let the user provide an accessor.
// This means that data_handle_type might not be ElementType*, etc.,
// so that we couldn't just use unique_ptr<ElementType[], Deleter> to implement it.
//
// Regarding (2), reset(p) implies that
// [p, p + mapping_.required_span_size()) is a valid range.
// While users could make a mistake with p,
// they might have made the same mistake with the constructor.
//
// Regarding (3), using span makes constructor preconditions
// more explicit and checkable.  It does mean that a common pattern
// for allocating memory and transferring it to the unique_mdarray
// becomes less safe, because there are more steps between releasing
// the unique_ptr's control of the allocation (see code below)
// and creating the unique_mdarray.
//
// auto ptr = std::make_unique_ptr<float[]>(m.required_span_size());
// std::span<float> sp{ptr.release(), m.required_span_size()};
// unique_mdarray<float, extents_type> md{sp, m};
//
// Contrast that with using a raw pointer.
//
// auto ptr = std::make_unique_ptr<float[]>(m.required_span_size());
// unique_mdarray<float, extents_type> md{ptr.release(), m};
//
// Note that unique_mdarray's constructor could throw if, for example,
// the layout mapping's copy constructor throws.
//
// ===================
// Template parameters
// ===================
//
// ElementType: The (possibly const-qualified) type of each element.
//   Const qualification is permitted, just as with unique_ptr.
// Extents: Specialization of std::extents.
// Layout: As in mdspan.
// Deleter: As in unique_ptr; must be an array deleter.
template<class ElementType,
  class Extents,
  class Layout = layout_right,
  class Deleter = std::default_delete<ElementType[]>>
class unique_mdarray {
public:
  using extents_type = Extents;
  using layout_type = Layout;
  using mapping_type = typename layout_type::template mapping<extents_type>;
  using element_type = ElementType;
  using value_type = std::remove_const_t<element_type>;
  using index_type = typename extents_type::index_type;
  using size_type = typename extents_type::size_type;
  using rank_type = typename extents_type::rank_type;

  // It's not a container; all operator[] access is const,
  // even if access to the element is not.
  //
  //using const_reference = std::add_const_t<ElementType>&;
  using reference = ElementType&;

  // Like unique_ptr.
  //
  // We define element_type above.
  // For pointer, see [unique.ptr.runtime.general] and [unique.ptr.single.general] 3.
  using pointer = typename std::unique_ptr<ElementType[], Deleter>::pointer;
  using deleter_type = Deleter;

  // Should this class have data_handle_type too?
  // It only really makes sense if we allow "pointer"
  // to be something other than ElementType*. 
  //
  //using data_handle_type = ElementType*;

  //////////////////////////////////////////////////
  // Constructors and other special member functions
  //////////////////////////////////////////////////

  // Default construction is only permitted
  // if a default-constructed unique_mdarray's mdspan would be valid:
  // that is, if the type is rank zero, has no static extents,
  // or all the static extents are zero.
  // This ensures that the resulting object is valid,
  // specifically that it does not "lie"
  // about the size of its multidimensional index space. 
  unique_mdarray() requires(
      extents_type::rank() == 0 ||
      extents_type::rank_dynamic() != 0 ||
      product_of_static_extents(extents_type{}) == 0
    ) = default;

  unique_mdarray(const unique_mdarray&) = delete;
  unique_mdarray& operator=(const unique_mdarray&) = delete; 
  ~unique_mdarray() = default;

  // Move construction or move assignment is only permitted
  // in the cases where default construction would be valid.
  // This ensures that the moved-from object is valid,
  // specifically that it does not "lie"
  // about the size of its multidimensional index space. 
  unique_mdarray(unique_mdarray&&) = delete;

  unique_mdarray(unique_mdarray&& moved_from) requires(
    (
      extents_type::rank() == 0 ||
      extents_type::rank_dynamic() != 0 ||
      product_of_static_extents(extents_type{}) == 0
    ) &&
    std::is_constructible_v<mapping_type, const extents_type&>
  )
    : ptr_(std::move(moved_from.ptr_))
    , mapping_(std::move(moved_from.mapping_))
  {
    moved_from.mapping_ = mapping_type{extents_type{}};
  }

  unique_mdarray& operator=(unique_mdarray&&) = delete;

  unique_mdarray& operator=(unique_mdarray&& moved_from) requires(
    (
      extents_type::rank() == 0 ||
      extents_type::rank_dynamic() != 0 ||
      product_of_static_extents(extents_type{}) == 0
    ) &&
    std::is_constructible_v<mapping_type, const extents_type&>
  )
  {
    if (&moved_from != this) {
      ptr_ = std::move(moved_from.ptr_);
      mapping_ = std::move(moved_from.mapping_);
      moved_from.mapping_ = mapping_type{extents_type{}};
    }
    return *this;
  }

  // unique_ptr constructors don't permit implicit conversions to pointer.

  // Just for now, I'll leave out all the cases of Deleter
  // being something funny, and the weird constraints on 
  // unique_ptr(type_identity_t<pointer>, d) constructors.
  // I'll just say "Deleter d" for now.

  // The "parent constructor" to which most of the other constructors defer.
  //
  // Specific mappings have required_span_size that is a constant expression
  // if all the extents are static.  In that case, if InputExtent is not dynamic_extent,
  // then we could turn the precondition into a static_assert.
  //
  // If we don't want to use span for the input range,
  // then we would use std::type_identity_t<pointer> data,
  // just as unique_ptr does.
  template<size_t InputExtent>
  unique_mdarray(std::span<element_type, InputExtent> sp, const mapping_type& m, Deleter d)
    : ptr_(sp.data(), d), mapping_(m)
  {
    assert(static_cast<size_t>(m.required_span_size()) <= sp.size());
  }

  template<size_t InputExtent>
  unique_mdarray(std::span<element_type, InputExtent> sp, const extents_type& e, Deleter d)
    requires(
      std::is_constructible_v<mapping_type, const extents_type&>
    )
    : unique_mdarray(sp.data(), mapping_type{e}, d)
  {
    assert(sp.size() >= static_cast<size_t>(mapping_.required_span_size()));
  }

  template<size_t InputExtent>
  unique_mdarray(std::span<element_type, InputExtent> sp, const mapping_type& m)
    requires(
      std::is_nothrow_default_constructible_v<Deleter>
    )
    : unique_mdarray(sp, m, Deleter{})
  {}

  template<size_t InputExtent>
  unique_mdarray(std::span<element_type, InputExtent> sp, const extents_type& e)
    requires(
      std::is_constructible_v<mapping_type, const extents_type&> &&
      std::is_nothrow_default_constructible_v<Deleter>
    )
    : unique_mdarray(sp, e, Deleter{})
  {}

  // This is the analog of the mdspan constructor that takes a list of extents
  // (as things convertible to index_type, generally integers).
  // That constructor doesn't accept an accessor.
  // Analogously, this constructor doesn't accept a deleter.
  template<size_t InputExtent, class... OtherIndexTypes>
  requires(
    std::is_nothrow_default_constructible_v<Deleter> &&
    (std::is_convertible_v<OtherIndexTypes, index_type> && ...) &&
    (std::is_nothrow_constructible_v<index_type, OtherIndexTypes> && ...) &&
    (
      sizeof...(OtherIndexTypes) == extents_type::rank() ||
      sizeof...(OtherIndexTypes) == extents_type::rank_dynamic()
    ) &&
    std::is_constructible_v<mapping_type, const extents_type&>
  )
  unique_mdarray(std::span<element_type, InputExtent> sp, OtherIndexTypes... exts)
    : unique_mdarray(sp, mapping_type{extents_type{std::move(exts)...}}, Deleter{})
  {}

  // unique_ptr(span) is explicit.
  //
  // Construction without extents_type or mapping_type implies
  // construction from extents_type{}.
  template<size_t InputExtent>
  explicit unique_mdarray(std::span<element_type, InputExtent> sp)
    requires (
      std::is_nothrow_default_constructible_v<Deleter> &&
      std::is_constructible_v<mapping_type, const extents_type&>
    )
    : unique_mdarray(sp, mapping_type{extents_type{}}, Deleter{})
  {}

  // unique_ptr(nullptr_t) is NOT explicit.
  // The constraints make the implicit conversion harmless,
  // as the resulting array will have size zero
  // (and therefore won't be accessible anyway).
  unique_mdarray(std::nullptr_t)
  requires (
    std::is_nothrow_default_constructible_v<Deleter> &&
    std::is_constructible_v</* decltype(ptr_) */
      std::unique_ptr<ElementType[], Deleter>,
      std::nullptr_t> &&
    std::is_constructible_v<mapping_type, const extents_type&> &&
    (
      extents_type::rank_dynamic() != 0 ||
      empty(extents_type{})
    )
  )
    : ptr_(nullptr), mapping_{extents_type{}}
  {}

  //////////////////////////////////////////////////
  // Functions adopted directly from unique_ptr
  //////////////////////////////////////////////////

  constexpr deleter_type& get_deleter() { return ptr_.get_deleter(); }
  constexpr const deleter_type& get_deleter() const { return ptr_.get_deleter(); }
  constexpr pointer get() const { return ptr_.get(); }
  constexpr explicit operator bool() const noexcept {
    return bool(ptr_);
  }

  template<class Enable = Deleter>
  friend constexpr void swap(
    unique_mdarray& x,
    unique_mdarray& y,
    std::enable_if_t<std::is_swappable_v<Enable>, void>* = nullptr) noexcept
  {
    using std::swap;
    swap(x.ptr_, y.ptr_);
    swap(x.mapping_, y.mapping_);
  }

  //////////////////////////////////////////////////////////
  // Get a nonowning mdspan that views the elements
  //////////////////////////////////////////////////////////
  constexpr operator
    mdspan<element_type, extents_type, layout_type, default_accessor<element_type>>() const {
    return {ptr_.get(), mapping_};
  }

  //////////////////////////////////////////////////////////
  // release and reset (both constrained, unlike unique_ptr)
  //////////////////////////////////////////////////////////

  // Only permit calling release() if we can ensure
  // the postcondition that extents() has size zero.
  template<class P = pointer> requires(
    std::is_same_v<P, pointer> &&
    (
      extents_type::rank_dynamic() != 0 ||
      impl::empty(extents_type{})
    ) &&
    std::is_constructible_v<mapping_type, const extents_type&>
  )
  constexpr pointer release() noexcept {
    auto p = ptr_.release();
    mapping_ = mapping_type{extents_type{}};
    return p;
  }

  // Only permit calling reset() if we can ensure
  // the postcondition that extents() has size zero.
  template<class P = pointer> requires(
    std::is_same_v<P, pointer> &&
    (
      extents_type::rank_dynamic() != 0 ||
      impl::empty(extents_type{})
    ) &&
    std::is_constructible_v<mapping_type, const extents_type&>
  )
  constexpr void reset() noexcept {
    (void) this->release();
  }

  // reset with a single pointer argument implies
  // that the extents haven't changed.
  // Thus, we don't have to recreate the mapping.
  constexpr void reset(std::type_identity_t<pointer> p) noexcept {
    ptr_.reset(p);
  }

  //////////////////////////////////////////////////////////
  // mdspan-like interface
  //////////////////////////////////////////////////////////

  static constexpr rank_type rank() noexcept {
    return extents_type::rank();
  }
  static constexpr rank_type rank_dynamic() noexcept {
    return extents_type::rank_dynamic();
  }
  static constexpr size_t static_extent(rank_type r) noexcept
  {
    return extents_type::static_extent(r);
  }
  constexpr index_type extent(rank_type r) const noexcept {
    return extents().extent(r);
  }

  // It's not a container; it works like unique_ptr<value_type[]>.
  // Thus, operator[] always returns reference.
  // If element_type is const-qualified, then so its the reference.
  // There's no non-const overload, as a container would have.

  template<class... OtherIndexTypes>
  requires(
    (std::is_convertible_v<OtherIndexTypes, index_type> && ...) &&
    (std::is_nothrow_constructible_v<index_type, OtherIndexTypes> && ...) &&
    sizeof...(OtherIndexTypes) == extents_type::rank()
  )
  reference MDSPAN_ARRAY_ACCESS_OPERATOR
    (OtherIndexTypes... indices) const {
    return ptr_[mapping_(static_cast<index_type>(std::move(indices))...)];
  }

  template<class OtherIndexType>
    constexpr reference
      operator[](std::span<OtherIndexType, rank()> indices) const
  {
    return [this, indices] <size_t... Which> (std::index_sequence<Which...>) -> reference {
      return ptr_[mapping_(static_cast<index_type>(indices[Which])...)];
    } (std::make_index_sequence<extents_type::rank()>());
  }

  template<class OtherIndexType>
    constexpr reference
      operator[](const std::array<OtherIndexType, rank()>& indices) const
  {
    return [this, &indices] <size_t... Which> (std::index_sequence<Which...>) -> reference {
      return ptr_[mapping_(static_cast<index_type>(indices[Which])...)];
    } (std::make_index_sequence<extents_type::rank()>());
  }

  constexpr size_type size() const noexcept {
    return forward_product_of_extents(extents());
  }
  constexpr bool empty() const noexcept {
    return empty(extents());
  }

  constexpr const extents_type& extents() const noexcept {
    return mapping_.extents();
  }
  // We don't include data_handle(), because the name suggests
  // that it could be something other than ElementType*.
  //
  //constexpr const data_handle_type& data_handle() const noexcept {
  //  return ptr_;
  //}
  constexpr const mapping_type& mapping() const noexcept {
    return mapping_;
  }

  static constexpr bool is_always_unique()
    { return mapping_type::is_always_unique(); }
  static constexpr bool is_always_exhaustive()
    { return mapping_type::is_always_exhaustive(); }
  static constexpr bool is_always_strided()
    { return mapping_type::is_always_strided(); }

  constexpr bool is_unique() const
    { return mapping_.is_unique(); }
  constexpr bool is_exhaustive() const
    { return mapping_.is_exhaustive(); }
  constexpr bool is_strided() const
    { return mapping_.is_strided(); }
  constexpr index_type stride(rank_type r) const
    { return mapping_.stride(r); }

private:
  std::unique_ptr<ElementType[], Deleter> ptr_{};
  typename Layout::template mapping<Extents> mapping_{};
};

//
// The analog of make_unique<T[]> needs a mapping instead of a size.
//
template<class ElementType, class Mapping>
constexpr unique_mdarray<ElementType, typename Mapping::extents_type, typename Mapping::layout_type>
  make_unique_mdarray(const Mapping& mapping)
{
  const auto num_elts = mapping.required_span_size();
  auto ptr = std::make_unique<ElementType[]>(num_elts);
  return {std::span<ElementType>{ptr.release(), num_elts}, mapping};
}

} // namespace md
