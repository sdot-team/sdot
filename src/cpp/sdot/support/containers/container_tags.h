#pragma once

#include "../common_macros.h"
#include <type_traits>

namespace sdot {

// ---------------------------------------------------------------------------
// Compile-time tag set carried by containers/views (e.g. TensorView<...,Tags...>).
//
// A tag is an empty type stating a statically-known fact about a container, used
// to specialize behavior at compile time with *zero* runtime cost. Tags are
// orthogonal to the data and to each other; the absence of a tag means "this
// fact is not known", never "false".
//
// All tags live in namespace `container_tags`.
// ---------------------------------------------------------------------------



/// true iff `Tag` appears in `Tags...` (empty pack -> false)
template<class Tag,class... Tags>
constexpr bool contains_tag = ( false || ... || std::is_same_v<Tag,Tags> );

// --- deriving the tag set when a new view is produced ----------------------
// Operations that derive a view (squeeze, row, unsqueeze, ...) can change the
// axes, so a tag may need to be rewritten or dropped. Each derivation passes an
// `Op` descriptor (see the container_ops below) through `transform_tag<Op,Tag>`,
// which yields a `TagList<...>` for that single tag: by default the tag is kept
// unchanged; an empty list drops it; several entries split it. The per-view tag
// pack after the operation is the concatenation over all current tags.

template<class... Tags> struct TagList {};

namespace container_ops {
    /// view derivation that removes axis `Axis` (squeeze / row)
    template<class Axis> struct squeeze {};
    /// view derivation that inserts a size-1 axis at `Axis` (unsqueeze)
    template<class Axis> struct unsqueeze {};
}

/// how a single `Tag` becomes under derivation `Op` — identity by default
template<class Op,class Tag>
struct transform_tag { using list = TagList<Tag>; };

namespace detail {
    template<class... Lists> struct cat_tag_lists { using list = TagList<>; };
    template<class... A> struct cat_tag_lists<TagList<A...>> { using list = TagList<A...>; };
    template<class... A,class... B,class... Rest>
    struct cat_tag_lists<TagList<A...>,TagList<B...>,Rest...> { using list = typename cat_tag_lists<TagList<A...,B...>,Rest...>::list; };

    /// expand a TagList into the template `View<Bound...,Tags...>`
    template<template<class...> class View,class List> struct apply_tag_list;
    template<template<class...> class View,class... Tags> struct apply_tag_list<View,TagList<Tags...>> { using type = View<Tags...>; };
}

/// tag pack (as a TagList) after applying derivation `Op` to the set `Tags...`
template<class Op,class... Tags>
using tags_after = typename detail::cat_tag_lists< typename transform_tag<Op,Tags>::list... >::list;

/// `View<Tags...>` where `Tags...` is the TagList `List` expanded
template<template<class...> class View,class List>
using view_with_tag_list = typename detail::apply_tag_list<View,List>::type;



} // namespace sdot
