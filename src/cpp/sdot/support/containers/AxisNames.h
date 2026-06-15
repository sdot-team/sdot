#pragma once

#include "container_tags.h" // IWYU pragma: export

namespace sdot {

namespace container_tags {

/// Names the axes of a view, one `Name` per axis in axis order. A `Name` is an empty marker
/// struct (e.g. `struct ax_n {};`). Carried as a tag in `TensorView<...,Tags...>`, it enables
/// indexing an axis by name — `squeeze( ax_n{}, i )` — instead of by position, and is rewritten
/// automatically when axes are derived (a squeeze drops the corresponding name).
///
/// Names need not cover every axis and an axis may stay anonymous; only named axes are
/// addressable by name. Batch axes are just a use of this general mechanism (no separate "batch"
/// concept lives in the tensor).
template<class... Names> struct AxisNames {};

} // namespace container_tags


// --- rewrite AxisNames under a squeeze( Axis ) derivation: drop the Axis-th name -------------

namespace detail {
    /// AxisNames with the `N`-th of `Names...` removed (kept ones accumulated in `Kept`).
    /// `Names...` may cover only the leading (named) axes: squeezing a position beyond them
    /// (a trailing, unnamed/core axis) leaves the names unchanged.
    template<int N,class Kept,class... Names> struct drop_axis_name;

    template<int N,class... Kept,class Head,class... Tail>
    struct drop_axis_name<N,container_tags::AxisNames<Kept...>,Head,Tail...> {
        using type = typename drop_axis_name<N - 1,container_tags::AxisNames<Kept...,Head>,Tail...>::type;
    };
    template<class... Kept,class Head,class... Tail>
    struct drop_axis_name<0,container_tags::AxisNames<Kept...>,Head,Tail...> {
        using type = container_tags::AxisNames<Kept...,Tail...>; // skip Head
    };
    template<int N,class... Kept> // squeezed position is beyond the named axes -> names unchanged
    struct drop_axis_name<N,container_tags::AxisNames<Kept...>> {
        using type = container_tags::AxisNames<Kept...>;
    };
}

template<class Axis,class... Names>
struct transform_tag<container_ops::squeeze<Axis>,container_tags::AxisNames<Names...>> {
    using list = TagList< typename detail::drop_axis_name<Axis::value,container_tags::AxisNames<>,Names...>::type >;
};


// --- resolve a name to its axis position among a view's tag set ------------------------------

namespace detail {
    /// position of `Name` within `Names...`, or -1 if absent.
    template<class Name,class... Names> struct name_index { static constexpr int value = -1; };
    template<class Name,class Head,class... Tail>
    struct name_index<Name,Head,Tail...> {
        static constexpr int rest  = name_index<Name,Tail...>::value;
        static constexpr int value = std::is_same_v<Name,Head> ? 0 : ( rest < 0 ? -1 : rest + 1 );
    };

    /// the AxisNames tag found in a view's `Tags...` (empty AxisNames if none).
    template<class... Tags> struct axis_names_of { using type = container_tags::AxisNames<>; };
    template<class... Names,class... Rest>
    struct axis_names_of<container_tags::AxisNames<Names...>,Rest...> { using type = container_tags::AxisNames<Names...>; };
    template<class Head,class... Rest>
    struct axis_names_of<Head,Rest...> { using type = typename axis_names_of<Rest...>::type; };

    template<class Name,class Names> struct name_index_in;
    template<class Name,class... Names>
    struct name_index_in<Name,container_tags::AxisNames<Names...>> { static constexpr int value = name_index<Name,Names...>::value; };
}

/// compile-time axis position of named axis `Name` among `Tags...` (-1 when `Name` is not a
/// named axis of that set). `is_axis_name` is the boolean companion.
template<class Name,class... Tags>
static constexpr int axis_position_of = detail::name_index_in<Name,typename detail::axis_names_of<Tags...>::type>::value;

template<class Name,class... Tags>
static constexpr bool is_axis_name = axis_position_of<Name,Tags...> >= 0;


} // namespace sdot
