# Specifying Sample Spaces of Graphs

The sample space for a model defines the set of all graphs our model assigns probability to.
For example, we may wish to only consider directed graphs with binary edges and no self loops,
which corresponds to graphs of type `spaces.SparseDirectedGraph`. This particular type also
stores graphs as sparse arrays and it suited for large graphs with low edge density.

## Pre-defined Sample Spaces

```@docs
spaces.SparseDirectedGraph
```

## Custom Sample Spaces

A space of graphs is a subtype of `spaces.SampleSpace` that implements the below interface. In addition to the below interface, all sample spaces must have a constructor that takes no arguments are returns a
value corresponding to an empty graph. This is used to initialize samplers, for example. Currently, it is assumed that edges are binary in all SampleSpaces and this interface will need to be augmented to support more edge values.

```@docs
spaces.random_index(::spaces.SampleSpace)
```

```@docs
Base.getindex(::spaces.SparseDirectedGraph, ::Any)
```

```@docs
Base.setindex!(::spaces.SparseDirectedGraph, ::Any, ::Any)
```

```@docs
Base.copy(::spaces.SparseDirectedGraph)
```