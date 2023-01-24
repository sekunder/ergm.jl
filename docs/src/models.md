# Specifying Exponential Random Graph Models

## Triplet Subgraphs

Many of the pre-defined ERGMs are based on counting triplet subgraphs. In all of the pre-defined models, the sufficient statistics count _induced_ triplet subgraphs. In the case of simple directed graphs, there are 16 isomorphism classes of triplet subgraphs, which are indexed as:

![](assets/models/triplet_subgraphs.png)

## Pre-defined Models

```@docs
models.SimpleModel
```

```@docs
models.DirectedSpatialTripletModel
```

## Custom Models

A model is a subtype `M` of the abstract type `models.Model` that implements the below interface. Models are also stateful, keeping track of one particular graph. This is useful for computing how the sufficient statistics change when making small changes to the underlying graph without having to recompute the statistics from scratch.

```@docs
models.get_state
```

```@docs
models.set_state
```

```@docs
models.get_stats
```

```@docs
models.test_update
```

```@docs
models.apply_update
```