# Sampling from ERGMs

Sampling from an ERGM is accomplished by constructing a `sampling.GibbsSampler` object from a
chosen ERGM (an object of type `models.Model`) and passing it to the function `sampling.sample`.

```@docs
sampling.GibbsSampler
```

## Choosing Sampler Parameters

```@docs
sampling.plot_diagnostics
```