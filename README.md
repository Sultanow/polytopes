Five-dimensional polytopes play a role in string theory, especially when they are reflexive. It is not trivial to find reflexive polytopes and there exists genetic approaches to finding them. We are implementing a genetic algorithm in Python and search for 5D polytope.

Berglund, He, Heyes, Hirst, Jejjala, and Lukas have demonstrated in their paper ["New Calabi-Yau Manifolds from Genetic Algorithms"](https://arxiv.org/abs/2306.06159) how a genetic approach can be used to identify new reflexive polytopes. The authors have found various polytopes and published their data set on [GitHub](https://github.com/elliheyes/Polytope-Generation/tree/main/Data). We take up this idea and embark on a search for new five-dimensional polytopes.

New polytope (found 2023-12-29) with 7 vertices:

```python
vertices = [[-2, 3, 1, -3, 0],
 [1, -2, -1, 2, 0],
 [4, -4, 1, 2, -2],
 [1, 2, 3, 0, -1],
 [-3, 3, -1, -2, 2],
 [1, 2, 3, -3, -1],
 [-2, -1, -3, 0, 2]]
```

The approach is roughly described in this [paper](https://github.com/Sultanow/polytopes/blob/main/doc/2024_Polytopes.pdf) (work in progress).
