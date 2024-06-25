---
layout: default
title: K-means Clustering from Scratch
---

# Build K-means Clustering from Scratch

## Introduction

K-means clustering is a popular unsupervised machine learning algorithm used to partition a dataset into `K` clusters. Each data point belongs to the cluster with the nearest mean, which serves as a prototype of the cluster.

## Implementation

Below is the implementation of K-means clustering using PyTorch:

```python
import torch

class KMeans():
    def __init__(self, n_clusters, tol=1e-4, max_iter=100):
        self.n_clusters = n_clusters
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = None

    def fit(self, x):
        index = torch.randperm(len(x))[:self.n_clusters]
        self.centroids = x[index]

        for i in range(self.max_iter):
            distances = self._compute_distance(x)
            assignments = torch.argmin(distances, dim=-1)
            new_centroids = torch.stack([x[assignments == i].mean(axis=0) for i in range(self.n_clusters)])
            if torch.all(torch.norm(new_centroids - self.centroids, dim=-1) <= self.tol):
                break
            self.centroids = new_centroids

    def _compute_distance(self, x):
        return torch.cdist(x, self.centroids)

    def predict(self, x):
        return self.fit(x)
```
## Explanation

We randomly select `n_clusters` data points as initial centroids using `torch.randperm`. The indices of these data points will serve as the names of the clusters, with each cluster named according to its index.

```python
index = torch.randperm(len(x))[:self.n_clusters]
self.centroids = x[index]
```
The `_compute_distance` method calculates the distance between each data point and the centroids using `torch.cdist`, which computes the pairwise distances between two sets of vectors. Each datapoint is assigned to the nearest centroid using `torch.argmin` across the last dimension

The `predict` method is a wrapper for the `fit` method, allowing the model to be used to assign clusters to new data points.

## Example Usage
```python
import torch

# Generate some random data
data = torch.randn(100, 2)

# Create an instance of KMeans
kmeans = KMeans(n_clusters=3)

# Fit the model
kmeans.fit(data)

# Print the centroids
print(kmeans.centroids)

# Predict the clusters for the data
clusters = kmeans.predict(data)
print(clusters)
```