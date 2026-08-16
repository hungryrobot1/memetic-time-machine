# Memetic Time Machine: Activation Geometry Experiments

Exploring whether neural network activations encode signatures of memetic
fitness — the tendency of phrases to propagate, mutate, and persist in cultural
transmission.

## Context

This repository accompanies the essay *The Memetic Time Machine*, which asks
whether quantitative methods analogous to molecular phylogenetics can be applied
to cultural evolution.

The core hypothesis: if memes are subject to selection pressures that can be
measured empirically (as [MemeTracker](https://snap.stanford.edu/memetracker/)
showed), then neural network embeddings might carry geometric signatures that
correlate with — or predict — memetic potency.

## Data source

> Leskovec, Backstrom, Kleinberg. "Meme-tracking and the Dynamics of the News
> Cycle." ACM SIGKDD, 2009. https://snap.stanford.edu/memetracker/

The dataset tracked 90 million news articles and blog posts during the 2008 US
presidential election, identifying phrase clusters, mutation patterns, and
propagation dynamics. Findings from that paper motivating these experiments:

- Strong memes have heavy-tailed variant distributions (γ < 2 vs. baseline γ ≈ 2.8)
- Propagation follows imitative dynamics with temporal proximity effects
- Volume near peak diverges logarithmically
- ~2.5 hour lag between news media peak and blog peak

## Structure

```
memetic-time-machine/
├── notebooks/
│   └── memetic_geometry_v2.ipynb   # the experiment
├── data/
│   └── memes_curated.json          # curated corpus, committed
└── scripts/
    └── parse_memetracker.py        # regenerate the corpus from the full dump
```

## Running it

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hungryrobot1/memetic-time-machine/blob/main/notebooks/memetic_geometry_v2.ipynb)

The notebook clones this repository for its data, so the badge works without any
setup. Requires a GPU runtime — Pythia-2.8B in fp16 is roughly 5.6 GB, which
fits a T4 comfortably.

<!-- TODO once you have re-run it: state the runtime type used and roughly how
     long a full pass takes. A reader wants to know what they're committing to
     before they provision anything. -->

To regenerate the curated corpus from the full MemeTracker dump (220 MB
compressed, not committed):

```sh
wget http://snap.stanford.edu/memetracker/srcdata/clust-qt08080902w3mfq5.txt.gz
python scripts/parse_memetracker.py clust-qt08080902w3mfq5.txt.gz -o data/memes_curated.json
```

## Experiments

1. **Clustering** — do strong memes (high propagation, many variants) occupy
   geometrically distinct regions of activation space from weak ones?
2. **Genealogical structure** — within a cluster, does cosine similarity between
   variants track position in the mutation tree?
3. **Temporal dynamics** *(planned)* — does geometry correlate with position in
   the meme lifecycle: pre-peak, peak, post-peak?
4. **Meme engineering** *(planned)* — if regularities exist, can activations be
   steered toward potent regions to generate phrases with higher predicted
   fitness?

## Method notes

- **Model:** Pythia-2.8B (EleutherAI), chosen because The Pile includes 2008-era
  content — the model's training window overlaps the period the data is drawn
  from, so the phrases are not wholly out of distribution.
- **Framework:** TransformerLens for activation extraction.
- **Projection:** UMAP, with convex-hull area and eccentricity as the geometric
  summaries per meme family.

## Follow-up worth doing

Re-running this with more recent models and different corpora — the 2008
election data and a 2023-era model were chosen to match, and varying either
independently is its own experiment.

## License

MIT. If you use this, cite the MemeTracker paper as well.
