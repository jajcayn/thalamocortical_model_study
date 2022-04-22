# Thalamocortical motif rate model

<p align="center">
  <a href="https://mybinder.org/v2/gh/jajcayn/thalamocortical_model_study/HEAD" target="_blank"><img src="https://mybinder.org/badge_logo.svg"></a>
  <a href="https://www.frontiersin.org/articles/10.3389/fncom.2022.769860" target="_blank"><img src="https://img.shields.io/badge/DOI-10.3389%2Ffncom.2022.769860-lightgrey"></a>
</p>

*Accompanying code for the Jajcay et al. "Cross-frequency slow oscillation-spindle coupling in a biophysically realistic thalamocortical neural mass model", Front. Comput. Neurosci. 16:769860*

## Abstract

Sleep manifests itself by the spontaneous emergence of characteristic oscillatory rhythms, which often time-lock and are implicated in the memory formation. Here, we analyze a neural mass model of the thalamocortical loop of which the cortical node can generate slow oscillations (approx. 1 Hz) while its thalamic component can generate fast sleep spindles of &#963;-band activity (12–15 Hz). We study the dynamics for different coupling strengths between the thalamic and cortical nodes, for different conductance values of the thalamic node’s potassium leak and hyperpolarization-activated cation-nonselective currents, and for different parameter regimes of the cortical node. The latter are: (1) a low activity (DOWN) state with noise-induced, transient excursions into a high activity (UP) state, (2) an adaptation induced slow oscillation limit cycle with alternating UP and DOWN states, and (3) a high activity (UP) state with noise-induced, transient excursions into the low activity (DOWN) state. During UP states, thalamic spindling is abolished or reduced. During DOWN states, the thalamic node generates sleep spindles, which in turn can cause DOWN to UP transitions in the cortical node. Consequently, this leads to spindle-induced UP state transitions in parameter regime (1), thalamic spindles induced in some but not all DOWN states in regime (2), and thalamic spindles following UP to DOWN transitions in regime (3). The spindle-induced &#963;-band activity in the cortical node, however, is typically strongest during the UP state, which follows a DOWN state “window of opportunity” for spindling. When the cortical node is parametrized in regime (3), the model well explains the interactions between slow oscillations and sleep spindles observed experimentally during Non-Rapid Eye Movement sleep. The model is computationally efficient and can be integrated into large-scale modeling frameworks to study spatial aspects like sleep wave propagation.

## How to run

### Locally

Fastest, complete control, requires python et al. already set up.

```bash
git clone https://github.com/jajcayn/thalamocortical_model_study.git
cd thalamocortical_model_study
pip install --upgrade -r requirements.txt
jupyter lab
```

### Docker

Easy to use, only docker required. Runs `jupyter` inside a docker container.

```bash
docker run -p XXXX:8899 ghcr.io/jajcayn/thalamocortical_model_study:v1.0
```

where XXXX is the port number on your machine. Then navigate to localhost:XXXX and should see `jupyterlab`. If you use the same port (`8899`), then you can just click on the link in the terminal and voila.

### Binder

Easiest to use, no setup required, slowest.

Just click here >> [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jajcayn/thalamocortical_model_study/HEAD)
