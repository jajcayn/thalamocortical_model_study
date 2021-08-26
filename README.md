# Thalamocortical motif rate model
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jajcayn/thalamocortical_model_study/HEAD)

*Accompanying code for the Jajcay et al. "Cross-frequency slow oscillation-spindle coupling in a biophysically realistic thalamocortical neural mass model", [journal here]*

## Abstract
Sleep manifests itself by the spontaneous emergence of characteristic oscillatory rhythms, which often time-lock and are implicated in the formation of memory. Here we analyze a mass model of the thalamocortical loop whose cortical node can generate slow oscillations (approx. 1 Hz) while its thalamic component can generate sleep spindles of &#963;-band activity (12-15 Hz). The cortical node consists of one excitatory and one inhibitory population, and slow oscillations emerge due to the activity-dependent adaptation of the former. The thalamic node consists of an excitatory population representing the thalamocortical relay nucleus and an inhibitory population as a proxy of the thalamic reticular nucleus. Sleep spindles emerge in the thalamic node due to the interplay of a potassium leak current, a T-type calcium current, and an anomalous rectifying current, the latter causing the typical waxing and waning profile. We study the dynamics for different coupling strengths between the thalamic and cortical nodes, for different conductance values of the thalamic node's potassium leak and anomalous rectifying currents, and for different parameter regimes of the cortical node. The latter are: (1) a low activity (DOWN) state with noise-induced, transient excursions into a high activity (UP) state, (2) an adaptation induced slow oscillation limit cycle with alternating UP and DOWN states, and (3) a high activity (UP) state with noise-induced, transient excursions into the low activity (DOWN) state. During UP states, thalamic spindling is abolished or reduced. During DOWN states, the thalamic node generates sleep spindles which in turn may cause DOWN to UP transitions in the cortical node. Consequently, this leads to spindle-caused UP state transitions in parameter regime (1), thalamic spindles induced in some but not all DOWN state "window of opportunity" in regime (2), and thalamic spindles following UP to DOWN transitions in regime (3). Spindle-induced &#963;-band activity in the cortical node, however, is typically highest during the UP state, which follows a DOWN state "window of opportunity". When the cortical node is parametrized in regime (3), the model well explains the interactions between slow oscillations and sleep spindles observed experimentally during Non-Rapid Eye Movement sleep. The model is computationally efficient and can be integrated into large-scale modeling frameworks to study of spatial aspects like sleep wave propagation.

## How to run

### Locally
Fastest, complete control, requires python et al. already set up.
```bash
git clone ...
cd ...
pip install --upgrade -r requirements.txt
jupyter lab
```

### Docker
Easy to use, only docker required. Runs `jupyter` inside a docker container.
```bash
docker run -p XXXX:XXXX ...
```
and navigate to localhost:XXXX

### Binder
Easiest to use, no setup required, slowest.

Just click here >> [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jajcayn/thalamocortical_model_study/HEAD)
