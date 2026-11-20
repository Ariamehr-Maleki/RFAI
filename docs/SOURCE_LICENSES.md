# Source attribution and release boundaries

Checked against provider pages on 20 November 2026. This record distinguishes
provider terms from a license for the authors' new software or manuscript.

| Material | Provider-declared license | Source and treatment |
|---|---|---|
| PreScience database | ODC-BY | https://huggingface.co/datasets/allenai/prescience ; obtain the frozen provider inputs separately; derived query identities, feature caches and predictions retain attribution. |
| ogbl-collab database | ODC-BY | https://ogb.stanford.edu/docs/linkprop/#ogbl-collab ; the license is the one under the collab section, not the CC0 license for adjacent datasets. Derived test-source identifiers are linked by hash to the pinned provider split. |
| exHarmony database | CC-BY-4.0 | https://huggingface.co/datasets/Reviewerly/exHarmony ; retain provider/paper attribution and identify our adapted split and derived results. |
| Pinned PreScience comparator source | Apache-2.0, preserved upstream | Keep the bundled upstream LICENSE and notices. Source commit is recorded in the evidence manifests. |
| Citation Style Language styles | CC-BY-SA, preserved upstream | Keep the attribution and license metadata embedded in the CSL files. |
| New research code | No blanket license assigned | Existing public repository is a review release. Authors must choose a general software reuse license before describing it as an open-source software deposit. |
| Manuscript, new figures and narrative | Author-controlled release | Local files prepared for author review; no journal submission or DOI deposit has occurred. |

The proposed review package contains derived results, caches, fitted models,
code, manifests and provenance. It does not bundle the full provider corpora,
pretrained encoder weights, executable environments or the NCN upstream checkout.
Provider licenses are not a blanket license for every included artifact.
File categories and original paths are retained in the joint manifest. A
permanent public release must preserve attribution and the license boundaries
above; no public-deposit URL or DOI should be invented before that release exists.

The original publication/reference text and the API models used in development
are not relicensed by this package. Public dataset accessibility alone does not
establish a license for third-party paper full text.
