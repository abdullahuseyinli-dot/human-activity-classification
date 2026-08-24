# POLIMI-ITW-S feasibility assessment

POLIMI-ITW-S was evaluated as a candidate dataset for the independent temporal
extension. Its shopping-mall videos, person boxes, 2-D skeletons, and sitting,
standing, walking, and running labels match the intended motion-identifiability study.

## Access and release boundary

The official AIRLab page links a request form for academic, non-commercial access:

- Dataset page: <https://airlab.deib.polimi.it/category/project-research-line/r-computer-vision/>
- Request form: <https://forms.office.com/Pages/ResponsePage.aspx?id=K3EXCvNtXUKAjjCd8ope6wktkzydnshFhgFT4ttaEGlUMVpSTVRBUlRKSUpGQVU0UlVCMzhEQlkwQS4u>

Access requires the applicant to supply an accurate affiliation and accept the
provider's release agreement personally. The study does not treat an application as
authorization, copy credentials into scripts, or redistribute provider files.

## Storage assessment

The provider reported the following component sizes when the assessment was recorded
on 24 August 2026:

| Component | Reported size |
| --- | ---: |
| RGB clips | 335 GB |
| Skeletons, boxes, and labels | 39.4 GB |
| Preprocessed arrays and labels | 17.7 GB |
| Complete release | 392.1 GB |

A safe complete acquisition would require a dedicated volume with at least 500 GB
free, leaving room for archive extraction, features, checkpoints, and retained raw
evidence. The dataset was not acquired because the study's available storage budget
was below this requirement. No POLIMI-ITW-S archive, label file, or subject image was
downloaded or evaluated.

## Dataset decision

Okutama-Action was selected for the completed temporal experiment because its provider
train and test archives fit locally and supply the required actions, person boxes,
tracks, continuous frames, synchronized drone views, and an untouched confirmation
partition. The substitution was fixed in
[`VCOCO_V3_EXTERNAL_CUDA_AMENDMENT.md`](VCOCO_V3_EXTERNAL_CUDA_AMENDMENT.md) before
any Okutama target model was fitted.

POLIMI-ITW-S remains suitable for a later shopping-mall replication if authorized
access and adequate storage are available. Such a replication would define its
recording- or track-grouped split before model outcomes are inspected and would retain
the provider ontology, checksums, exclusions, and release terms unchanged.
