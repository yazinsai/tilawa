Frozen training plan, before any target timing observations:

- Use only explicitly split=train, kind=none help-train recordings with no events, repeats, audit exclusion or unsupported label. Exclude an entire speaker if any label puts that speaker in eval/test/heldout.
- Existing 16-token local horizon. Center log emitted-run lengths on the horizon's geometric mean; scale changes leave the observation unchanged.
- Compile each reference word into contiguous vowel nuclei and the intervening consonant groups. Short and lengthened nuclei have separate duration marginals. No lexical/location identity is a model feature.
- Gaussian copula between emitted nuclei of the SAME word, with correlations shared by short/long nucleus pair and single/multiple-consonant group. The independent control uses exactly the same class duration marginals and sets correlations to zero, discarding grouping.
- Missing emitted nuclei are integrated out: the effective Gaussian Markov correlation is the product over intervening syllabic edges. Token placements, insertion paths and word boundaries are marginalized on the reference edit graph. Original deletion charges remain in both partitions.
- Exact transcript-constrained identity-edit posterior supplies weighted single-duration and nucleus-pair training statistics. No hard tracker visit labels. Three deterministic speaker folds choose shrinkage {10,100,1000} by held-partition likelihood; fit final model only after selection.
- Runtime adds only -log(Z_joint/Z_independent) to each local endpoint's existing cost when an existing local run restart is proposed. Both partitions traverse the same retained row graph and movement edges. Score-to-edit units follow the existing tracker convention of 3 nats per edit.
- Fit and freeze before target replay. Target measurements test the premise; they do not select model parameters.
