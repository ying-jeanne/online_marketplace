# Paper Summary: Always Valid Inference

**Full Title:** Always Valid Inference: Continuous Monitoring of A/B Tests  
**Authors:** Ramesh Johari, Pete Koomen, Leonid Pekelis, David Walsh  
**Journal:** Operations Research, 2022, vol. 70, no. 3, pp. 1806–1821  
**Affiliation:** Stanford University & Optimizely, Inc.

---

## Overview

This paper addresses a critical problem in online A/B testing: **how to properly handle "peeking"** — the practice of continuously monitoring test results and stopping early when outcomes appear significant. The authors develop a statistical methodology called **"always valid inference"** that maintains valid statistical guarantees regardless of when experimenters choose to look at the data or stop the test.

---

## The Problem

### Traditional A/B Testing Issues

1. **Fixed-horizon testing**: Classical statistical tests assume you decide the sample size upfront and only look at results once at the end
2. **Peeking invalidates p-values**: Looking at results multiple times and stopping early when p < 0.05 dramatically inflates Type I error (false positive) rates
3. **Industry reality**: In practice, platforms like Optimizely run thousands of A/B tests daily, and experimenters constantly monitor results and make adaptive stopping decisions

### The Dilemma

- **Practitioners want**: Freedom to monitor tests continuously and stop early when they see clear signals
- **Traditional statistics requires**: Pre-commitment to sample size and single analysis
- **The gap**: Standard methods fail when applied to the adaptive, continuous monitoring scenarios common in industry

---

## The Solution: Always Valid Inference

The authors develop a framework that provides **valid confidence sequences** — confidence intervals that remain statistically valid at all times, regardless of:
- How often you look at the data
- When you decide to stop the test
- Whether stopping rules depend on the observed data

### Key Innovation: mSPRT (Mixture Sequential Probability Ratio Test)

The core methodology is based on sequential testing theory, specifically:

1. **Confidence sequences** instead of fixed-time confidence intervals
2. **Mixture priors** over effect sizes to create robust tests
3. **Non-asymptotic guarantees** that work for any sample size

### Mathematical Foundation

The method uses:
- **Sequential probability ratio tests (SPRT)** from Wald (1945)
- **Mixture over alternative hypotheses** to avoid requiring pre-specification of effect size
- **Law of iterated logarithm** for optimal boundary construction
- **Sub-Gaussian mixture techniques** for robustness to heavy-tailed data

---

## Key Results

### 1. **Statistical Validity**

The method guarantees:
- **Type I error control**: False positive rate ≤ α (e.g., 5%) regardless of peeking
- **Always valid p-values**: Valid at any stopping time, not just pre-planned analyses
- **Confidence sequences**: 95% confidence intervals that contain the true effect with 95% probability at all times

### 2. **Practical Performance**

Compared to traditional fixed-horizon testing:
- **Similar power** when running to completion
- **Earlier stopping** when effects are large (potential cost savings)
- **Robustness** to variance misspecification and non-normality
- **No inflation** of false positive rates from continuous monitoring

### 3. **Platform Implementation**

The methodology was implemented in **Optimizely**, a major commercial A/B testing platform:
- Handles **thousands of concurrent experiments**
- Provides **real-time** always-valid p-values and confidence intervals
- Allows **unrestricted peeking** without statistical penalty
- Used by major tech companies for production experiments

---

## Technical Contributions

### 1. **mSPRT for Means**

- Develops mixture sequential tests for comparing means between groups
- Uses a prior over possible effect sizes (typically normal distribution)
- Provides explicit formulas for confidence sequences
- Extends to handle unknown and potentially misspecified variances

### 2. **Variance Estimation**

- Proposes a "plug-in" approach using observed sample variances
- Shows robustness when true variance differs from assumed variance
- Provides theoretical guarantees even with variance misspecification

### 3. **Heavy-Tailed Data**

- Extends method to handle non-normal, heavy-tailed distributions
- Uses sub-Gaussian mixture techniques
- Maintains validity without strict normality assumptions

### 4. **Computational Efficiency**

- Provides closed-form expressions for test statistics
- Enables real-time computation for thousands of simultaneous tests
- Scalable to industry-scale experimentation platforms

---

## Practical Implications

### For Practitioners

1. **Freedom to peek**: Monitor experiments continuously without statistical penalty
2. **Adaptive stopping**: Stop early when results are clear (save time/money) or continue when uncertain
3. **No pre-commitment**: Don't need to specify sample size upfront
4. **Valid inference**: Get correct p-values and confidence intervals at any time

### For Platform Designers

1. **Real-time dashboards**: Can show always-valid statistics in live dashboards
2. **Automated decision-making**: Can build auto-stopping rules with valid error control
3. **Multiple comparisons**: Framework extends to testing multiple variants
4. **Commercial viability**: Proven at scale in production systems

---

## Connections to Related Work

### Sequential Analysis
- Builds on **Wald's SPRT** (1945) and **group sequential methods** (cancer trials)
- Extends classical sequential testing to modern online experimentation contexts

### Multi-Armed Bandits
- Complements bandit algorithms (which optimize for reward)
- Focuses on inference rather than optimization
- Can be combined with bandit approaches for best-of-both-worlds

### Multiple Testing
- Framework can incorporate **false discovery rate (FDR)** control
- Enables valid inference across thousands of simultaneous experiments
- Related to **α-investing** and other online multiple testing methods

---

## Limitations and Extensions

### Limitations Discussed

1. **Assumes independent observations**: Same-user returning creates dependencies (though often acceptable approximation)
2. **Variance misspecification**: Robust to misspecification but not completely assumption-free
3. **Network effects**: Doesn't directly handle experiments with spillover between users

### Future Directions

- **Multi-armed bandit integration**: Combining inference with reward optimization
- **Heterogeneous treatment effects**: Inference for subgroup effects
- **Long-term effects**: Handling delayed conversions and long-term metrics
- **Network experimentation**: Accounting for interference between units

---

## Key Takeaways

1. **Peeking is a real problem**: Continuous monitoring can severely inflate false positive rates in traditional testing

2. **Always valid inference solves it**: The mSPRT methodology provides statistically valid results at any stopping time

3. **Practical and performant**: Implemented at scale in production, with similar power to fixed-horizon tests but greater flexibility

4. **Theoretical rigor**: Built on solid sequential analysis foundations with non-asymptotic guarantees

5. **Industry impact**: Enables the modern practice of continuous experimentation with proper statistical safeguards

---

## Relevance to Your Course (DBA5113)

This paper is highly relevant for:

- **Online experimentation**: Core methodology for digital platform optimization
- **Sequential decision-making**: How to make valid statistical decisions over time
- **Operations research**: Optimizing experiment duration and stopping rules
- **Data-driven decision-making**: Rigorous framework for adaptive business decisions
- **Platform economics**: Practical implementation in major tech platforms

The always-valid inference framework represents a **bridge between classical statistics and modern online experimentation**, enabling the adaptive, continuous monitoring practices that are essential in today's digital economy while maintaining rigorous statistical guarantees.
