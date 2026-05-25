<h1>OOffline Handwritten Arabic Word Recognition using Multi-Stream HMM with Explicit State Duration</h1>

This repository contains the implementation of a hybrid two-stage pipeline for offline unconstrained handwritten Arabic word recognition (OUHAWR).

<b>Overview</b>

The system combines:
<ul>
<item> holistic pre-filtering stage that fuses three local texture descriptors — LPQ⁺, WLD, and LDNP — via Discriminant Correlation Analysis (DCA) with Sufficient Dimensionality Reduction (SDR), classified by an SVM to produce an N-best hypothesis list.</item>
<item>An analytic recognition stage based on Synchronous Multi-stream Hidden Markov Models (SMSHMMs) with explicit state duration modelling using a mixture of Gamma and Laplace distributions, decoded via a fast two-level decoding algorithm.</item>
</ul>
<b>Key Contributions</b>
<ol>
<item>Novel SMSHMM formalism with duration modelling integrated into the decoding process</item>
<item>Closed-form EM update rules for Laplace parameters, mixture coefficients, and stream weights</item>
<item>Four-stream feature encoding over sliding vertical-stride windows</item>
<item>State-of-the-art results on the IFN/ENIT benchmark</item>
</ol>
