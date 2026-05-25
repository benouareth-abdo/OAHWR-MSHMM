<h1>Offline Handwritten Arabic Word Recognition using Multi-Stream HMM with Explicit State Duration</h1>

This repository contains the implementation of a hybrid two-stage pipeline for offline unconstrained handwritten Arabic word recognition (OUHAWR).

<H2>Overview</H2>

<b>The system combines:</b>
<ul>
<li> holistic pre-filtering stage that fuses three local texture descriptors — LPQ⁺, WLD, and LDNP — via Discriminant Correlation Analysis (DCA) with Sufficient Dimensionality Reduction (SDR), classified by an SVM to produce an N-best hypothesis list.</li>
<li>An analytic recognition stage based on Synchronous Multi-stream Hidden Markov Models (SMSHMMs) with explicit state duration modelling using a mixture of Gamma and Laplace distributions, decoded via a fast two-level decoding algorithm.</li>
</ul>

<b>Key Contributions</b>

<ol>
<li>Novel SMSHMM formalism with duration modelling integrated into the decoding process</li>
<li>Closed-form EM update rules for Laplace parameters, mixture coefficients, and stream weights</li>
<li>Four-stream feature encoding over sliding vertical-stride windows</li>
<li>State-of-the-art results on the IFN/ENIT benchmark</li>
</ol>
