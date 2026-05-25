Offline Unconstrained Handwritten Arabic Word Recognition (OUHAWR)
This repository contains the implementation of a hybrid two-stage pipeline for offline unconstrained handwritten Arabic word recognition (OUHAWR).
Overview
<b>The system combines:</b>

A holistic pre-filtering stage that fuses three local texture descriptors — LPQ⁺, WLD, and LDNP — via Discriminant Correlation Analysis (DCA) with Sufficient Dimensionality Reduction (SDR), classified by an SVM to produce an N-best hypothesis list.
An analytic recognition stage based on Synchronous Multi-stream Hidden Markov Models (SMSHMMs) with explicit state duration modelling using a mixture of Gamma and Laplace distributions, decoded via a fast two-level decoding algorithm.

<b>Key Contributions</b>

Novel SMSHMM formalism with duration modelling integrated into the decoding process
Closed-form EM update rules for Laplace parameters, mixture coefficients, and stream weights
Four-stream feature encoding over sliding vertical-stride windows
State-of-the-art results on the IFN/ENIT benchmark
