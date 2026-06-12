# SMNP: Enhancing LLM-based Method Name Prediction with Code Summarization

This repository contains the implementation, datasets, prompts, evaluation scripts, and experimental resources for our study:

**Understanding before Naming: Enhancing LLM-based Method Name Prediction with Code Summarization**

## Overview

Method Name Prediction (MNP) aims to automatically generate meaningful method names for given method code snippets. Existing LLM-based MNP approaches usually generate method names directly from source code and are commonly evaluated using token-overlap metrics. However, these metrics may not faithfully reflect human judgments of method name quality.

This project investigates whether code summarization can serve as an explicit intermediate understanding step for LLM-based method name prediction. Instead of directly generating method names from code snippets, we explore a summarization-and-refinement strategy, where an LLM first summarizes the functionality of a method and then generates a method name based on the summary.

Based on our findings, we further propose **SMNP**, a summarization-enhanced method name prediction framework.

## Repository Structure

```text
.
├── code/                 # Source code and running scripts
├── data/                 # Experimental datasets
├── figure for prompt/    # Figures used in prompts or prompt design
├── human-eval/           # Human evaluation materials and results
├── llm-eval/             # LLM-based evaluation scripts and results
├── method_name/          # Generated method names and related outputs
├── metric/               # Metric-based evaluation scripts and results
├── requirements.txt      # Required Python packages
└── README.md             # Project description
```



