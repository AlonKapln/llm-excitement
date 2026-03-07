# Research Paper Task

## Your Goal

You are to draft a research article, which we will then review and refine, for the following experiment, in ACL format,
provided in the folder.
You are welcome to use the papers provided for citations, or perhaps, give others which you think might be necessary.

# Our research question / Hypothesis

We Hypothesized that LLMs We hypothesize that LLMs possess interpretable features that specifically encode the reception
of Positive
vs. Negative feedback from the user. We aimed to find how such features affect models responses. Specifically, we used
the features we find
for steering the model for harmful responses ("Jail breaking").

# Literature Review and references

Use the papers we uploaded in the papers_for_citations folder (or any other related papers you know of) to provide
background on the research topic and contextualize our work within it.
you should critically evaluate previous research and identify gaps that our project filled in.

# Methodology

For this part, make sure the draft covers the measures we took to answer our research question. You are very much
encouraged to use the notebook code
we used to handle all the nitty-gritty parts. It should cover the models used, method of obtainig features, the datasets
used and how we steered the model.

## Selecting the features

Make sure to explain why we chose the datasets for the positive, negative and neutral features, how we extracted the
relevant features etc. It's all in the notebook.

## Experiment

Explain about the steering using the found features. write about how we also extracted jailbreak features, and used them
for better results in the steering experiment.
explain how we used the mean activation for each feature and multiplied it by the coefficient.
Write abuot how we steered once with jb + positive, or only jb, etc etc.
you can mention we used the article that talked about the output score and input score to pick a late layer for steering,
and use the top 15 most common features for steering as they did in that paper as well.

# Results and discussion

We tagged the 550 responses we got ourselves
We found that activating the jb features and positive with negative coefficients gives best results, suggesing the
positive features are correlated to how much the model
is sure of his disposition and should "follow his gut"; in contrast to the negative features, where positive strong
coefficiets gave the best results.
Show the results in a clear and concise manner. We attached the graphs showing the refusal rates for each experiment.
Explain that by themselves, the positive and negative features were not enough for steering, but combined with the JB
features we got the best results.
Explain how our hypothesis did not match our findings exactly, but the negative and positive features seem to describe
how much the model is sure and trusts his responses.

# Resources

Our code is in final_project (2).ipynb, and the graphs are in the results folder. The papers for citations are in the
papers_for_citations folder. You can also use any other papers you think are relevant to our work.
The instructions for the final project are in LLM_interp_course_2025a___project_guidelines.pdf.
and the ACL format template is in the latex folder.
If you need me to save the graphs from the notebook in a different format or if you need any other resources, please let
me know.

please output one latex file with the draft of the paper, and make sure to include the bibliography file with the
references. You can use the ACL format template provided in the latex folder as a starting point.