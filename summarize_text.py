#!/usr/bin/env python
# coding: utf-8

# In[8]:


import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

# Load the pre-trained T5 model and tokenizer
model_name = 't5-small'  # You can use 't5-base' or 't5-large' for better results
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def summarize_text(text, max_length=150, min_length=50, length_penalty=2.0, num_beams=4):
    # Prepend the "summarize:" prefix to the text
    input_text = "summarize: " + text

    # Tokenize the input text
    inputs = tokenizer.encode(input_text, return_tensors='tf', max_length=512, truncation=True)

    # Generate the summary
    summary_ids = model.generate(
        inputs,
        max_length=max_length,
        min_length=min_length,
        length_penalty=length_penalty,
        num_beams=num_beams,
        early_stopping=True
    )

    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

if __name__ == "__main__":
    # Example text to summarize
    text = """
We develop a physics-informed neural network (PINN) to evaluate closure terms
for turbulence and chemical source terms in the Sandia turbulent non-premixed flames.
The approach relies on temperature, major species and velocity point measurements to
develop closure for the transport of momentum and the thermo-chemical state, through
principal components (PCs). The PCs are derived using principal component analysis
(PCA) implemented on the measured thermo-chemical scalars. With the solution for
the PCs, the number of governing equations and associated closure terms is reduced
relative to the solution of the measured species and temperature. The PINNs are trained
on two flame conditions, the so-called Sandia Flames D and F, and are validated on an
additional flame, Flame E. In addition to the radial and axial spatial coordinates, the
Reynolds number is prescribed as an additional input parameter. A relatively shallow
network attached to PINNs is used to relate the unconditional means of the PCs to
the source terms in their transport equations. The results show that PCs, species,
the mixture fraction and the axial and radial velocity components can adequately be
represented with PINNs compared to experimental statistics. Moreover, PINNs are
able to reconstruct closure terms associated with turbulence and scalars’ transport,
including the averaged PCs’ chemical source term
    """

    # Summarize the text
    summary = summarize_text(text)
    print("Original Text:", text)
    print("\nSummary:", summary)


# In[ ]:




