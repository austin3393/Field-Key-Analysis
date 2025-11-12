# 🧠 **Field Key Alignment and LLM Output Optimization**

### 🩺 **Overview**
Field keys are internal representations of text field titles from customer EHRs. They help inform the LLM—along with other metadata—about what content to generate for each corresponding text field.  

In the **Retrieval-Augmented Generation (RAG)** pipeline, field keys exert the **strongest influence on LLM output quality** at the earliest stage of processing—when limited metadata exists to provide additional context.  
This early stage is also when **end-user skepticism is highest**, meaning low-quality or irrelevant outputs can significantly undermine trust and adoption within the customer organization.  
Ensuring initial output quality is therefore critical to establishing confidence in the product.  

Our team set out to test the hypothesis that the **quality of LLM outputs** is directly related to the **semantic alignment** between a field key and its corresponding field title.  

---

### 🎯 **Goal**
To develop a **ranking and scoring system** that quantifies the semantic alignment between field keys and field titles, and to use this data to generate improved, more contextually accurate field keys that would enhance LLM output quality.  

---

### ⚙️ **Methodology**
We used the **GPT-4o-mini** model to simulate production conditions and evaluate each field key.  
For each key–title pair, the model generated:  

- 📊 A **semantic alignment score** (0–1 scale)  
- 💬 A **reasoning statement** explaining the score  
- 💡 One or more **suggested improved field keys**  

This method closely mirrored the model’s real-world behavior in production, ensuring our evaluation pipeline reflected how field keys influence outputs in practice.  

We then conducted **A/B testing** on several of the lowest-ranked field keys and their improved counterparts to assess the difference in LLM output quality.  
A **licensed clinical social worker** and a **clinical therapist** independently validated the outputs.  

Their assessments confirmed that field keys with low alignment scores consistently produced low-quality outputs, while improved keys led to more relevant, contextually accurate responses—**validating our alignment-scoring hypothesis and pipeline design.**  


---

## 📂 Project Structure
```
field_keys_analysis.ipynb   # Jupyter Notebook with EDA and scoring
requirements.txt            # Python dependencies
```

---


## ⚙️ Setup

* 1. Clone the repo
* 2. Create and activate a virtual environment
* 3. Install dependencies
with 
```
pip install -r requirements.txt
```


---

### 🚀 **Next Steps**
- 🧩 **Aggregate and analyze alignment reasoning patterns** to identify common causes of poor-quality keys (e.g., ambiguity, missing context, incorrect qualifiers).  
- 🧭 **Integrate insights into implementation configuration playbooks**, guiding future setups across organizations.  
- 🔄 **Scale the pipeline** to systematically refine and standardize existing field keys.  
- 💼 **Present results to executive leadership** for approval to deploy the pipeline organization-wide.  
